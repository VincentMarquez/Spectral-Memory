import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# K-L MEMORY (WITH CPU FALLBACK FOR EIGENDECOMP)
# ================================================================

class KLMemory(nn.Module):
    """
    Karhunen-Loève Memory Tokens 

    - Maintains a history buffer of hidden states: H ∈ R^{T × d_model}
    - Applies classical K-L decomposition (Empirical or Kernelized GP)
    - Then passes K-L components through a learnable 2-layer MLP f_θ
      to produce M memory tokens.

    Key paper features included:
        * Mean-centering of H before covariance (empirical path)
        * √λ eigenvalue scaling of K-L components
        * Gradient stop through K-L (only MLP is trainable)
        * Empirical vs Kernelized K-L (select via `kl_strategy`)
        * 2-layer MLP projection f_θ with GELU + Dropout
        * CPU-only eigendecomposition in float64 (MPS-safe)

    Public API:
        - reset(): clears history
        - append(h_states): appends pooled hidden states (N, d_model)
        - forward(B): returns memory tokens [Mem, B, d_model]
    """

    def __init__(
        self,
        d_model: int,
        memory_depth: int = 3000,
        n_components: int = 16,
        memory_tokens: int = 8,
        kl_strategy: str = "empirical",   # "empirical" or "kernel"
        tau: float = 64.0,                # GP timescale for kernelized K-L
        kernel_type: str = "exp",         # "exp", "gauss", or "matern"
        detach_kl: bool = True,           # stop gradients through K-L
        use_lambda_scaling: bool = True,  # multiply coeffs by sqrt(λ_k)
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_depth = memory_depth
        self.n_components = n_components
        self.memory_tokens = memory_tokens

        # K-L settings
        self.kl_strategy = kl_strategy.lower()
        assert self.kl_strategy in ["empirical", "kernel"], \
            "kl_strategy must be 'empirical' or 'kernel'"

        self.tau = tau
        self.kernel_type = kernel_type.lower()
        self.detach_kl = detach_kl
        self.use_lambda_scaling = use_lambda_scaling

        # ---- Learnable projection f_θ: K-L components -> memory tokens ----
        # Flatten C_KL ∈ R^{K×d} -> R^{K*d}, then:
        #  Linear(K*d, 2K*d) -> GELU -> Dropout -> Linear(2K*d, M*d)
        kd = n_components * d_model
        md = memory_tokens * d_model

        self.projection_mlp = nn.Sequential(
            nn.Linear(kd, 2 * kd),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(2 * kd, md),
        )

        # LayerNorm over token dimension
        self.norm = nn.LayerNorm(d_model)

        # History buffer: (T, d_model)
        self.register_buffer("_history", torch.zeros(0, d_model))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self):
        """Reset the history buffer."""
        device = self._history.device
        self._history = torch.zeros(0, self.d_model, device=device)

    def append(self, h_states: torch.Tensor):
        """
        Append pooled hidden states to the history buffer.

        Args:
            h_states: Tensor of shape (N, d_model) or (d_model,)
                      Typically this is a pooled summary per (batch*channel).
        """
        if h_states.numel() == 0:
            return
        # Do not backprop through history writes
        h_states = h_states.detach()
        if h_states.dim() == 1:
            h_states = h_states.unsqueeze(0)

        self._history = torch.cat([self._history, h_states], dim=0)
        if self._history.shape[0] > self.memory_depth:
            self._history = self._history[-self.memory_depth :]

    def forward(self, B: int) -> torch.Tensor:
        """
        Produce memory tokens for a batch of size B.

        Returns:
            memory_tokens: Tensor of shape (Mem, B, d_model)
        """
        T = self._history.shape[0]
        device = self._history.device

        # Not enough history yet -> zero memory
        if T < max(4, self.n_components):
            return torch.zeros(
                self.memory_tokens, B, self.d_model, device=device
            )

        # ---- Classical K-L decomposition (offloaded to CPU) ----
        C_kl = self._compute_kl_components(self._history)   # (K_eff, d_model)

        # Optionally stop gradients through K-L (only MLP learns)
        if self.detach_kl:
            C_kl = C_kl.detach()

        # ---- Learnable projection f_θ (MLP) -> memory tokens ----
        K_eff = C_kl.shape[0]  # could be < n_components if T small
        kd_eff = K_eff * self.d_model

        c_flat = C_kl.reshape(-1)  # (K_eff * d,)

        # If history shorter than n_components, pad so Linear dims still match
        target_kd = self.n_components * self.d_model
        if kd_eff < target_kd:
            pad_size = target_kd - kd_eff
            pad = torch.zeros(pad_size, device=c_flat.device, dtype=c_flat.dtype)
            c_flat = torch.cat([c_flat, pad], dim=0)

        m_flat = self.projection_mlp(c_flat)  # (M*d,)
        mem = m_flat.view(self.memory_tokens, self.d_model)  # (M, d)

        mem = self.norm(mem)  # LayerNorm over d_model

        # Broadcast to batch: (Mem, B, D)
        mem = mem.unsqueeze(1).expand(-1, B, -1)
        return mem

    # ------------------------------------------------------------------
    # Internal helpers: K-L decompositions (CPU-only)
    # ------------------------------------------------------------------
    def _compute_kl_components(self, H: torch.Tensor) -> torch.Tensor:
        """
        Compute classical K-L components C_KL ∈ R^{K×d_model}.

        Important:
            - All heavy K-L math is done on CPU in float64.
            - This avoids MPS float64 limitations.
        """
        orig_device = H.device
        orig_dtype = H.dtype

        # Move a detached copy to CPU for float64 eigendecomposition
        H_cpu = H.detach().to(device="cpu", dtype=orig_dtype)

        if self.kl_strategy == "empirical":
            C_cpu = self._empirical_kl(H_cpu)   # (K_eff, d_model), float64
        else:
            C_cpu = self._kernel_kl(H_cpu)      # (K_eff, d_model), float64

        # Move back to original device and dtype (e.g., MPS float32)
        C = C_cpu.to(device=orig_device, dtype=orig_dtype)
        return C

    def _empirical_kl(self, H: torch.Tensor) -> torch.Tensor:
        """
        Strategy A: Empirical Discrete K-L (time-axis PCA).

        Steps:
            1. Mean-center H across time
            2. Compute covariance C = (H_c @ H_c^T) / T
            3. Eigendecompose C in float64
            4. Project onto top-K temporal modes
            5. Optionally scale by sqrt(λ_k)
        """
        # H is on CPU here
        T, d = H.shape

        # Step 1: Center the history (time-axis mean)
        Hc = H - H.mean(dim=0, keepdim=True)  # (T, d)

        # Step 2: time-axis empirical covariance in float64
        Hc64 = Hc.to(torch.float64)
        C = (Hc64 @ Hc64.T) / max(T, 1)  # (T, T), float64

        # Step 3: Eigendecomposition (ascending order)
        evals, evecs = torch.linalg.eigh(C)  # float64

        # Step 4: Select top-K modes
        idx = torch.argsort(evals, descending=True)
        K_eff = min(self.n_components, T)
        idx = idx[:K_eff]
        lams = torch.clamp(evals[idx], min=0.0)  # (K_eff,)
        phi = evecs[:, idx]                      # (T, K_eff)

        # Normalize eigenvectors
        phi = phi / (phi.norm(dim=0, keepdim=True) + 1e-12)

        # Project history onto eigenmodes: coeffs = phi^T Hc
        coeffs = phi.T @ Hc64  # (K_eff, d), float64

        if self.use_lambda_scaling:
            # Scale by sqrt(eigenvalues) as in continuous K-L
            coeffs = torch.sqrt(lams + 1e-12)[:, None] * coeffs  # (K_eff, d)

        return coeffs  # CPU, float64

    def _kernel_kl(self, H: torch.Tensor) -> torch.Tensor:
        """
        Strategy B: Kernelized Time-Axis K-L with GP prior.

        - Construct temporal kernel K_τ(i,j) with τ / T scaling
        - Eigendecompose K in float64
        - Project history onto eigenmodes
        - Scale by sqrt(λ_k)
        """
        # H is on CPU here
        T, d = H.shape

        # (Optional) mean-centering for kernel path as well
        Hc = H - H.mean(dim=0, keepdim=True)
        H64 = Hc.to(torch.float64)

        # Construct normalized temporal grid t ∈ [0, 1]
        t = torch.linspace(0.0, 1.0, steps=T, device=H64.device, dtype=torch.float64)
        dt = 1.0 / max(T - 1, 1)

        # Effective timescale (τ / T)
        tau_eff = self.tau / max(T, 1)

        # Pairwise distances |t_i - t_j|
        diff = t.unsqueeze(1) - t.unsqueeze(0)
        r = diff.abs()  # (T, T), float64

        if self.kernel_type == "gauss":
            K = torch.exp(- (r ** 2) / (2.0 * (tau_eff ** 2) + 1e-12))
        elif self.kernel_type == "matern":
            # simple Matérn-like: (1 + r/τ) * exp(-r/τ)
            K = (1.0 + r / (tau_eff + 1e-12)) * torch.exp(-r / (tau_eff + 1e-12))
        else:  # "exp" kernel
            K = torch.exp(-r / (tau_eff + 1e-12))

        # Measure correction (Riemann scaling): ≈ τ * dt
        K = (self.tau * dt) * K

        # Enforce symmetry and add jitter
        K = 0.5 * (K + K.T)
        eps = 1e-8 if T <= 2048 else 1e-6
        K = K + eps * torch.eye(T, dtype=torch.float64, device=H64.device)

        # Eigendecomposition (with SVD fallback if needed)
        try:
            evals, evecs = torch.linalg.eigh(K)
        except RuntimeError:
            U, S, _ = torch.linalg.svd(K, full_matrices=False)
            evals, evecs = S, U

        # Select top-K modes
        idx = torch.argsort(evals, descending=True)
        K_eff = min(self.n_components, T)
        idx = idx[:K_eff]
        lams = torch.clamp(evals[idx], min=0.0)        # (K_eff,)
        phi = evecs[:, idx]                            # (T, K_eff)
        phi = phi / (phi.norm(dim=0, keepdim=True) + 1e-12)

        # Project history onto eigenmodes
        coeffs = phi.T @ H64  # (K_eff, d), float64

        # Scale by sqrt(λ_k)
        if self.use_lambda_scaling:
            coeffs = torch.sqrt(lams + 1e-12)[:, None] * coeffs

        return coeffs  # CPU, float64

# ================================================================
#  PLATINUM MODEL 
# ================================================================

class Model(nn.Module):
    """
    KLMemory Transformer 

    Features:
    - Channel Independence (CI)
    - Flatten Head Decoder
    - Spectral Covariance Memory (KLMemory with MLP projection)
    - Attention Pooling: Learns "What to Remember" instead of averaging.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        # 1. Channel Independence Embedding
        self.enc_embedding = nn.Linear(1, configs.d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, configs.seq_len, configs.d_model)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)

        # ---- KL Memory ----
        kl_strategy = getattr(configs, "kl_strategy", "empirical")
        kl_tau = getattr(configs, "kl_tau", 64.0)
        kl_kernel = getattr(configs, "kl_kernel", "exp")
        kl_detach = getattr(configs, "kl_detach", True)
        kl_lambda_scale = getattr(configs, "kl_lambda_scale", True)
        kl_mlp_dropout = getattr(configs, "kl_mlp_dropout", configs.dropout)

        self.memory = KLMemory(
            d_model=configs.d_model,
            memory_depth=getattr(configs, "memory_depth", 3000),
            n_components=getattr(configs, "n_components", 16),
            memory_tokens=getattr(configs, "memory_tokens", 4),
            kl_strategy=kl_strategy,
            tau=kl_tau,
            kernel_type=kl_kernel,
            detach_kl=kl_detach,
            use_lambda_scaling=kl_lambda_scale,
            mlp_dropout=kl_mlp_dropout,
        )

        # 3. Attention Pooling Layer (for smart memory summaries)
        self.pooling_layer = nn.Linear(configs.d_model, 1)

        # 4. Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=configs.e_layers
        )

        # 5. Decoder (Flatten Head)
        self.head = nn.Linear(self.seq_len * configs.d_model, self.pred_len)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # --- RevIN-like Normalization (per-batch, per-channel) ---
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        # --- Channel Independence Reshape ---
        # [B, L, C] -> [B*C, L, 1]
        B, L, C = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).reshape(B * C, L, 1)

        # --- Embedding ---
        enc_out = self.enc_embedding(x_enc)          # [B*C, L, D]
        enc_out = enc_out + self.pos_encoding[:, :L, :]
        enc_out = self.dropout(enc_out)
        enc_out = enc_out.permute(1, 0, 2)           # [L, B*C, D]

        # --- Memory Injection ---
        # Fetch global context memory tokens
        mem_tokens = self.memory(B * C)              # [Mem, B*C, D]

        # Concatenate memory tokens + local sequence
        enc_input = torch.cat([mem_tokens, enc_out], dim=0)  # [Mem+L, B*C, D]

        # --- Transformer Encoder ---
        enc_output = self.encoder(enc_input)

        # Slice out the sequence part (discard memory outputs)
        out = enc_output[self.memory.memory_tokens:]  # [L, B*C, D]

        # --- Smart Memory Update (Attention Pooling) ---
        if self.training:
            # (L, B*C, D) -> (L, B*C, 1) -> softmax over time dimension L
            pool_weights = F.softmax(self.pooling_layer(out), dim=0)
            # Weighted sum over time: (L, B*C, D) * (L, B*C, 1) -> (B*C, D)
            batch_summary = (out * pool_weights).sum(dim=0)   # (B*C, D)

            # Collapse (B*C) → single global step context: (1, D)
            step_context = batch_summary.mean(dim=0, keepdim=True)
            self.memory.append(step_context)

        # --- Decoding ---
        # [L, B*C, D] -> [B*C, L*D]
        out_flat = out.permute(1, 0, 2).reshape(B * C, -1)
        forecast = self.head(out_flat)               # [B*C, Pred]

        # --- Reshape & Denormalize ---
        forecast = forecast.reshape(B, C, -1).permute(0, 2, 1)  # [B, Pred, C]
        forecast = forecast * stdev + means
        return forecast

    def reset_memory(self):
        """
        Reset KLMemory state between phases (train/val/test)
        or between independent runs.
        """
        if hasattr(self, "memory"):
            self.memory.reset()
