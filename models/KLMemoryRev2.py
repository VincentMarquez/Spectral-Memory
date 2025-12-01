import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

"""
KLMemoryRev2.py Not used, Optimization (Need to test 12/1/2025)

--------------------------------------------------------------------------
OFFICIAL IMPLEMENTATION
--------------------------------------------------------------------------
Paper: "The Spectrum Remembers: Spectral Memory" (Marquez, 2025)
Author: Vincent Marquez

Copyright (c) 2025 Vincent Marquez
Licensed under the MIT License.
--------------------------------------------------------------------------

Abstract:
This model introduces Spectral Memory, a mechanism that captures the 
evolution of training dynamics. It writes training-trajectory summaries 
into a persistent buffer, extracts dominant modes via Karhunen-Loève 
decomposition, and projects them into memory tokens for global context.

Key Engineering Features:
1. Numerical Stability: CPU-offloaded float64 eigendecomposition 
   for robust execution on Apple Silicon (MPS) and CUDA.
2. Scalability: "Diamond" projection architecture (factorized_projection=True)
   that scales linearly O(K) rather than O(K^2).
3. Reproducibility: A dense "paper reproduction" path matching Eqs. 10–13.

--------------------------------------------------------------------------
CITATION
--------------------------------------------------------------------------
@article{marquez2025spectral,
  title={The Spectrum Remembers: Spectral Memory},
  author={Marquez, Vincent},
  journal={Independent Research},
  year={2025},
  month={November}
}
--------------------------------------------------------------------------
"""

# ================================================================
# K-L MEMORY MODULE
# ================================================================

class KLMemory(nn.Module):
    """
    Karhunen-Loève (K-L) Memory Module.

    Maintains a history buffer of hidden states and performs spectral 
    decomposition to extract dominant temporal patterns.

    Projection Architectures
    ------------------------
    1. REPRODUCTION MODE (factorized_projection=False):
       - Matches Eq. (10–13) in the paper.
       - Flattens all eigencomponents into one dense vector.
       - **CRITICAL:** For reproducing paper results (Table 4), always set 
         factorized_projection=False.
       
    2. SCALABLE MODE (factorized_projection=True) [DEFAULT]:
       - Diamond / factorized projection.
       - O(K) scaling vs O(K^2) for dense.
       - Recommended for new experiments and deeper memory (K > 32).

    IMPORTANT API NOTE:
    All arguments after `memory_tokens` are KEYWORD-ONLY to ensure API stability.
    
    Args:
        d_model (int): Hidden dimension size.
        memory_depth (int): History buffer size (Paper: T=3000).
        n_components (int): Number of eigenmodes to extract (Paper: K=16).
        memory_tokens (int): Number of injected tokens (Paper: M=4).
        factorized_projection (bool): Scalable (True) vs dense repro (False).
        kl_strategy (str): 'empirical' (PCA) or 'kernel' (GP prior).
        tau (float): Length scale for kernelized K-L.
        kernel_type (str): 'exp', 'gauss', or 'matern'.
        detach_kl (bool): Stop gradients through K-L.
    """

    def __init__(
        self,
        d_model: int,
        memory_depth: int = 3000,
        n_components: int = 16,
        memory_tokens: int = 4,
        *,  # <--- KEYWORD-ONLY BARRIER
        factorized_projection: bool = True,
        kl_strategy: str = "empirical",
        tau: float = 64.0,
        kernel_type: str = "exp",
        detach_kl: bool = True,
        use_lambda_scaling: bool = True,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_depth = memory_depth
        self.n_components = n_components
        self.memory_tokens = memory_tokens
        self.factorized_projection = factorized_projection

        # Warn user if they are relying on the implicit default for Reproducibility
        if self.factorized_projection:
            warnings.warn(
                "KLMemory initialized with factorized_projection=True (Default/Scalable). "
                "To reproduce paper results (Table 4), set factorized_projection=False.",
                UserWarning,
                stacklevel=2,
            )

        # K-L Settings
        self.kl_strategy = kl_strategy.lower()
        if self.kl_strategy not in ["empirical", "kernel"]:
            raise ValueError(f"Invalid kl_strategy '{self.kl_strategy}'. "
                             "Must be 'empirical' or 'kernel'.")

        self.tau = tau
        self.kernel_type = kernel_type.lower()
        # Validated kernel types. To add new kernels, implement in _kernel_kl.
        if self.kernel_type not in ["exp", "gauss", "matern"]:
            raise ValueError(f"Invalid kernel_type '{self.kernel_type}'. "
                             "Must be 'exp', 'gauss', or 'matern'.")

        self.detach_kl = detach_kl
        self.use_lambda_scaling = use_lambda_scaling

        # ============================================================
        # PROJECTION LAYERS
        # ============================================================
        if self.factorized_projection:
            # --- SCALABLE / DIAMOND ARCHITECTURE ---
            # 1) Shared component encoder: [K, d] -> [K, d]
            self.component_encoder = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(d_model * 2, d_model),
            )
            # 2) Compressor / mixer along component axis
            self.compressor = nn.Linear(n_components, memory_tokens)

        else:
            # --- DENSE REPRODUCTION ARCHITECTURE (Paper Exact) ---
            # Flatten: [K, d] -> [K·d] -> MLP -> [M·d]
            kd = n_components * d_model
            md = memory_tokens * d_model

            self.projection_mlp = nn.Sequential(
                nn.Linear(kd, 2 * kd),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(2 * kd, md),
            )

        self.norm = nn.LayerNorm(d_model)

        # History buffer (T, d_model) – persistent, not trainable
        self.register_buffer("_history", torch.zeros(0, d_model))

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------
    def reset(self):
        """Clears the memory history buffer."""
        device = self._history.device
        self._history = torch.zeros(0, self.d_model, device=device)

    def append(self, h_states: torch.Tensor):
        """
        Appends pooled hidden states to history.
        Args: h_states: Tensor of shape (N, d_model) or (d_model,)
        """
        if h_states.numel() == 0: return

        h_states = h_states.detach()
        if h_states.dim() == 1: h_states = h_states.unsqueeze(0)

        self._history = torch.cat([self._history, h_states], dim=0)

        # FIFO eviction
        if self._history.shape[0] > self.memory_depth:
            self._history = self._history[-self.memory_depth :]

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def forward(self, B: int) -> torch.Tensor:
        """
        Generates memory tokens for a batch of size B.
        Returns: mem_tokens: Tensor of shape [M, B, d_model]
        """
        T = self._history.shape[0]
        device = self._history.device

        # Cold Start: not enough history -> zero memory
        if T < max(4, self.n_components):
            return torch.zeros(
                self.memory_tokens, B, self.d_model, device=device
            )

        # 1. Compute Eigencomponents (K-L) on CPU
        C_kl = self._compute_kl_components(self._history)  # (K_eff, d)

        if self.detach_kl: C_kl = C_kl.detach()

        # 2. Pad to fixed n_components (Crash Prevention)
        K_eff = C_kl.shape[0]
        if K_eff < self.n_components:
            pad = torch.zeros(self.n_components - K_eff, self.d_model, device=device, dtype=C_kl.dtype)
            C_kl = torch.cat([C_kl, pad], dim=0)
        elif K_eff > self.n_components:
            C_kl = C_kl[: self.n_components]

        # 3. Projection
        if self.factorized_projection:
            mem = self._forward_scalable(C_kl)
        else:
            mem = self._forward_dense(C_kl)

        # 4. Norm & Broadcast
        mem = self.norm(mem)                        # [M, d]
        mem = mem.unsqueeze(1).expand(-1, B, -1)    # [M, B, d]
        return mem

    # ------------------------------------------------------------------
    # Internal Projections
    # ------------------------------------------------------------------
    def _forward_scalable(self, C: torch.Tensor) -> torch.Tensor:
        """Diamond Path (Scalable). O(K) complexity."""
        encoded = self.component_encoder(C)   # [K, d]
        mixed = self.compressor(encoded.T)    # [d, M]
        return mixed.T                        # [M, d]

    def _forward_dense(self, C: torch.Tensor) -> torch.Tensor:
        """Dense Path (Paper Reproduction). O(K^2) complexity."""
        c_flat = C.reshape(-1)                # [K·d]
        m_flat = self.projection_mlp(c_flat)  # [M·d]
        return m_flat.view(self.memory_tokens, self.d_model)

    # ------------------------------------------------------------------
    # K-L Decompositions (CPU-only, float64)
    # ------------------------------------------------------------------
    def _compute_kl_components(self, H: torch.Tensor) -> torch.Tensor:
        """Orchestrates K-L on CPU to avoid MPS float64 limits."""
        orig_device = H.device
        orig_dtype = H.dtype
        H_cpu = H.detach().to(device="cpu", dtype=orig_dtype)

        if self.kl_strategy == "empirical":
            C_cpu = self._empirical_kl(H_cpu)
        else:
            C_cpu = self._kernel_kl(H_cpu)

        return C_cpu.to(device=orig_device, dtype=orig_dtype)

    def _empirical_kl(self, H: torch.Tensor) -> torch.Tensor:
        """Strategy A: Empirical Discrete K-L (time-axis PCA)."""
        T, d = H.shape
        Hc = H - H.mean(dim=0, keepdim=True)
        Hc64 = Hc.to(torch.float64) # Precision cast

        C = (Hc64 @ Hc64.T) / max(T, 1)
        evals, evecs = torch.linalg.eigh(C)

        idx = torch.argsort(evals, descending=True)[:min(self.n_components, T)]
        lams = torch.clamp(evals[idx], min=0.0)
        phi = evecs[:, idx] / (evecs[:, idx].norm(dim=0, keepdim=True) + 1e-12)

        coeffs = phi.T @ Hc64
        if self.use_lambda_scaling:
            coeffs = torch.sqrt(lams + 1e-12)[:, None] * coeffs
        return coeffs

    def _kernel_kl(self, H: torch.Tensor) -> torch.Tensor:
        """Strategy B: Kernelized K-L (GP prior)."""
        T, d = H.shape
        Hc64 = (H - H.mean(dim=0, keepdim=True)).to(torch.float64)

        t = torch.linspace(0.0, 1.0, steps=T, device=Hc64.device, dtype=torch.float64)
        dt = 1.0 / max(T - 1, 1)
        tau_eff = self.tau / max(T, 1)

        diff = t.unsqueeze(1) - t.unsqueeze(0)
        r = diff.abs()

        if self.kernel_type == "gauss":
            K = torch.exp(- (r ** 2) / (2.0 * (tau_eff ** 2) + 1e-12))
        elif self.kernel_type == "matern":
            K = (1.0 + r / (tau_eff + 1e-12)) * torch.exp(-r / (tau_eff + 1e-12))
        else:
            K = torch.exp(-r / (tau_eff + 1e-12))

        K = (self.tau * dt) * K
        K = 0.5 * (K + K.T) + (1e-6 * torch.eye(T, dtype=torch.float64, device=Hc64.device))

        try:
            evals, evecs = torch.linalg.eigh(K)
        except RuntimeError:
            U, S, _ = torch.linalg.svd(K, full_matrices=False)
            evals, evecs = S, U

        idx = torch.argsort(evals, descending=True)[:min(self.n_components, T)]
        lams = torch.clamp(evals[idx], min=0.0)
        phi = evecs[:, idx] / (evecs[:, idx].norm(dim=0, keepdim=True) + 1e-12)

        coeffs = phi.T @ Hc64
        if self.use_lambda_scaling:
            coeffs = torch.sqrt(lams + 1e-12)[:, None] * coeffs
        return coeffs


# ================================================================
#  PLATINUM TRANSFORMER MODEL
# ================================================================

class Model(nn.Module):
    """
    Main Transformer Model with KLMemory Injection.

    Features:
    - Channel Independence (CI)
    - Spectral Covariance Memory (KLMemory)
    - Attention Pooling: Learns "what to remember"
    - Flatten-head decoder (assumes seq_len is fixed at configs.seq_len)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # RESTORED ATTRIBUTES (For external compatibility/logging)
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # 1. Channel-Independent Embedding
        self.enc_embedding = nn.Linear(1, configs.d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, configs.seq_len, configs.d_model)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)

        # 2. KL Memory
        factorized_flag = getattr(configs, "factorized_projection", True)
        
        self.memory = KLMemory(
            d_model=configs.d_model,
            memory_depth=getattr(configs, "memory_depth", 3000),
            n_components=getattr(configs, "n_components", 16),
            memory_tokens=getattr(configs, "memory_tokens", 4),
            # KEYWORD ONLY ARGUMENTS START HERE:
            factorized_projection=factorized_flag,
            kl_strategy=getattr(configs, "kl_strategy", "empirical"),
            tau=getattr(configs, "kl_tau", 64.0),
            kernel_type=getattr(configs, "kl_kernel", "exp"),
            detach_kl=getattr(configs, "kl_detach", True),
            use_lambda_scaling=getattr(configs, "kl_lambda_scale", True),
            mlp_dropout=getattr(configs, "kl_mlp_dropout", configs.dropout),
        )

        # 3. Attention Pooling
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
        # NOTE: This assumes runtime L == configs.seq_len.
        # If L changes at inference, this will crash.
        self.head = nn.Linear(self.seq_len * configs.d_model, self.pred_len)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, L, C]

        # ------------------------------------------------------------
        # 1. Parameter-Free Normalization (RevIN-like)
        # ------------------------------------------------------------
        # Rationale: Addresses Non-Stationarity (Distribution Shift).
        # We normalize the input statistics (mean/var) to zero-mean/unit-var
        # so the model learns relative dynamics, not absolute scale.
        #
        # CRITICAL IMPLEMENTATION DETAIL:
        # Both 'means' and 'stdev' are DETACHED.
        # This treats them as fixed input statistics, not learnable parameters.
        # This prevents the model from "cheating" by optimizing the normalization
        # to lower the loss without actually learning the temporal structure.
        # ------------------------------------------------------------
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach() # Detached to enforce fixed-statistic normalization
        x_enc /= stdev

        # --- 2. Channel Independence ---
        B, L, C = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).reshape(B * C, L, 1)

        # --- 3. Embedding ---
        enc_out = self.enc_embedding(x_enc)                  # [B*C, L, D]
        enc_out = enc_out + self.pos_encoding[:, :L, :]
        enc_out = self.dropout(enc_out)
        enc_out = enc_out.permute(1, 0, 2)                   # [L, B*C, D]

        # --- 4. Memory Injection ---
        mem_tokens = self.memory(B * C)                      # [M, B*C, D]
        enc_input = torch.cat([mem_tokens, enc_out], dim=0)  # [M+L, B*C, D]

        # --- 5. Transformer Encoder ---
        enc_output = self.encoder(enc_input)                 # [M+L, B*C, D]
        out = enc_output[self.memory.memory_tokens:]         # [L, B*C, D]

        # --- 6. Memory Update (Training Only) ---
        if self.training:
            pool_weights = F.softmax(self.pooling_layer(out), dim=0)
            batch_summary = (out * pool_weights).sum(dim=0)
            step_context = batch_summary.mean(dim=0, keepdim=True)
            self.memory.append(step_context)

        # --- 7. Decode (Flatten Head) ---
        out_flat = out.permute(1, 0, 2).reshape(B * C, -1)
        forecast = self.head(out_flat)
        forecast = forecast.reshape(B, C, -1).permute(0, 2, 1)

        # --- 8. Denormalize ---
        forecast = forecast * stdev + means
        return forecast

    def reset_memory(self):
        """Reset KLMemory state between splits / runs."""
        if hasattr(self, "memory"):
            self.memory.reset()


# ================================================================
#  USAGE DEMO
# ================================================================
if __name__ == "__main__":
    print("Initializing Spectral Memory (Official Implementation)...")

    class Config:
        seq_len = 96
        pred_len = 96
        d_model = 64
        n_heads = 4
        d_ff = 256
        e_layers = 2
        dropout = 0.1
        enc_in = 7  # Restored
        dec_in = 7
        c_out = 7   # Restored
        
        # Memory Params
        memory_depth = 100
        n_components = 16
        memory_tokens = 4
        factorized_projection = True
        
        # K-L params
        kl_strategy = "empirical"
        kl_tau = 64.0
        kl_kernel = "exp"
        kl_detach = True
        kl_lambda_scale = True
        kl_mlp_dropout = 0.1

    cfg = Config()
    model = Model(cfg)

    x = torch.randn(32, 96, 7)  # [B,L,C]

    # Warm up memory
    model.train()
    print("Warming up memory buffer...")
    for i in range(5):
        _ = model(x, None, None, None)

    # Inference
    model.eval()
    with torch.no_grad():
        out = model(x, None, None, None)

    print(f"\nSuccess! Output shape: {out.shape}")
    print(f"Memory used: {model.memory._history.shape[0]} / {cfg.memory_depth}")
