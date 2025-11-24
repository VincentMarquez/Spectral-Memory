
---

# Vincent Marquez Spectral Memory (VMSM)

### **A New Class of General Spectral Memory Architecture for Sequence Models**

**Vincent Marquez Spectral Memory (VMSM)** is a new memory mechanism designed for long-range sequence modeling.
It introduces a **general spectral memory architecture** that compresses historical hidden states using online Karhunen–Loève decomposition and transforms the dominant spectral modes into **learnable memory tokens**.

VMSM works as a plug-in module inside Transformers, SSMs, RNNs, or any encoder stack.

---

# 🔍 What Makes VMSM a New Memory Class?

### **1. Spectral Memory Tokens (SMTs)**

VMSM introduces a new memory object: **Spectral Memory Tokens**, derived from the dominant eigenmodes of the hidden-state history.

These are **not**:

* attention keys
* recurrent hidden states
* SSM state vectors

Instead, they are **spectral modes shaped into learnable memory tokens.**

---

### **2. Full Spectral Memory Pipeline**

VMSM is **not PCA**, **not SVD**, and **not a low-rank trick**.
It is a complete architectural memory mechanism:

```
Historical Hidden States → Covariance Kernel
→ KL Decomposition → Top Spectral Modes
→ Learnable Mixer → Memory Tokens → Reinjection
```

No prior architecture uses this pipeline.

---

### **3. Architecture-Agnostic**

VMSM slots into any sequence model:

* Transformers (Autoformer, PatchTST, Informer, etc.)
* SSMs (S4, S5, Mamba)
* RNN/GRU/LSTM stacks
* Hybrid models
* Custom research architectures

It operates **independently** of attention, recurrence, or SSM update rules.

---

### **4. Long-Range Memory With Spectral Stability**

VMSM stores the most persistent spectral patterns, enabling:

* strong long-range memory
* noise suppression
* stable context retention
* significant memory compression (**O(T·d) → O(k·d)**)
* improved performance even in shallow models

Ideal for long-term forecasting and long-context modeling.

---

# 📦 Why Use VMSM?

* **Consistent ETTh1 improvements** with minimal architectural cost
* **Drop-in module** — no redesign of the backbone
* **Interpretable memory** (eigenmodes correspond to real temporal patterns)
* **Works on consumer hardware (CPU/MPS)**
* **Compatible with Patch embeddings, SSM filters, and attention blocks**

---

# 🧠 How VMSM Differs From Other Memory Types

| Memory Type    | What It Stores                  | Limitation                       | How VMSM Differs                  |
| -------------- | ------------------------------- | -------------------------------- | --------------------------------- |
| Attention      | Key/Value projections of tokens | O(n²) cost, short-range collapse | Stores global spectral modes      |
| RNN/SSM        | Recurrent hidden state          | Exponential decay                | Eigenmodes persist indefinitely   |
| PCA/SVD tricks | Offline compression             | Not learnable, not dynamic       | Online, learnable, task-adaptive  |
| Convolutions   | Local filters                   | Limited receptive field          | Global, frequency-aware structure |

VMSM is a **new memory class** because no other mechanism performs **online spectral extraction → learnable tokenization → reinjection**.

---

# 🧱 Minimal Usage (Pseudocode)

```python
memory = VMSM(d_model=512, memory_depth=3000, n_components=32, memory_tokens=8)

h = encoder_hidden_states   # [B, L, d_model]
m = memory(h)               # spectral memory tokens
out = model_with_memory(h, m)
```

---



# 📊 ETTh1 Benchmark Results (SeqLen = 96)

Evaluated on prediction horizons {96, 192, 336, 720} using the official **Time-Series-Library**.

### **Run 1**

| Horizon | MSE   | MAE   |
| ------- | ----- | ----- |
| 96      | 0.387 | 0.408 |
| 192     | 0.424 | 0.430 |
| 336     | 0.452 | 0.448 |
| 720     | 0.473 | 0.472 |
| **Avg** | 0.434 | 0.440 |

### **Run 2**

| Horizon | MSE   | MAE   |
| ------- | ----- | ----- |
| 96      | 0.388 | 0.408 |
| 192     | 0.425 | 0.430 |
| 336     | 0.451 | 0.448 |
| 720     | 0.485 | 0.482 |
| **Avg** | 0.437 | 0.442 |

### **Run 4**

| Horizon | MSE   | MAE   |
| ------- | ----- | ----- |
| 96      | 0.411 | 0.422 |
| 192     | 0.421 | 0.429 |
| 336     | 0.455 | 0.447 |
| 720     | 0.469 | 0.473 |
| **Avg** | 0.439 | 0.443 |

Run (3) and more datasets (Weather, ECL, Traffic, ILI) incoming.

---

# 🛠️ Reproducing Results

### Install

```bash
pip install torch pandas numpy scikit-learn
```

### Download Dataset

```bash
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv -P ./dataset/
```

### Run (Apple Silicon)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
for pred_len in 96 192 336 720; do 
  python run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ \
    --data_path ETTh1.csv --model_id ETTh1_96_${pred_len} --model KLMemory \
    --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
    --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 \
    --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --itr 1 \
    --use_gpu false --gpu_type mps 
done
```

---

# 🧩 Architecture Overview

### **K-L Memory (VMSM v1/v2/v3 — Used for Benchmarks)**

**Description:** This is the version used for all benchmark numbers. It uses a **low-rank bottleneck projection** to compress eigen-patterns efficiently, keeping the model lightweight (only \~7M parameters) and fast on both NVIDIA GPUs and Apple Silicon.

```python
#  Logic (Simplified)
H = F.normalize(self._history, dim=1)    # [T, d_model]
K = torch.exp(-dist / tau)               # Time-kernel (Gaussian/Exp)
L, V = torch.linalg.eigh(K)              # Eigen decomposition
patterns = self._history.T @ V_top       # Principal patterns
tokens = self.component_mixer(patterns)  # Bottleneck: (K*d -> 64 -> M*d)
```

### 2\. K-L Memory (VMSM v4) (High-Capacity / Research)

**Description:** A high-capacity, implementation designed for experimental research.

**Key Differences:**

  * **Dense Projection:** Uses a wide, 2-layer MLP (no bottleneck) for maximum theoretical capacity ($O(d_{model}^2)$ parameters).
  * **Precision:** Offloads eigen-solves to CPU (float64) for maximum numerical stability.
  * **Features:** Includes $\sqrt{\lambda}$ scaling, attention-based memory writing, and optional gradient detachment.

  **⚠️**This version has a larger parameters due to the dense projection layers (\~182M parameters).

---

⭐ Temporal Integrity and Memory Management

Memory-based forecasting architectures require special care to avoid cross-phase contamination and ensure that evaluation faithfully reflects generalization. This implementation includes explicit safeguards to guarantee that KLMemory / VMSM behaves properly under the standard Time-Series-Library benchmarking protocol.

1. Cross-Phase Memory Isolation

The KLMemory state buffer is never shared across training, validation, or test phases.
To enforce this:

Memory is reset at the start of every training epoch

Memory is reset immediately before validation

Memory is reset immediately before test evaluation

These resets are performed through the model’s reset_memory() method, which the experiment runner invokes automatically at each phase transition.
Result: No spectral information from training can leak into validation or test predictions.

2. Gradient Isolation of the KL Decomposition

The Karhunen–Loève computation (detach_kl=True) is non-trainable by design:

Eigendecomposition runs on detached hidden states

Only the downstream MLP projection is optimized

No gradients flow through the covariance matrix or eigenvectors

This prevents the model from “bending” the spectral basis toward specific training examples and ensures that KLMemory learns distribution-level structure, not shortcut encodings of specific sequences.

3. Temporal-Leakage-Free Benchmarking

This implementation follows the official Time-Series-Library protocol:

Training batches are shuffled (as done in PatchTST, iTransformer, TimesNet)

Validation and test loaders remain unshuffled

Covariance extraction is robust to batch ordering and does not access future horizon values

The KLMemory mechanism compresses latent training dynamics, not raw future values or label information, and is therefore fully compliant with the TSL temporal evaluation standard.



# 📘 Comparison to Related Work

| Method          | Basis Type            | Learnable | Adaptive    | Notes                |
| --------------- | --------------------- | --------- | ----------- | -------------------- |
| Autoformer      | Trend/Seasonal        | ✗         | Fixed       | Strong 2021 baseline |
| iTransformer    | Inverted Attention    | ✓         | Task-only   | Current SOTA         |
| PatchTST        | Patch Embeddings      | ✓         | Task-only   | Very competitive     |
| **VMSM (Ours)** | KL Eigenbasis + Mixer | ✓         | Data + Task | Simple, fast, robust |

---

# 🔗 Citation

```
Marquez, Vincent. (2025). VMSM: Vincent Marquez Spectral Memory. GitHub Repository.
```

---

# 📄 License

MIT License — see [LICENSE](LICENSE).



















