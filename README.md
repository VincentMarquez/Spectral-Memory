
# K-L Memory: Spectral Memory for Long-Term Forecasting

K-L Memory combines **Karhunen–Loève spectral decomposition** with neural projection layers to create a **long-context memory module** for time-series forecasting.
Preliminary It achieves **state-of-the-art** performance on ETTh1 *without pretraining*, *without massive depth*, and *trained from scratch in only 10 epochs, with consumer grade hardware (Apple M4 mini 10CPU-10GPU 16GB)*. 

---

## 📊 ETTh1 Benchmark Results (SeqLen=96)

We evaluate on prediction horizons {96, 192, 336, 720} using the official **Time-Series-Library** setup.
# K-L Memory: Spectral Memory for Long-Term Forecasting

K-L Memory combines Karhunen–Loève spectral decomposition with neural projection layers to create a long-context memory module for time-series forecasting. Preliminary results show that it achieves state-of-the-art performance on ETTh1 without pretraining, without massive depth, and trained from scratch in only 10 epochs on consumer-grade hardware (Apple M4 mini 10CPU–10GPU, 16GB).

## 📊 ETTh1 Benchmark Results (SeqLen = 96)

We evaluate on prediction horizons {96, 192, 336, 720} using the official Time-Series-Library setup.

### Run 1

| Horizon | MSE   | MAE   |
|---------|-------|-------|
| 96      | 0.387 | 0.408 |
| 192     | 0.424 | 0.430 |
| 336     | 0.452 | 0.448 |
| 720     | 0.473 | 0.472 |
| **Avg** | 0.434 | 0.440 |

### Run 2

| Horizon | MSE   | MAE   |
|---------|-------|-------|
| 96      | 0.388 | 0.408 |
| 192     | 0.425 | 0.430 |
| 336     | 0.451 | 0.448 |
| 720     | 0.485 | 0.482 |
| **Avg** | 0.437 | 0.442 |

### Run 4

| Horizon | MSE   | MAE   |
|---------|-------|-------|
| 96      | 0.411 | 0.422 |
| 192     | 0.421 | 0.429 |
| 336     | 0.455 | 0.447 |
| 720     | 0.469 | 0.473 |
| **Avg** | 0.439 | 0.443 |

More runs and more datasets (Weather, ECL, Traffic, ILI) are incoming.

Run (3) and more datasets (Weather, ECL, Traffic, ILI) incoming.

---

## 🎯 Key Idea

K-L Memory performs **spectral decomposition** over historical hidden states:

* Extracts **dominant temporal patterns**
* Compresses them into a small number of **memory tokens**
* Injects these tokens back into the model as long-term context

Benefits:

* **Mathematically structured** (Karhunen–Loève / PCA)
* **Efficient** (T = 2048 → K = 16 components → M = 4–8 tokens)
* **Noise-robust** (eigenvalue truncation)
* **Rapid convergence** (no pretraining required)

---

## 🛠️ Reproducing Results

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
for pred_len in 96 192 336 720; do python run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_96_${pred_len} --model KLMemory --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len $pred_len --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --itr 1 --use_gpu false --gpu_type mps; done
done
```


### 🧩 Architecture Overview

### 1\. Spectral Covariance Memory (Default / Production)

**Description:** This is the version used for all benchmark numbers. It uses a **low-rank bottleneck projection** to compress eigen-patterns efficiently, keeping the model lightweight (only \~0.7M parameters) and fast on both NVIDIA GPUs and Apple Silicon.

```python
# Production Logic (Simplified)
H = F.normalize(self._history, dim=1)    # [T, d_model]
K = torch.exp(-dist / tau)               # Time-kernel (Gaussian/Exp)
L, V = torch.linalg.eigh(K)              # Eigen decomposition
patterns = self._history.T @ V_top       # Principal patterns
tokens = self.component_mixer(patterns)  # Bottleneck: (K*d -> 64 -> M*d)
```

### 2\. K-L Memory v4 (High-Capacity / Research)

**Description:** A high-capacity, paper-faithful implementation designed for experimental research.

**Key Differences:**

  * **Dense Projection:** Uses a wide, 2-layer MLP (no bottleneck) for maximum theoretical capacity ($O(d_{model}^2)$ parameters).
  * **Precision:** Offloads eigen-solves to CPU (float64) for maximum numerical stability.
  * **Features:** Includes $\sqrt{\lambda}$ scaling, attention-based memory writing, and optional gradient detachment.

  **⚠️**This version has a larger parameters due to the dense projection layers (\~182M parameters).




## 📘 Comparison to Related Work

| Method                  | Basis Type          | Learnable | Adaptive    | Notes                     |
| ----------------------- | ------------------- | --------- | ----------- | ------------------------- |
| Autoformer              | Trend/Seasonal      | ✗         | Fixed       | Strong 2021 baseline      |
| iTransformer            | Inverted Attention  | ✓         | Task-only   | Current SOTA              |
| Compressive Transformer | Learned compression | ✓         | Yes         | No inductive basis        |
| **K-L Memory (Ours)**   | K-L + MLP           | ✓         | Data + Task | Competitive, fast, simple |

---

## 🔗 Citation

```
Marquez, Vincent. (2025). K-L Memory. GitHub repository.
```


## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
