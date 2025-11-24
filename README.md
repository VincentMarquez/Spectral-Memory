K-L Memory: Spectral Covariance Memory for Long-Term Forecasting

K-L Memory combines Karhunen–Loève spectral decomposition with neural projection layers to create a long-context memory module for time-series forecasting.
It achieves competitive / near-state-of-the-art performance on ETTh1 without pretraining, without massive depth, and trained from scratch in only 10 epochs.

📊 ETTh1 Benchmark Results (SeqLen=96)

We evaluate on prediction horizons {96, 192, 336, 720} using the official Time-Series-Library setup.

Run 1
Horizon	MSE	MAE
96	0.387	0.408
192	0.424	0.430
336	0.452	0.448
720	0.473	0.472
Avg	0.434	0.440
Run 2
Horizon	MSE	MAE
96	0.388	0.408
192	0.425	0.430
336	0.451	0.448
720	0.485	0.482
Avg	0.437	0.442

More runs (3–5) and more datasets (Weather, ECL, Traffic, ILI) incoming.

🎯 Key Idea

K-L Memory performs spectral decomposition over historical hidden states:

Extracts dominant temporal patterns

Compresses them into a small number of memory tokens

Injects these tokens back into the model as long-term context

Benefits:

Mathematically structured (Karhunen–Loève / PCA)

Efficient (T = 2048 → K = 16 components → M = 4–8 tokens)

Noise-robust (eigenvalue truncation)

Rapid convergence (no pretraining required)

🛠️ Reproducing Results
Install
pip install torch pandas numpy scikit-learn

Download Dataset
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv -P ./dataset/

Run (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

for pred_len in 96 192 336 720; do 
    python run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_${pred_len} \
      --model KLMemory \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --batch_size 32 \
      --learning_rate 1e-4 \
      --train_epochs 10 \
      --use_gpu false \
      --gpu_type mps
done

🧩 Architecture Overview
1. Spectral Covariance Memory (Default)

This is the version used for all benchmark numbers.

H = F.normalize(self._history, dim=1)  # [T, d_model]
K = H @ H.T                             # feature Gram matrix
L, V = torch.linalg.eigh(K)            # eigen decomposition
patterns = self._history.T @ V_top     # principal patterns
tokens = self.component_mixer(patterns)

2. K-L Memory v4 (Optional)

Additional stability features:

Time-axis covariance:
C = (Hc @ Hc.T) / T

CPU float64 eigen-solves

sqrt(λ) scaling

Attention-based memory writing

Optional gradient-detach mode

The default results use the feature-kernel version, v4 is available for experimentation.

📘 Comparison to Related Work
Method	Basis Type	Learnable	Adaptive	Notes
Autoformer	Trend/Seasonal	✗	Fixed	Strong 2021 baseline
iTransformer	Inverted Attention	✓	Task-only	Current SOTA
Compressive Transformer	Learned compression	✓	Yes	No inductive basis
K-L Memory (Ours)	K-L + MLP	✓	Data + Task	Competitive, fast, simple
🔗 Citation
Marquez, Vincent. (2025). K-L Memory. GitHub repository.

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
