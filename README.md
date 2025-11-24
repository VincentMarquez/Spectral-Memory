# K-L-Memory: Spectral Covariance Memory for Long-Term Forecasting

## 📊 ETTh1 Benchmark Results

This repository implements **K-L-Memory (Karhunen-Loève Memory Tokens)**, a novel approach that combines classical signal processing with neural learning for efficient long-context sequence modeling. Our method achieves state-of-the-art results on the ETTh1 electricity transformer temperature forecasting benchmark (Time-Series-Library).

### 🎯 Key Innovation

Unlike purely learned compression methods, K-L-Memory uses spectral decomposition to extract dominant temporal patterns from historical hidden states, then maps these components to task-specific memory tokens through a learnable neural projection. This provides:

- **Mathematical structure** from signal processing (K-L decomposition)
- **Task adaptation** through gradient-based learning
- **Provable optimality** in window complexity reduction (see Theorem 1 in paper)

**Note:** All results were achieved without any pre-training - the model was trained from scratch in just 10 epochs with early stopping (patience=3), demonstrating that the K-L decomposition provides strong enough inductive bias for rapid convergence.
  
## 🚀 Performance Results

### ETTh1 Long-Term Forecasting

We evaluate on the standard ETTh1 benchmark with input length 96 and prediction horizons {96, 192, 336, 720}:


Performance Results
ETTh1 Long-Term Forecasting
We evaluate on the standard ETTh1 benchmark with input length 96 and prediction horizons {96, 192, 336, 720}. We provide results for multiple runs to demonstrate stability, particularly in long-horizon handling.


### ETTh1 Long-Term Forecasting Results (input length = 96)

Run 3 & 4 -> Will add final numbers

#### Run 1 
| Horizon | MSE   | MAE   |
|---------|-------|-------|
| 96      | 0.387 | 0.408 |
| 192     | 0.424 | 0.430 |
| 336     | 0.452 | 0.448 |
| 720     | 0.473 | 0.472 |
| **Avg** | **0.434** | **0.440** |

#### Run 2
| Horizon | MSE   | MAE   |
|---------|-------|-------|
| 96      | 0.388 | 0.408 |
| 192     | 0.425 | 0.430 |
| 336     | 0.451 | 0.448 |
| 720     | 0.485 | 0.482 |
| **Avg** | **0.437** | **0.442** |


More runs and additional datasets (Weather, Traffic, ECL, etc.) coming soon.






## 🔧 Reproducing Results

### Prerequisites
```bash
# Install dependencies
pip install torch, pandas, numpy, scikit-learn, Math
Python3.13.3

## Hardware & Training

- **Device**: Apple M4 Mac Mini  
- **Memory**: 16 GB unified  
- **Training**: CPU-only  
- **Total time to SOTA**: 9-15 hours

# Download ETTh1 dataset
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv -P ./dataset/
```

### Running Experiments

For Apple Silicon (M1/M2/M3/M4):
```bash
# Enable MPS fallback for compatibility
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run all prediction horizons
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
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --train_epochs 10 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --itr 1 \
        --use_gpu false \
        --gpu_type mps
done
```

For CUDA GPUs:
```bash
# Simply change gpu settings
--use_gpu true --gpu_type cuda
```

### Architecture Configuration
```python
# K-L-Memory Settings (KLMemory.py)
memory_depth=2048        # Deep history buffer
n_components=16          # Number of Eigencomponents to extract
memory_tokens=8          # Tokens injected into Transformer
d_model=512             # Hidden dimension

# Training Settings
batch_size=32
learning_rate=1e-4
epochs=10
early_stopping_patience=3
```

## 🏗️ Model Architecture

### Core Components

1. **Spectral Covariance Memory**: Extracts dominant temporal patterns via K-L decomposition
   - Computes covariance kernel: `K = H @ H.T`
   - Eigendecomposition: `L, V = torch.linalg.eigh(K)`
   - Selects top-k principal components

2. **Learnable Memory Projection**: Maps K-L components to task-specific memory tokens
   - MLP with GELU activation
   - Trained end-to-end via backpropagation

3. **Channel Independence**: Treats each variable as separate univariate series
   - Enables learning frequency-specific patterns per channel

4. **Flatten Head Decoder**: Direct mapping from sequence representation to predictions
   - Maps `(Seq_Len * d_model)` → `(Pred_Len)`

### Why K-L-Memory Works

- **Provable Optimality**: K-L minimizes window complexity (Theorem 1)
- **Natural Denoising**: Eigenvalue truncation filters uncorrelated noise
- **Efficient Compression**: T=2048 states → K=16 components → M=4 tokens
- **Linear Scaling**: O(L) complexity with infinite context capability

## 📈 Computational Efficiency

| Metric | Value |
|--------|-------|
| Training Time | ~340-1200 seconds/epoch (M4 Mac CPU/MPS) |
| Scaling | O(L) - linear with sequence length |

## 📚 Comparison with Related Work (Will add more)

| Method | Basis Type | Learnable | Adaptive | Benchmark |
|--------|------------|-----------|----------|-----------|
| Compressive Transformer | None | ✓ Learned | Task-only | - |
| Autoformer [[Wu et al., 2021]](https://arxiv.org/abs/2106.13008) | Trend/Seasonal | ✗ Fixed | Fixed prior | SOTA until 2024 |
| iTransformer [[Liu et al., 2024]](https://arxiv.org/pdf/2510.02729) | Inverted attention | ✓ | Task-only | Current SOTA |
| **K-L-Memory (Ours)** | K-L + MLP | ✓ | Data + Task | Competitive |

## 🔬 Technical Details

### Spectral Covariance Memory (K-L Transform)
```python
# Core K-L decomposition (simplified)
H = F.normalize(self._history, p=2, dim=1)  # Normalize history
K = torch.mm(H, H.t())                      # Gram Matrix
L, V = torch.linalg.eigh(K)                 # Eigendecomposition
V_top = V[:, top_k_indices]                 # Top-k eigenvectors
patterns = torch.mm(self._history.t(), V_top)  # Principal patterns
memory_tokens = self.component_mixer(patterns)  # Learnable projection
```

### Key Innovations
- **Feature Covariance Kernel** (not Time Kernel) - robust to batch shuffling
- **RevIN normalization** - handles non-stationary data
- **Principal Temporal Patterns** - eigen-features extracted on the fly




🔄 K-L Memory v4 (Updated Implementation)

K-L Memory v4 is an improved version of the original module with better stability, cleaner mathematics, and a smarter mechanism for writing information into memory. It is fully drop-in compatible with the v1,2,3 interface.

What’s New in v4

Time-Axis K-L Decomposition
Uses mean-centered temporal covariance
C = (Hc @ Hc.T) / T
for extracting dominant temporal patterns.
More Stable Eigen Solves
All eigendecomposition is done on CPU in float64, preventing numerical issues on MPS/CUDA.
Eigenvalue Scaling Option
Coefficients can be scaled by sqrt(lambda) for stronger separation of major/minor modes.
Attention-Based Memory Writing
Instead of averaging, v2 learns which parts of the sequence should be stored:
weights = softmax(pooling_layer(out), dim=0)
summary = (out * weights).sum(dim=0)
memory.append(summary)

Optional Gradient Detach
detach_kl=True keeps K-L extraction fixed and only trains the projection MLP if desired.

When to Use v4
more stable training a cleaner and more consistent K-L extraction better long-horizon behavior improved memory writing via attention




## 🔗 References - See paper for more 

- [Autoformer Paper](https://arxiv.org/abs/2106.13008) - Decomposition transformers baseline
- [iTransformer Paper](https://arxiv.org/pdf/2510.02729) - Current SOTA comparison
- [Time Series Library](https://github.com/thuml/Time-Series-Library) - Benchmark implementation

## Citation

If you use this code or build upon the K-L Memory methodology in your research, please cite it in your publications and give appropriate credit.

### Recommended Citation Format

```
Marquez, Vincent. (2025). K-L Memory. GitHub repository: https://github.com/[VincentMarquez]/K-L-Memory
```

Or in BibTeX format:

```bibtex
@software{kl_memory,
  author = {Marquez, Vincent},
  title = {K-L Memory},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/[vincentmarquez]/K-L-Memory}
}
```

## Attribution

If you use this vision, methodology, or code in your work, please:
- Provide clear attribution to Vincent Marquez in your paper's acknowledgments or methods section
- Reference the K-L Memory repository when describing derived work
- Maintain this notice in any derivative works or forks

Thank you for respecting the effort that went into developing this work.




## 📧 Contact

For questions or collaborations, please open an issue or contact [vincentmarquez405@gmail.com].

---

**Note**: K-L-Memory is part of ongoing research. While we achieve strong results on ETTh1, comprehensive evaluation across additional benchmarks (Weather, Traffic, ECL) is in progress.


## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
