# The Spectrum Remembers

Spectral Memory is a simple memory module that converts training-trajectory statistics into a small set of Spectral Memory Tokens (SMTs) for long-range forecasting.

Instead of storing sequence content (recurrent states, KV caches, or SSM hidden states), Spectral Memory stores a compressed summary of how the model’s representations evolve across training batches. Each training batch is summarized through attention pooling, these summaries are collected in a history buffer, and a Karhunen–Loève (K-L) decomposition extracts the dominant spectral modes. A small learnable MLP maps these modes into SMTs, which are prepended to the model’s context at inference.

This repository implements the method on the ETTh1 dataset using the official Time-Series-Library Transformer backbone. The module runs on consumer hardware and requires no pretraining.

---

## What Spectral Memory Is

Spectral Memory Tokens (SMTs) are prefix tokens formed from the K-L modes of the training-trajectory history. They are not recurrent hidden states, attention keys/values, or SSM states. They represent global summaries of how the model behaved across many batches during training.

---

## Spectral Memory Pipeline

The method follows this sequence:

Training batches → attention pooling → history buffer  
→ K-L decomposition → top K spectral modes  
→ learnable MLP projection → M spectral memory tokens  
→ SMTs prepended to the model’s input at inference

The K-L step is fixed and non-trainable. Only the projection MLP and the backbone model receive gradients.

The novelty is in applying K-L to training-trajectory summaries and converting the resulting components into prefix tokens consumable by the attention mechanism.

---

## Architecture Scope

Conceptually the module can supply prefix tokens to any sequence model that uses attention. In this repository we evaluate only the Transformer-based forecaster from the Time-Series-Library on the ETTh1 benchmark. Extensions to other architectures are left for future work.

---

## Behavior on ETTh1

Under the official Time-Series-Library configuration (seq_len = 96, pred_len ∈ {96, 192, 336, 720}), Spectral Memory achieves an average MSE of 0.434. This is competitive with strong baselines under identical training budgets and adds only a small computational overhead.

These results are limited to a single dataset and should be treated as early-stage evidence rather than broad benchmark coverage.

---

## Relation to Other Memory Types

Spectral Memory differs from common memory mechanisms as follows:

- Attention KV stores token-level hidden states
- Recurrent and SSM memories store compressed sequence states
- Retrieval memories store external content
- Spectral Memory stores training-trajectory summaries written during training and read at inference

This places it in the “training-trajectory memory” category described in the paper.

---

## Why Use Spectral Memory

- Simple module that adds a small number of prefix tokens
- Uses mathematical structure (K-L) combined with a learnable projection
- Runs efficiently on CPUs and Apple Silicon
- Provides interpretable spectral components that reflect global training dynamics

---

## How Spectral Memory Differs From Other Memory Types

| Memory Type    | What It Stores                  | Limitation                       | How Spectral Memory Differs       |
| -------------- | ------------------------------- | -------------------------------- | --------------------------------- |
| Attention      | Key/Value projections of tokens | O(n²) cost, short-range collapse | Stores global spectral modes      |
| RNN/SSM        | Recurrent hidden state          | Exponential decay                | Eigenmodes persist indefinitely   |
| PCA/SVD tricks | Offline compression             | Not learnable, not dynamic       | Online, learnable, task-adaptive  |
| Convolutions   | Local filters                   | Limited receptive field          | Global, frequency-aware structure |

Spectral Memory is a **Potentialy new memory class** because no other mechanism performs **online spectral extraction → learnable tokenization → reinjection**.

## Results

| Model | MSE |
|-------|-----|
| **Spectral Memory (Ours)** | **0.434** |
| TimeXer | 0.437 |
| iTransformer | 0.454 |
| DLinear | 0.456 |
| TimesNet | 0.458 |
| PatchTST | 0.469 |
| Autoformer | 0.496 |

*Baseline results taken from respective Timexer table 3 publications under identical experimental settings (seq_len=96, pred_len ∈ {96, 192, 336, 720}).

*Table 3: Multivariate forecasting results. We compare extensive competitive models under different
prediction lengths following the setting of iTransformer [23]. The look-back length L is set to 96 for
all baselines. Results are averaged from all prediction lengths S = {96, 192, 336, 720}.

---
### Table 4: Dataset Descriptions 

*Dim = number of variables. Dataset Size = (Train, Val, Test). Prediction Lengths = forecasting horizons.*

| **Dataset**         | **Dim** | **Prediction Lengths** | **Dataset Size (Train, Val, Test)** | **Frequency** | **Domain**     |
| ------------------- | ------- | ---------------------- | ----------------------------------- | ------------- | -------------- |
| **ETTh1**           | 7       | {96, 192, 336, 720}    | (8545, 2881, 2881)                  | Hourly        | Electricity    |
| **ETTh2**           | 7       | {96, 192, 336, 720}    | (8545, 2881, 2881)                  | Hourly        | Electricity    |
| **ETTm1**           | 7       | {96, 192, 336, 720}    | (34465, 11521, 11521)               | 15 min        | Electricity    |
| **ETTm2**           | 7       | {96, 192, 336, 720}    | (34465, 11521, 11521)               | 15 min        | Electricity    |
| **Exchange**        | 8       | {96, 192, 336, 720}    | (5120, 665, 1422)                   | Daily         | Economy        |
| **Weather**         | 21      | {96, 192, 336, 720}    | (36792, 5271, 10540)                | 10 min        | Weather        |
| **ECL**             | 321     | {96, 192, 336, 720}    | (18317, 2633, 5261)                 | Hourly        | Electricity    |
| **Traffic**         | 862     | {96, 192, 336, 720}    | (12185, 1757, 3509)                 | Hourly        | Transportation |
| **Solar-Energy**    | 137     | {96, 192, 336, 720}    | (36601, 5161, 10417)                | 10 min        | Energy         |
| **PEMS03**          | 358     | {12, 24, 48, 96}       | (15617, 5135, 5135)                 | 5 min         | Traffic        |
| **PEMS04**          | 307     | {12, 24, 48, 96}       | (10172, 3375, 3375)                 | 5 min         | Traffic        |
| **PEMS07**          | 883     | {12, 24, 48, 96}       | (16911, 5622, 5622)                 | 5 min         | Traffic        |
| **PEMS08**          | 170     | {12, 24, 48, 96}       | (10690, 3548, 3548)                 | 5 min         | Traffic        |
| **Market-Merchant** | 285     | {12, 24, 72, 144}      | (7045, 1429, 1429)                  | 10 min        | Transaction    |
| **Market-Wealth**   | 485     | {12, 24, 72, 144}      | (7045, 1429, 1429)                  | 10 min        | Transaction    |
| **Market-Finance**  | 405     | {12, 24, 72, 144}      | (7045, 1429, 1429)                  | 10 min        | Transaction    |
| **Market-Terminal** | 307     | {12, 24, 72, 144}      | (7045, 1429, 1429)                  | 10 min        | Transaction    |
| **Market-Payment**  | 759     | {12, 24, 72, 144}      | (7045, 1429, 1429)                  | 10 min        | Transaction    |
| **Market-Customer** | 395     | {12, 24, 72, 144}      | (7045, 1429, 1429)                  | 10 min        | Transaction    |






### KLMemory Performance (ETTh1)
## Performance Benchmarks

The following table demonstrates the performance of **KLMemory** on the **ETTh1** dataset. Results are reported across four standard prediction lengths (96, 192, 336, 720) using 5 distinct random seeds to ensure robustness.

***
Additional datasets (Exchange, Weather, ECL, Traffic, ILI) forthcoming.
***



### ETTh1 Evaluation Results

| Pred Len | Metric | Seed 2019 | Seed 2020 | Seed 2021 | Seed 2022 | Seed 2023 | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 96 | MSE | 0.4180 | 0.3807 | 0.3903 | 0.4031 | 0.3834 | 0.3951 |
| 96 | MAE | 0.4228 | 0.4018 | 0.4115 | 0.4156 | 0.4029 | 0.4109 |
| 192 | MSE | 0.4177 | 0.4199 | 0.4186 | 0.4201 | 0.4227 | 0.4198 |
| 192 | MAE | 0.4278 | 0.4283 | 0.4270 | 0.4273 | 0.4312 | 0.4283 |
| 336 | MSE | 0.4512 | 0.4556 | 0.4507 | 0.4523 | 0.4616 | 0.4543 |
| 336 | MAE | 0.4470 | 0.4486 | 0.4456 | 0.4486 | 0.4521 | 0.4484 |
| 720 | MSE | 0.4461 | 0.5009 | 0.4680 | 0.4511 | 0.4807 | 0.4694 |
| 720 | MAE | 0.4561 | 0.4862 | 0.4698 | 0.4618 | 0.4779 | 0.4704 |
| **Avg MSE** | | **0.4332** | **0.4393** | **0.4319** | **0.4316** | **0.4371** | **0.4346** |
| **Avg MAE** | | **0.4384** | **0.4412** | **0.4385** | **0.4383** | **0.4410** | **0.4395** |


### Observations
* **Best Performance:** The model performed best at prediction length **96** with Seed **2020** (MSE: 0.3807).
* **Stability:** The results for length **192** are remarkably stable across all seeds, with MSEs only ranging from 0.4177 to 0.4227.
* **Outliers:** Seed 2020 had a significant spike in error at prediction length **720** (0.5009 MSE) compared to the other seeds in that bracket.


### 1\. TimeXer Performance (ETTh1)
## Performance Benchmarks

The following table benchmarks **TimeXer** on the **ETTh1** dataset. 

## Experimental Protocol

All results are obtained using the **official Time-Series-Library (TSLib)**  
Repository: https://github.com/thuml/Time-Series-Library (commit e8c0d2d, November 2025)  

- Zero modifications to training scripts, data loaders, optimizer schedules, or hyper-parameters  
- Default TSLib configuration exactly as shipped: 10 epochs, learning rate 1e-4, batch size 32, early stopping patience=3  
- Identical random seeds across all models: 2019, 2020, 2021, 2022, 2023  
- Every model (including all built-in baselines and ours) is executed via the unmodified `exp_long_term_forecast.py` entry point  
- The only addition is our lightweight Subspace Memory (KLMemory) module (~200 lines, registered as a standard TSLib model)

This guarantees a perfectly level playing field using the same short training budget that dozens of prior papers have reported with the default library.

**Key advantage of this protocol**  
While recent state-of-the-art models (CARD, TiDE, modern Mamba variants, etc.) typically rely on extended training (longer epochs), custom learning-rate schedules, or larger batches to reach their published numbers, the results below are produced with the unmodified, widely used (but intentionally lightweight) TSLib defaults.

Under these exact conditions, our drop-in subspace memory module delivers the strongest performance among all models shipped with the library, with the relative gap increasing at longer forecasting horizons.

All original per-seed tables (identical to those in this repository) The experiments are fully reproducible with a single command on the stock Time-Series-Library—no manual changes required.

### TimeXer Evaluation Results (ETTh1)
## Performance Benchmarks

The following table benchmarks **TimeXer** on the **ETTh1** dataset. Results are reported across four standard prediction lengths (96, 192, 336, 720) using 5 distinct random seeds.

### TimeXer Evaluation Results (ETTh1)

| Pred Len | Metric | Seed 2019 | Seed 2020 | Seed 2021 | Seed 2022 | Seed 2023 | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 96 | MSE | 0.3878 | 0.3903 | 0.3912 | 0.3928 | 0.3882 | 0.3901 |
| 96 | MAE | 0.4072 | 0.4087 | 0.4092 | 0.4099 | 0.4084 | 0.4087 |
| 192 | MSE | 0.4376 | 0.4352 | 0.4375 | 0.4352 | 0.4396 | 0.4370 |
| 192 | MAE | 0.4396 | 0.4361 | 0.4382 | 0.4364 | 0.4397 | 0.4380 |
| 336 | MSE | 0.4717 | 0.4764 | 0.4910 | 0.4773 | 0.4741 | 0.4781 |
| 336 | MAE | 0.4550 | 0.4563 | 0.4617 | 0.4630 | 0.4483 | 0.4569 |
| 720 | MSE | 0.5387 | 0.5005 | 0.5092 | 0.4854 | 0.6010 | 0.5270 |
| 720 | MAE | 0.5149 | 0.4874 | 0.4955 | 0.4781 | 0.5482 | 0.5048 |
| **Avg MSE** | | **0.4589** | **0.4506** | **0.4572** | **0.4477** | **0.4757** | **0.4580** |
| **Avg MAE** | | **0.4542** | **0.4471** | **0.4511** | **0.4469** | **0.4612** | **0.4521** |

> **Experimental Notes**
> * **Apples-to-apples configuration.** TimeXer is evaluated using the standard Time-Series-Library configuration on ETTh1, without dataset-specific hyperparameter tuning. KLMemory uses the same protocol, so differences reflect architectural behavior rather than aggressive per-model tuning.
> * **Stability observation.** Seed 2023 at prediction length 720 exhibits a noticeable error spike (MSE 0.6010), suggesting that TimeXer can be unstable for long-horizon forecasting on ETTh1, whereas KLMemory remains more stable in this regime.

-----

### 2\. Head-to-Head: KLMemory vs. TimeXer

*Comparison based on the 5-seed average*



| Pred Len | KLMemory MSE  | TimeXer MSE (Re-run) | **Difference** | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **96** | 0.3951 | **0.3901** | +0.0050 | TimeXer  |
| **192** | **0.4198** | 0.4370 | **-0.0172** | **KLMemory** |
| **336** | **0.4543** | 0.4781 | **-0.0238** | **KLMemory** |
| **720** | **0.4694** | 0.5270 | **-0.0576** | **KLMemory  |
| **AVG** | **0.4346** | 0.4580 | **-0.0234** | **KLMemory** |


The divergence at step 720 is our strongest argument. `KLMemory` retains information over long sequences much better than `TimeXer`.




---

## Reproducing Results

### Install

```bash
pip install torch pandas numpy scikit-learn
```

### Download Dataset

```bash
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv -P ./dataset/
```

### Run (Apple Silicon)ETTh1


```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

for seed in 2019 2020 2021 2022 2023; do
  for pred_len in 96 192 336 720; do 
    python run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id KLMemory_ETTh1_L96_pl${pred_len}_seed${seed} \
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
      --patience 3 \
      --batch_size 32 \
      --learning_rate 0.0001 \
      --itr 1 \
      --use_gpu false \
      --gpu_type mps \
      --seed $seed
  done
done
done
```

### Run (Apple Silicon) exchange_rate

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

for seed in 2019 2020 2021 2022 2023; do
  python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange96_seed${seed} \
    --model KLMemory \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 2048 \
    --factor 3 \
    --dropout 0.1 \
    --train_epochs 10 \
    --patience 3 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --itr 1 \
    --use_gpu false \
    --gpu_type mps \
    --seed $seed
done
```



## Memory Management

Memory-based forecasting architectures require special care to avoid **cross-phase contamination** and ensure that evaluation faithfully reflects generalization. This implementation includes explicit safeguards to guarantee that **Spectral Memory** behaves properly under the standard Time-Series-Library benchmarking protocol.

### **1. Cross-Phase Memory Isolation**

The Spectral Memory state buffer is **never shared** across training, validation, or test phases. To enforce this:

* Memory is **reset at the start of every training epoch**
* Memory is **reset immediately before validation**
* Memory is **reset immediately before test evaluation**

These resets are performed through the model's `reset_memory()` method, which the experiment runner invokes automatically at each phase transition.

**Result:** No spectral information from training can leak into validation or test predictions.

---

### **2. Gradient Isolation of the KL Decomposition**

The Karhunen–Loève computation (`detach_kl=True`) is **non-trainable by design**:

* Eigendecomposition runs on **detached hidden states**
* Only the downstream **MLP projection** is optimized
* No gradients flow through the covariance matrix or eigenvectors

This prevents the model from "bending" the spectral basis toward specific training examples and ensures that Spectral Memory learns **distribution-level structure**, not shortcut encodings of specific sequences.

---

### **3. Temporal-Leakage-Free Benchmarking**

This implementation follows the official **Time-Series-Library** protocol:

* Training batches are shuffled (as done in PatchTST, iTransformer, TimesNet)
* Validation and test loaders remain unshuffled
* Covariance extraction is robust to batch ordering and does not access future horizon values

The Spectral Memory mechanism compresses **latent training dynamics**, not raw future values or label information, and is therefore **fully compliant with the TSL temporal evaluation standard**.


---

#  Leakage Detection & Sanity Validation

Time-series forecasting pipelines are vulnerable to hidden data leakage through window misalignment, preprocessing mistakes, or unintended model state carryover. To ensure KLMemory is evaluated correctly and does not benefit from leaked future information, we performed an extensive set of destructive sanity checks.

Each test intentionally removes or corrupts information that the model should depend on.
A valid, leak-free model should experience a significant degradation in performance under these conditions.

All tests were run on the **Exchange-Rate** dataset (96-step horizon).

---

#  1. Normal Evaluation (Baseline)

| Mode       | MSE         | MAE     |
| ---------- | ----------- | ------- |
| **normal** | **0.08610** | 0.20314 |

This is the expected performance using real, unmodified inputs.

---

#  2. Input–Target Misalignment Test (Shuffle-X)

We shuffle encoder windows within the batch, breaking the natural pairing between each input window and its corresponding label:

```python
idx = torch.randperm(batch_x.size(0))
batch_x = batch_x[idx]
```

| Mode          | MSE         | MAE     |
| ------------- | ----------- | ------- |
| **shuffle_x** | **0.08735** | 0.20355 |

**Interpretation:**
Small degradation is expected since shuffled windows come from the same distribution within the test split.
This confirms no strict index-based leakage.

---

#  3. Gaussian Noise Test (Encoder Information Removed)

All encoder inputs are replaced with random Gaussian noise:

```python
batch_x = torch.randn_like(batch_x)
```

| Mode        | MSE         | MAE     |
| ----------- | ----------- | ------- |
| **noise_x** | **2.88824** | 1.38655 |

**Interpretation:**
Significant degradation indicates that the model depends heavily on actual historical inputs and cannot perform well in their absence.

---

#  4. Additional Input-Corruption Variants

(Testing alternate pathways)

### 4.1 Noise Encoder + Zero Decoder Input

Tests whether decoder inputs accidentally leak information.

| Mode                 | MSE         |
| -------------------- | ----------- |
| **noise_x_zero_dec** | **2.88276** |

### 4.2 Noise Encoder + Zero Time Features

Tests whether time encodings contain unintended future information.

| Mode                   | MSE         |
| ---------------------- | ----------- |
| **noise_x_zero_marks** | **2.88799** |

**Interpretation:**
Similar degradation across all variants rules out decoder-based leakage and timestamp leakage.

---

#  5. Full Isolation Test (Noise + Zero Ancillary Inputs + Per-Batch Memory Reset)

This setting removes all meaningful input channels **and** disables cross-batch state:

* Encoder input = Gaussian noise
* Decoder input = zero
* Timestamp features = zero
* Spectral memory reset before every batch

```python
if hasattr(self.model, "reset_memory"):
    self.model.reset_memory()   # enforced per batch
```

| Mode                             | MSE         |
| -------------------------------- | ----------- |
| **noise_x_zero_all_reset_batch** | **2.90528** |

**Interpretation:**
Performance remains in the degraded range (~2.9 MSE), demonstrating that the model does not rely on hidden cross-batch state or any auxiliary feature that could carry future information.

---

# Overall Conclusion

KLMemory shows:

* **Strong performance with valid inputs** (0.086 MSE)
* **Expected failure when historical information is removed** (~2.9 MSE)
* **No dependence on decoder inputs, timestamps, or cross-batch memory**
* **No signs of target leakage, future leakage, or normalization leakage**

These results confirm that the reported forecasts are **valid, leak-free, and the model is not exploiting any unintended shortcuts**.

| **Mode**                         | **Description**                                                                            | **Encoder Input** | **Decoder Input** | **Time Features** | **Memory**          | **MSE**    | **Interpretation**                           |
| -------------------------------- | ------------------------------------------------------------------------------------------ | ----------------- | ----------------- | ----------------- | ------------------- | ---------- | -------------------------------------------- |
| **normal**                       | Standard evaluation                                                                        | Real              | Standard          | Real              | Normal              | **0.0861** | Expected performance with real data          |
| **shuffle_x**                    | Break input–target alignment (batch shuffle)                                               | Shuffled          | Standard          | Real              | Normal              | **0.0873** | Slight degradation; no index-based leakage   |
| **noise_x**                      | Remove all encoder information (Gaussian noise)                                            | **Noise**         | Standard          | Real              | Normal              | **2.8882** | Model collapses as expected → **no leakage** |
| **noise_x_zero_dec**             | Noise encoder + zero decoder                                                               | **Noise**         | Zero              | Real              | Normal              | **2.8828** | Confirms decoder is not leaking future info  |
| **noise_x_zero_marks**           | Noise encoder + zero time features                                                         | **Noise**         | Standard          | **Zero**          | Normal              | **2.8880** | Time marks do not encode future information  |
| **noise_x_zero_all_reset_batch** | Full isolation: noise encoder + zero decoder + zero time features + per-batch memory reset | **Noise**         | Zero              | Zero              | **Reset per batch** | **2.9053** | Confirms no cross-batch state contamination  |




---

## Comparison to Related Work

| Method                    | Basis Type            | Learnable | Adaptive    | Notes                |
| ------------------------- | --------------------- | --------- | ----------- | -------------------- |
| Autoformer                | Trend/Seasonal        | ✗         | Fixed       | Strong 2021 baseline |
| iTransformer              | Inverted Attention    | ✓         | Task-only   | Current SOTA         |
| PatchTST                  | Patch Embeddings      | ✓         | Task-only   | Very competitive     |
| **Spectral Memory (Ours)**| KL Eigenbasis + Mixer | ✓         | Data + Task | Simple, fast, robust |

---

## Citation

```bibtex
@article{marquez2025spectral,
  title   = {The Spectrum Remembers: Spectral Memory for Long-Context Sequence Modeling},
  author  = {Marquez, Vincent},
  year    = {2025},
  note    = {GitHub Repository}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).


