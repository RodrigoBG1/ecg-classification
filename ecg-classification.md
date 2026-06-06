# Multi-Label ECG Classification on PTB-XL+
## Comparative Study of Deep Learning Architectures

---

## 1. Project Overview

This project presents a multi-label ECG classification system trained and evaluated on the PTB-XL+ dataset. The system classifies 12-lead ECG recordings into five non-exclusive diagnostic categories using three deep learning architectures: a ResNet-1D with squeeze-and-excitation attention and stochastic depth, a patch-based Transformer (PatchTST), and a CNN-Transformer hybrid. The primary evaluation metric is macro AUC-ROC (threshold-independent), with macro F1, AUPRC, sensitivity, and specificity as secondary metrics.

### Diagnostic Classes

| Label | Full Name | Prevalence (Train) |
|---|---|---|
| NORM | Normal ECG | ~81% |
| MI | Myocardial Infarction | ~32% |
| STTC | ST/T-wave Changes | ~23% |
| CD | Conduction Disturbance | ~23% |
| HYP | Hypertrophy | ~13% |

Each recording is assigned a binary 5-dimensional label vector; classes are not mutually exclusive.

### Condiciones detectables por clase (44 códigos SCP totales)

**NORM — ECG Normal**
- `NORM` — ECG sin patología diagnóstica

**MI — Infarto de Miocardio** (14 subtipos)
- `IMI` — Infarto inferior
- `ASMI` — Infarto anteroseptal
- `ILMI` — Infarto inferolateral
- `AMI` — Infarto anterior
- `ALMI` — Infarto anterolateral
- `LMI` — Infarto lateral
- `IPLMI` — Infarto inferoposterolateral
- `IPMI` — Infarto inferoposterior
- `PMI` — Infarto posterior
- `INJAS` — Lesión subendocárdica en derivaciones anteroseptales
- `INJAL` — Lesión subendocárdica en derivaciones anterolaterales
- `INJIN` — Lesión subendocárdica en derivaciones inferiores
- `INJLA` — Lesión subendocárdica en derivaciones laterales
- `INJIL` — Lesión subendocárdica en derivaciones inferolaterales

**STTC — Cambios ST/T** (13 subtipos)
- `NDT` — Anomalías no diagnósticas de la onda T
- `NST_` — Cambios ST inespecíficos
- `ISC_` — Isquemia inespecífica
- `ISCAL` — Isquemia en derivaciones anterolaterales
- `ISCIN` — Isquemia en derivaciones inferiores
- `ISCIL` — Isquemia en derivaciones inferolaterales
- `ISCAS` — Isquemia en derivaciones anteroseptales
- `ISCLA` — Isquemia en derivaciones laterales
- `ISCAN` — Isquemia en derivaciones anteriores
- `ANEUR` — Cambios ST/T compatibles con aneurisma ventricular
- `LNGQT` — Intervalo QT prolongado
- `DIG` — Efecto digitálico
- `EL` — Alteración electrolítica o por fármacos

**CD — Trastorno de Conducción** (11 subtipos)
- `CLBBB` — Bloqueo completo de rama izquierda
- `CRBBB` — Bloqueo completo de rama derecha
- `ILBBB` — Bloqueo incompleto de rama izquierda
- `IRBBB` — Bloqueo incompleto de rama derecha
- `LAFB` — Bloqueo fascicular anterior izquierdo
- `LPFB` — Bloqueo fascicular posterior izquierdo
- `IVCD` — Trastorno inespecífico de conducción intraventricular
- `1AVB` — Bloqueo AV de primer grado
- `2AVB` — Bloqueo AV de segundo grado
- `3AVB` — Bloqueo AV de tercer grado (completo)
- `WPW` — Síndrome de Wolff-Parkinson-White

**HYP — Hipertrofia** (5 subtipos)
- `LVH` — Hipertrofia ventricular izquierda
- `RVH` — Hipertrofia ventricular derecha
- `LAO/LAE` — Sobrecarga/agrandamiento auricular izquierdo
- `RAO/RAE` — Sobrecarga/agrandamiento auricular derecho
- `SEHYP` — Hipertrofia septal

---

## 2. Dataset

### PTB-XL+

- **Source:** PhysioNet — PTB-XL: A Large Publicly Available Electrocardiography Dataset v1.0.3
- **Size:** 21,799 clinical 12-lead ECG recordings from 18,869 patients
- **Sampling rate:** 500 Hz (5,000 samples per 10-second recording); 100 Hz version also available
- **Leads:** 12 standard leads (I, II, III, aVR, aVL, aVF, V1–V6)
- **Annotations:** SCP-ECG codes from up to two cardiologists, mapped to diagnostic superclasses
- **Metadata file:** `ptbxl_database.csv` with patient demographics, strat_fold, and SCP code strings

### Dataset Split

Partitioning follows the official stratified fold scheme; folds are never re-shuffled.

| Split | Folds | Samples |
|---|---|---|
| Train | 1–8 | 17,418 |
| Validation | 9 | 2,183 |
| Test | 10 | 2,198 |

### Label Construction (`data/labels.py`)

SCP codes are extracted from the `scp_codes` column (stored as a Python dict string), filtered for diagnostic codes with likelihood ≥ 50%, and mapped to one of the five superclasses via a predefined lookup table derived from `scp_statements.csv`. Recordings with no diagnostic code are excluded from label supervision but remain in the splits for loss computation under all-zero labels.

---

## 3. Data Pipeline

### 3.1 Normalization (`data/preprocessing.py`, `scripts/preprocess.py`)

Per-lead z-score normalization is applied to all splits using statistics fitted exclusively on the training set to prevent data leakage.

The normalizer uses a single-pass streaming (Welford) algorithm that processes one file at a time, keeping memory usage constant regardless of dataset size (~O(1) RAM). The resulting per-lead mean and standard deviation (shape `(12, 1)`) are saved to `config/norm_stats.npy` and loaded at training/inference time.

Class imbalance weights (`pos_weight`) for BCEWithLogitsLoss are computed as:

```
pos_weight[c] = (N - n_c) / n_c
```

where N is the total number of training samples and n_c is the positive count for class c. Saved to `data/processed/pos_weight.npy`.

### 3.2 Augmentation (`data/preprocessing.py`)

Augmentations are applied exclusively to the training split. The PTBXLDataset class applies the following pipeline stochastically per sample:

| Transform | Parameters | Effect |
|---|---|---|
| `GaussianNoise` | σ = 0.02 | Additive white noise per sample |
| `RandomAmplitudeScale` | scale ∈ [0.8, 1.2] | Uniform lead amplitude jitter |
| `RandomCrop` | crop=4500, output=5000 | Simulate truncated recordings via zero-padded crop |
| `RandomLeadDropout` | p=0.2, max_leads=2 | Zeros up to 2 random leads for robustness |

### 3.3 Dataset Loading (`data/dataset.py`)

`PTBXLDataset` is a PyTorch `Dataset` that reads WFDB-format records from `records500/` using the `wfdb` library. For each index, it:

1. Loads the raw signal array (shape `(5000, 12)`) from the corresponding `.hea`/`.dat` file pair
2. Transposes to `(12, 5000)` (channels-first)
3. Applies the normalization transform
4. Applies augmentation if in training mode
5. Returns a `(torch.FloatTensor(12, 5000), torch.FloatTensor(5))` tuple

---

## 4. Model Architectures

### 4.1 ResNet-1D (`models/resnet1d.py`)

A residual convolutional network adapted for 1-D time series with two architectural enhancements: Squeeze-and-Excitation (SE) channel attention and Stochastic Depth (DropPath).

**Architecture:**

```
Input: (B, 12, 5000)

Stem:
  Conv1d(12 → 64, k=15, s=2, pad=7) → BN → ReLU   [→ (B, 64, 2500)]
  MaxPool1d(k=4, s=4)                                [→ (B, 64, 625)]

Stage 1:  2 × ResBlock(64,  64,  k=7, stride=1)     [→ (B, 64,  625)]
Stage 2:  2 × ResBlock(64,  128, k=7, stride=2)     [→ (B, 128, 313)]
Stage 3:  2 × ResBlock(128, 256, k=7, stride=2)     [→ (B, 256, 157)]
Stage 4:  2 × ResBlock(256, 512, k=7, stride=2)     [→ (B, 512,  79)]

Head:
  AdaptiveAvgPool1d(1)   [→ (B, 512)]
  Dropout(0.3)
  Linear(512 → 5)        [→ (B, 5) logits]
```

**ResBlock internals:**
```
x → Conv(k=7) → BN → ReLU → Conv(k=7) → BN → SEBlock → DropPath → + skip → ReLU
```

**SEBlock1D:** Per-channel attention via `AdaptiveAvgPool1d(1) → Linear(C→C/16) → ReLU → Linear(C/16→C) → Sigmoid`, applied as a channel-wise multiplicative gate.

**Stochastic Depth (DropPath):** At training time, randomly drops the entire residual branch with probability p, which is linearly scheduled from 0 to 0.2 across the 8 blocks (block i → rate = i/7 × 0.2). At inference, the branch is always kept with a scale correction.

**Parameters:** ~8.9M

---

### 4.2 PatchTST (`models/patchtst.py`)

A Transformer adapted for time-series that operates channel-independently on non-overlapping patches. Each lead is processed separately, and lead representations are aggregated by mean pooling before the classification head.

**Architecture:**

```
Input: (B, 12, 5000)

Per-lead embedding (applied independently to each of 12 leads):
  Unfold: (B, 12, 5000) → (B, 12, 100, 50)     [100 patches of 50 samples]
  Linear projection: (B, 12, 100, 50) → (B, 12, 100, d_model)
  Positional encoding: learnable (100, d_model) added

Reshape for Transformer: (B × 12, 100, d_model)

Transformer Encoder (N=4 layers):
  MultiHeadAttention(n_heads=8)
  FFN (d_model → ff_dim → d_model)
  LayerNorm + dropout

Mean pool over time: (B × 12, 100, d_model) → (B × 12, d_model)
Reshape: (B, 12, d_model)
Mean pool over leads: (B, d_model)

Head:
  Dropout(p=0.1)
  Linear(d_model → 5)
```

**Hyperparameters (config.yaml):** d_model=128, n_heads=4, n_layers=4, ff_dim=256, attn_dropout=0.1, dropout=0.1

Patch size of 50 samples corresponds to 100 ms at 500 Hz, capturing one typical P-wave or QRS complex width.

---

### 4.3 CNN-Transformer Hybrid (`models/cnn_transformer.py`)

A two-stage architecture that combines a convolutional front-end for local feature extraction with a Transformer encoder for long-range temporal context.

**Architecture:**

```
Input: (B, 12, 5000)

CNN Encoder (depthwise-separable residual blocks):
  Stem: Conv1d(12 → d_model, k=1)               [→ (B, 256, 5000)]
  Block 1: DWSResBlock(256, 256, k=7, stride=2)  [→ (B, 256, 2500)]
  Block 2: DWSResBlock(256, 256, k=7, stride=2)  [→ (B, 256, 1250)]
  Block 3: DWSResBlock(256, 256, k=7, stride=2)  [→ (B, 256, 625)]
  Block 4: DWSResBlock(256, 256, k=7, stride=2)  [→ (B, 256, 313)]

DWSResBlock internals:
  Conv1d(depthwise, k=7) → BN → ReLU → Conv1d(pointwise, k=1) → BN → SE → DropPath → + skip

Positional Encoding: learnable (T, d_model) added to CNN output

Transformer Encoder (N=4 layers):
  MultiHeadAttention(d_model=256, n_heads=8)
  FFN (256 → 1024 → 256)
  LayerNorm + dropout(0.1)
                                                  [→ (B, 313, 256)]
Head:
  AdaptiveAvgPool1d(1)                            [→ (B, 256)]
  Dropout(0.1)
  Linear(256 → 5)                                 [→ (B, 5) logits]
```

**Parameters:** ~3.7M (2.4× smaller than ResNet-1D)

The depthwise-separable convolutions reduce parameter count while maintaining receptive field. The CNN encoder reduces sequence length from 5000 to 313 (16×), making self-attention computationally tractable (attention scales as O(T²)).

---

## 5. Training Infrastructure

### 5.1 Loss Function

Binary Cross-Entropy with Logits with per-class positive weights:

```
L = BCEWithLogitsLoss(logits, targets, pos_weight=w)
```

where `w[c] = (N - n_c) / n_c` (computed on training set, loaded from `data/processed/pos_weight.npy`).

**Label smoothing (ε=0.1):** Applied before loss computation. Positive labels → 0.95; negative labels → 0.05. Prevents overconfident predictions.

### 5.2 Optimizer and Scheduler

| Component | Setting |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Gradient clipping | L2 norm ≤ 1.0 |
| Warmup | 5 linear epochs (0.01 → 1.0 × lr) |
| LR schedule | Cosine annealing (T_max=epochs, η_min=1e-6) |

### 5.3 Regularization

- **Mixup (α=0.4):** Applied per batch. Pairs of samples (x_i, x_j) and labels (y_i, y_j) are linearly interpolated with λ ~ Beta(0.4, 0.4). Applied with ~40% probability per batch. Effective against overconfidence.
- **Stochastic Depth:** Applied within ResNet-1D and CNN-Transformer blocks (drop_path_rate=0.3).
- **Dropout:** 0.3 before the final linear layer in ResNet-1D; 0.1 in Transformer layers.
- **Data augmentation:** Four stochastic transforms described in Section 3.2.

### 5.4 Training Loop (`training/trainer.py`)

- **Mixed precision:** `torch.cuda.amp.autocast` + `GradScaler` on CUDA; FP32 fallback on CPU.
- **Early stopping:** Patience=15 epochs based on validation macro AUC-ROC. Best checkpoint saved to `results/checkpoints/{model}_best.pt`.
- **Logging:** Training loss, validation loss, validation AUC-ROC, and validation macro F1 logged every epoch via `utils/logger.py`.

### 5.5 Hyperparameter Summary

| Hyperparameter | Value |
|---|---|
| Batch size (train) | 32 |
| Batch size (val/test) | 64 |
| Max epochs | 100 |
| Early stopping patience | 15 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Mixup alpha | 0.4 |
| Label smoothing | 0.1 |
| Warmup epochs | 5 |
| Gradient clip norm | 1.0 |

---

## 6. Evaluation Methodology

### 6.1 Metrics (`training/metrics.py`)

All metrics are computed on the held-out test set (fold 10).

| Metric | Description |
|---|---|
| **Macro AUC-ROC** | Primary metric. Mean area under the ROC curve across 5 classes (threshold-independent). |
| **Macro F1** | Mean F1 score across 5 classes at optimal per-class threshold. |
| **Macro AUPRC** | Mean area under the Precision-Recall curve (more informative under class imbalance). |
| **Accuracy** | Exact match ratio of predicted binary vectors to ground truth. |
| **Sensitivity** | Macro mean of per-class recall. |
| **Specificity** | Macro mean of per-class true negative rate. |
| **Per-class breakdown** | AUC, F1, precision, recall, specificity, AUPRC, optimal threshold per class. |

**Threshold optimization:** For each class independently, the decision threshold is selected by grid search over [0.10, 0.91] with step 0.01 on the validation set to maximize per-class F1. These thresholds are then applied to test set predictions.

### 6.2 Statistical Testing

Bootstrap resampling (n=1000 iterations) is used in `notebooks/comparacion_modelos.ipynb` to compute confidence intervals on macro AUC and perform pairwise statistical comparisons between architectures.

---

## 7. Results

### 7.1 Aggregate Metrics (Test Set)

| Model | Params | AUC-ROC ↑ | F1 ↑ | AUPRC ↑ | Accuracy ↑ | Sensitivity ↑ | Specificity ↑ | Inference (ms/sample) | Train time |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet-1D** | 8.9M | 0.9258 | **0.7409** | **0.8141** | **0.7643** | 0.7798 | **0.8988** | **0.214** | 57.6 min |
| CNN-Transformer | 3.7M | **0.9264** | 0.7382 | 0.8081 | 0.7602 | **0.7959** | 0.8874 | 0.366 | 48.3 min |
| PatchTST | 0.82M | 0.9100 | 0.7099 | 0.7728 | 0.7402 | 0.7458 | 0.8939 | 0.705 | 122.1 min |

### 7.2 Per-Class Performance — ResNet-1D

| Class | AUC-ROC | F1 | Precision | Recall | Specificity | AUPRC | Threshold |
|---|---|---|---|---|---|---|---|
| NORM | 0.951 | 0.862 | 0.807 | 0.925 | — | — | — |
| MI | 0.930 | 0.746 | 0.753 | 0.738 | — | — | — |
| STTC | 0.935 | 0.757 | 0.716 | 0.802 | — | — | — |
| CD | 0.909 | 0.756 | 0.759 | 0.754 | — | — | — |
| HYP | 0.903 | 0.584 | 0.511 | 0.679 | — | — | — |
| **Macro** | **0.926** | **0.741** | — | — | **0.899** | **0.814** | — |

### 7.3 Per-Class Performance — CNN-Transformer

| Class | AUC-ROC | F1 | Precision | Recall | Specificity | AUPRC | Threshold |
|---|---|---|---|---|---|---|---|
| NORM | 0.948 | 0.855 | 0.796 | 0.924 | 0.815 | 0.924 | 0.52 |
| MI | 0.932 | 0.744 | 0.717 | 0.773 | 0.898 | 0.838 | 0.63 |
| STTC | 0.936 | 0.763 | 0.697 | 0.844 | 0.890 | 0.810 | 0.72 |
| CD | 0.917 | 0.745 | 0.747 | 0.744 | 0.927 | 0.830 | 0.64 |
| HYP | 0.899 | 0.583 | 0.503 | 0.695 | 0.907 | 0.639 | 0.72 |
| **Macro** | **0.926** | **0.738** | — | — | **0.887** | **0.808** | — |

### 7.4 Per-Class Performance — PatchTST

| Class | AUC-ROC | F1 | Precision | Recall | Specificity | AUPRC | Threshold |
|---|---|---|---|---|---|---|---|
| NORM | 0.936 | 0.847 | 0.836 | 0.860 | 0.868 | 0.902 | 0.63 |
| MI | 0.912 | 0.725 | 0.682 | 0.775 | 0.879 | 0.796 | 0.52 |
| STTC | 0.927 | 0.724 | 0.712 | 0.737 | 0.911 | 0.794 | 0.77 |
| CD | 0.898 | 0.683 | 0.673 | 0.694 | 0.902 | 0.784 | 0.76 |
| HYP | 0.877 | 0.570 | 0.499 | 0.664 | 0.910 | 0.587 | 0.57 |
| **Macro** | **0.910** | **0.710** | — | — | **0.894** | **0.773** | — |

### 7.5 Key Findings

1. **ResNet-1D and CNN-Transformer achieve near-identical macro AUC** (0.9258 vs 0.9264), with no statistically significant difference under bootstrap testing (p > 0.05).
2. **CNN-Transformer is 2.4× smaller** (3.7M vs 8.9M parameters) and trains 15% faster, making it preferable in resource-constrained settings.
3. **PatchTST is the lightest model** (0.82M parameters, 11× smaller than ResNet-1D) but lags ~1.6% in macro AUC (0.910 vs 0.926) and takes 2.1× longer to train (122 vs 58 min), driven by the quadratic attention cost over 100 patches × 12 leads.
4. **NORM is the easiest class across all models** (AUC 0.936–0.951, F1 0.847–0.862) due to high prevalence and a distinctive baseline pattern.
5. **HYP is the most challenging class across all models** (AUC 0.877–0.903, F1 0.570–0.584) — driven by low prevalence and morphological overlap with NORM.
6. **ResNet-1D achieves the highest macro F1** (0.741), suggesting better-calibrated decision boundaries under class imbalance.
7. **PatchTST inference is 3× slower** than ResNet-1D (0.705 vs 0.214 ms/sample), making it least suitable for latency-sensitive deployment despite its small parameter footprint.

---

## 8. Notebooks

All notebooks are located in `notebooks/` and use pre-computed predictions saved in `results/metrics/`.

### `resnet1d.ipynb`
End-to-end ResNet-1D training, evaluation, and visualization.
- Loads dataset, normalizer, and pos_weight
- Trains model or loads best checkpoint
- Runs test evaluation
- Generates: ROC curves (per-class + macro), precision-recall curves, confusion matrices (one per class), training curves (loss + AUC vs epoch), per-class F1/AUC bar charts
- Saves all figures to `results/figures/` in both PDF and PNG formats

### `cnn_transformer.ipynb`
Equivalent notebook for the CNN-Transformer architecture. Mirrors resnet1d.ipynb structure.

### `patchtst.ipynb`
Equivalent notebook for PatchTST. Training and evaluation workflow; some visualizations may be incomplete.

### `comparacion_modelos.ipynb`
Multi-model comparison and statistical analysis. **Does not re-train.** Requires all three individual model notebooks to have been executed first so that their prediction arrays and metrics are present in `results/metrics/`.

**Cell 1 — Load results**
- Verifies existence of all required files for each model: `{model}_metrics.json`, `{model}_y_true.npy`, `{model}_y_pred_proba.npy`, `{model}_y_pred_binary.npy`
- Loads JSON metrics and NumPy arrays into a shared `results` dict keyed by model name
- Prints `macro_auc` for each model as a sanity check

**Cell 2 — Master comparison table**
- Builds a 9-column DataFrame: Model, Macro-AUC, Macro-F1, AUPRC, Accuracy, Sensitivity, Specificity, Params(M), Inference(ms/sample)
- Bold-highlights the best value per column via Pandas Styler
- Exports to `results/metrics/comparison_table.csv` and `results/metrics/comparison_table.tex` (LaTeX with caption and label)

**Cell 3 — Grouped bar chart**
- Plots AUC-ROC, Macro-F1, AUPRC, and Accuracy as grouped bars (one group per metric, one bar per model)
- Annotates each bar with its numeric value
- Saves `results/figures/comparison_metrics_bar.{pdf,png}`

**Cell 4 — Overlaid macro ROC curves**
- Computes per-model macro ROC by interpolating all 5 per-class ROC curves onto a common FPR grid (`np.linspace(0, 1, 200)`) and averaging TPR
- Plots all three models on a single axes with `AUC=x.xxx` in the legend
- Saves `results/figures/comparison_roc_macro.{pdf,png}`

**Cell 5 — Radar / spider chart**
- 5-axis polar chart: Macro-AUC, Macro-F1, AUPRC, Sensitivity, Specificity
- Each model rendered as a filled polygon (α=0.15) plus a border line
- Saves `results/figures/comparison_radar.{pdf,png}`

**Cell 6 — AUC heatmap (3 × 5)**
- Seaborn heatmap of AUC-ROC per model (rows) per class (columns)
- Color scale `YlOrRd`, range [0.5, 1.0]; cells annotated with 3-decimal values
- Saves `results/figures/comparison_auc_heatmap.{pdf,png}`

**Cell 7 — F1 heatmap (3 × 5)**
- Seaborn heatmap of F1 per model (rows) per class (columns)
- Color scale `YlOrRd`, range [0.0, 1.0]
- Saves `results/figures/comparison_f1_heatmap.{pdf,png}`

**Cell 8 — Bootstrap statistical comparison**
- Pairwise bootstrap (n=1000, `np.random.default_rng(42)`) for each model pair
- Two-sided test: `p = 2 × min(P(diff ≤ 0), P(diff ≥ 0))`
- Reports observed AUC difference and p-value; significance encoded as `ns / * / ** / ***` (thresholds: 0.05 / 0.01 / 0.001)
- All pairs tested: ResNet vs PatchTST, ResNet vs CNN-Transformer, PatchTST vs CNN-Transformer

---

## 9. Generated Artifacts

### Checkpoints (`results/checkpoints/`)

| File | Size | Description |
|---|---|---|
| `resnet1d_best.pt` | ~38 MB | Best ResNet-1D by validation AUC (epoch 31) |
| `cnn_transformer_best.pt` | ~15 MB | Best CNN-Transformer by validation AUC |
| `patchtst_best.pt` | ~3 MB | Best PatchTST by validation AUC (0.82M params) |

### Metrics (`results/metrics/`)

Comparison (generated by `comparacion_modelos.ipynb`):
- `comparison_table.csv` — 9-column aggregate metrics table for all models
- `comparison_table.tex` — LaTeX-formatted table (caption + label) for direct inclusion in a paper

Per model:
- `{model}_metrics.json` — aggregate and per-class metrics (AUC, F1, AUPRC, accuracy, sensitivity, specificity)
- `{model}_metrics.csv` — per-class table with optimal thresholds
- `{model}_y_true.npy` — ground truth label matrix (N_test × 5)
- `{model}_y_pred_proba.npy` — predicted probabilities (N_test × 5, after sigmoid)
- `{model}_y_pred_binary.npy` — binarized predictions at optimal thresholds
- `{model}_thresholds.npy` — per-class optimal threshold vector (shape: 5)
- `{model}_norm_stats.npy` — normalization statistics used at test time

### Figures (`results/figures/`)

Per model:
- `{model}_roc_curves.{pdf,png}` — 5-subplot ROC curves (one subplot per class)
- `{model}_pr_curves.{pdf,png}` — precision-recall curves
- `{model}_confusion_matrix.{pdf,png}` — 5 binary confusion matrices
- `{model}_training_curves.{pdf,png}` — training/validation loss and AUC over epochs
- `{model}_per_class_metrics.{pdf,png}` — grouped bar charts of per-class F1 and AUC

Comparison:
- `comparison_metrics_bar.{pdf,png}` — AUC, F1, AUPRC, accuracy grouped by model
- `comparison_roc_macro.{pdf,png}` — macro ROC overlaid for all models
- `comparison_radar.{pdf,png}` — radar chart across 5 aggregate metrics
- `comparison_auc_heatmap.{pdf,png}` — per-class AUC heatmap (models × classes)
- `comparison_f1_heatmap.{pdf,png}` — per-class F1 heatmap (models × classes)

### Preprocessed Artifacts

| File | Generated by | Purpose |
|---|---|---|
| `config/norm_stats.npy` | `scripts/preprocess.py` | Per-lead z-score stats (mean, std), shape (12,) each |
| `data/processed/pos_weight.npy` | `scripts/preprocess.py` | Per-class BCE positive weights, shape (5,) |

---

## 10. Project Structure

```
ecg-classification/
├── config/
│   ├── config.yaml              # All hyperparameters
│   └── norm_stats.npy           # Generated: per-lead normalization stats
│
├── data/
│   ├── dataset.py               # PTBXLDataset (PyTorch Dataset)
│   ├── labels.py                # SCP code → binary label vector
│   ├── preprocessing.py         # Normalizer + augmentation transforms
│   └── processed/
│       └── pos_weight.npy       # Generated: class imbalance weights
│
├── models/
│   ├── resnet1d.py              # ResNet-1D (SE + DropPath)
│   ├── patchtst.py              # Patch-based Transformer
│   ├── patchtst_opt.py          # Optimized PatchTST variant
│   └── cnn_transformer.py       # CNN encoder + Transformer hybrid
│
├── training/
│   ├── trainer.py               # Training loop, checkpointing, early stopping
│   └── metrics.py               # AUC, F1, AUPRC, per-class metrics
│
├── scripts/
│   ├── preprocess.py            # Compute norm_stats and pos_weight
│   ├── train_resnet.py          # ResNet-1D entry point
│   ├── train_patchtst.py        # PatchTST entry point
│   ├── train_cnn_transformer.py # CNN-Transformer entry point
│   └── evaluate.py              # Flexible test-set evaluator
│
├── notebooks/
│   ├── resnet1d.ipynb           # ResNet-1D training + evaluation + figures
│   ├── cnn_transformer.ipynb    # CNN-Transformer training + evaluation + figures
│   ├── patchtst.ipynb           # PatchTST training + evaluation + figures
│   └── comparacion_modelos.ipynb # Multi-model comparison + statistical tests
│
├── utils/
│   └── logger.py                # Centralized stdout + file logging
│
├── results/
│   ├── checkpoints/             # Best model .pt files
│   ├── metrics/                 # Per-model JSON/CSV/NPY metrics and predictions
│   └── figures/                 # Publication-ready PDF + PNG figures
│
├── logs/
│   ├── train_resnet.log
│   ├── train_cnn_transformer.log
│   ├── train_patchtst.log
│   └── preprocess.log
│
├── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
│   ├── ptbxl_database.csv
│   ├── scp_statements.csv
│   └── records500/              # WFDB .hea/.dat files
│
├── CLAUDE.md                    # Project instructions for Claude Code
└── ecg-classification.md        # This document
```

---

## 11. Technology Stack

| Category | Library / Tool | Version | Usage |
|---|---|---|---|
| Deep learning | PyTorch | ≥ 2.0.0 | Models, training loop, mixed precision |
| ECG I/O | wfdb | ≥ 4.1.0 | Read WFDB records from records500/ |
| Numerical | NumPy | ≥ 1.24.0 | Array ops, norm stats, label vectors |
| Data manipulation | Pandas | ≥ 2.0.0 | Load ptbxl_database.csv, scp_statements.csv |
| Evaluation | scikit-learn | ≥ 1.3.0 | ROC, F1, AUPRC, confusion matrix |
| Configuration | PyYAML | ≥ 6.0 | Load config/config.yaml |
| Progress | tqdm | ≥ 4.65.0 | Epoch/batch progress bars |
| Visualization | Matplotlib, Seaborn | — | ROC curves, heatmaps, radar charts |
| Notebooks | Jupyter | — | Interactive training and comparison |
| Language | Python | 3.10+ | Type hints, pathlib |
| GPU acceleration | CUDA / AMP | — | Mixed precision, GradScaler |

---

## 12. Reproducibility

### Fixed Practices
- **No re-shuffling:** Dataset split uses `strat_fold` from PTB-XL directly (folds 1-8/9/10).
- **Normalization leakage prevention:** Stats fitted on training fold only.
- **Saved artifacts:** Normalization stats, pos_weight, and best checkpoints are all persisted.
- **Deterministic evaluation:** Threshold optimization on validation set, applied once to test set.
- **Logged hyperparameters:** All training runs produce a `.log` file with model config and epoch metrics.

### Running Order

```powershell
# Step 1 — compute normalization stats and class weights (~7 min)
python scripts/preprocess.py `
    --data_path "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" `
    --output_path data/processed/ `
    --sampling_rate 500

# Step 2 — train ResNet-1D (up to 100 epochs, early stopping at patience=15)
python scripts/train_resnet.py `
    --data_path "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" `
    --norm_stats config/norm_stats.npy `
    --pos_weight data/processed/pos_weight.npy `
    --sampling_rate 500

# Step 3 — train CNN-Transformer
python scripts/train_cnn_transformer.py `
    --data_path "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" `
    --norm_stats config/norm_stats.npy `
    --pos_weight data/processed/pos_weight.npy `
    --sampling_rate 500

# Step 4 — train PatchTST
python scripts/train_patchtst.py `
    --data_path "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" `
    --norm_stats config/norm_stats.npy `
    --pos_weight data/processed/pos_weight.npy `
    --sampling_rate 500

# Step 5 — evaluation and figures (Jupyter notebooks)
# Run notebooks/resnet1d.ipynb, cnn_transformer.ipynb, patchtst.ipynb
# Then run notebooks/comparacion_modelos.ipynb for comparison
```

---

## 13. Design Conventions

- **Type hints:** All functions use Python 3.10+ annotations throughout.
- **pathlib.Path everywhere:** No hardcoded string paths; all filesystem operations via `pathlib`.
- **Smoke tests:** Every module includes an `if __name__ == "__main__"` block for standalone testing.
- **Logits-only forward pass:** All models return raw logits; sigmoid and threshold are applied only at evaluation time, ensuring numerical stability with BCEWithLogitsLoss.
- **Split-aware augmentation:** PTBXLDataset applies augmentation transforms only in training mode, preventing val/test contamination.
- **Streaming normalization:** Constant O(1) memory normalization fitting independent of dataset size.
- **Per-class positive weights:** Directly passed to BCEWithLogitsLoss to handle class imbalance without oversampling.
- **Threshold optimization per class:** Thresholds independently tuned on validation set, respecting the multi-label nature of the problem.
