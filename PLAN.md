# Practical Work 1 — Deep CNNs on MAMe Dataset
## Master Plan

---

## 0. Overview & Deliverables

| Item | Description |
|------|-------------|
| Dataset | MAMe 256×256 (37,407 images, 29 classes) |
| Required models | (1) Standard CNN from scratch, (2) Non-standard CNN from scratch |
| Optional extras | Transfer learning, known architecture replication |
| Report | Max 9 pages (architecture descriptions, training details, results, conclusions) |
| Delivery | Report PDF + source code |

---

## 1. Environment & Infrastructure

### 1.1 BSC MareNostrum5 Setup (do this first)
- Log in via `ssh nct01204@alogin1.bsc.es`
- Create working directory: `~/DL-Lab-CNN/`
- Download MAMe 256×256 from Kaggle and upload to BSC via `scp`
- Verify GPU allocation: queue `acc_training`, account `nct_367`
- Run a minimal 1-epoch test job immediately to confirm environment works and get runtime estimates

### 1.2 Dataset on BSC
- Place dataset in a scratch or GPFS path (NOT home, which is slow)
- Structure expected:
  ```
  MAMe/
    train/   (20,300 images, 700/class × 29 classes)
    val/     (1,450 images, 50/class × 29 classes)
    test/    (15,657 images, unbalanced)
  ```
- Verify class label mapping matches the official MAMe metadata CSV

### 1.3 Local Development
- Use the MNIST PyTorch example as a template for your data loading + training loop
- Develop and debug locally on a tiny data subset (e.g., 3 classes, 50 images each)
- Only run full experiments on BSC

---

## 2. Data Pipeline

### 2.1 Loading
- Use PyTorch `ImageFolder` or a custom `Dataset` class reading the MAMe metadata CSV
- Split: use the official train/val/test splits (do NOT create your own splits)

### 2.2 Preprocessing (baseline, no augmentation)
- Resize to fixed input size (e.g., 128×128 or 224×224 depending on model capacity)
- Normalize: compute mean and std on the training set per channel; apply to all splits
- Convert to float32 tensors

### 2.3 Data Augmentation (for regularization phase)
Options to explore (apply only to training set):
- Random horizontal flip
- Random crop / resize crop
- Color jitter (brightness, contrast, saturation)
- Random rotation (small angles, e.g., ±15°)
- Random erasing / cutout

> Augmentation is a hyperparameter — track which augmentations are active for each experiment.

---

## 3. Model Architectures

### 3.1 Standard Architecture (implemented from scratch)
**Definition:** A standard CNN follows the classic pattern:
`[Conv → Activation → (Pool)] × N → Flatten → FC layers → Softmax`

Design choices to document:
- Number of conv blocks (depth)
- Number of filters per layer (width), typically doubling: 32 → 64 → 128 → 256
- Kernel sizes (typically 3×3)
- Pooling strategy: MaxPool 2×2 after each block
- FC layer sizes
- Activation function: ReLU throughout, Softmax at output

**Guiding principle:** Design it yourself — do not just copy VGG/AlexNet. Take inspiration but make deliberate design decisions.

**Variants to test:**
- Small (underfit): 2–3 conv blocks, small widths, no regularization
- Large (overfit): 4–5 conv blocks, larger widths, no regularization
- Tuned: add BatchNorm, Dropout, data augmentation

---

### 3.2 Non-Standard Architecture (implemented from scratch)
**Definition:** Departures from the simple Conv→Pool stack. Options:

**Option A — Residual-style connections (ResNet-inspired)**
- Add skip connections between conv blocks
- Allows much deeper networks without vanishing gradients
- Design your own block structure (do not replicate ResNet exactly)

**Option B — Inception-style parallel branches**
- Each block has parallel 1×1, 3×3, 5×5 convolutions whose outputs are concatenated
- Captures multi-scale features
- Design your own simplified version

**Option C — Depthwise separable convolutions (MobileNet-inspired)**
- Replace standard conv with depthwise + pointwise convolutions
- Much lower parameter count for same receptive field

> Pick ONE non-standard type and justify the choice in the report.

---

### 3.3 Optional: Transfer Learning
- Use a pretrained backbone (e.g., ResNet50, EfficientNet-B0) from `torchvision.models`
- Strategy A (feature extraction): freeze backbone, train only the new classification head
- Strategy B (fine-tuning): unfreeze last N blocks, use a low learning rate
- Replace the final FC layer to output 29 classes
- Compare against from-scratch models to quantify the value of pretraining on ImageNet

---

## 4. Training Methodology

### 4.1 The 3-Phase Workflow (required for each model)

**Phase 1 — Induce Underfitting**
Goal: confirm the model is too small / undertrained to memorize the training set.
- Use a very small model (few layers, few filters)
- Train for a small number of epochs
- Expected: train accuracy low, val accuracy similar to train → underfitting confirmed
- Document: architecture config, epochs, optimizer, LR, train/val curves

**Phase 2 — Induce Overfitting**
Goal: confirm the model has enough capacity to memorize the training set.
- Use a large model (more layers, more filters)
- Train without any regularization (no dropout, no augmentation)
- Expected: train accuracy high (>90%), val accuracy much lower → overfitting confirmed
- Document: same as above

**Phase 3 — Regularize to Find Best Generalization**
Goal: close the gap between train and val accuracy.
Techniques to apply one-by-one (to understand each one's contribution):
- Batch Normalization (after conv layers, before activation)
- Dropout (after FC layers, rate ≈ 0.3–0.5)
- L2 weight decay (in optimizer, e.g., 1e-4)
- Data augmentation (see Section 2.3)
- Learning rate scheduling (step decay or cosine annealing)
- Early stopping based on val loss

---

### 4.2 Optimization

| Hyperparameter | Candidates to try | Notes |
|---|---|---|
| Optimizer | SGD+momentum, Adam, AdamW | Start with Adam for speed |
| Learning rate | 1e-2, 1e-3, 1e-4 | Use LR finder or start at 1e-3 |
| LR schedule | StepLR, CosineAnnealing, ReduceLROnPlateau | |
| Batch size | 32, 64, 128 | Larger = faster but coarser gradients |
| Epochs | 20–100 depending on model | Use early stopping |
| Weight init | Kaiming (He) for ReLU, Xavier otherwise | PyTorch default is usually fine |
| Momentum | 0.9 (for SGD) | Standard |
| Weight decay | 0, 1e-5, 1e-4 | L2 regularization |

---

### 4.3 Weights Initialization
- Standard: PyTorch default (Kaiming uniform for conv layers)
- For custom layers: use `torch.nn.init.kaiming_normal_` for ReLU networks
- Document initialization choice per experiment

---

## 5. Evaluation & Metrics

### 5.1 Primary Metrics
- **Top-1 Accuracy** on validation set (main metric for model selection)
- **Top-1 Accuracy** on test set (report only for final selected model per architecture)
- **Loss curves** (train + val per epoch) for every experiment

### 5.2 Secondary Analysis
- **Confusion matrix** on test set for the best model — identify which classes are most confused
- **Per-class accuracy** — the test set is unbalanced, so overall accuracy may be misleading
- If time allows: **t-SNE** visualization of features from the penultimate layer

### 5.3 What to Report Per Experiment
For every run, log and save:
- Model architecture (config)
- All hyperparameters
- Best val accuracy + epoch at which it was achieved
- Final test accuracy (only for selected models)
- Train/val loss and accuracy curves (save as plots)

---

## 6. Hyperparameter Selection Strategy

1. Start with a fixed small architecture and vary one hyperparameter at a time
2. Use validation accuracy as the selection criterion — never touch test set during tuning
3. Coarse search first (e.g., LR ∈ {1e-2, 1e-3, 1e-4}), then fine-tune the best region
4. For architecture depth/width: double or halve at a time
5. For regularization: add techniques incrementally and measure each one's impact
6. Keep a log/table of all experiments (config → val acc)

---

## 7. Experiment Schedule

| Step | Experiment | Goal |
|------|-----------|------|
| 1 | Data pipeline test (1 epoch, tiny subset) | Confirm pipeline works on BSC |
| 2 | Standard CNN — underfitting config | Phase 1 |
| 3 | Standard CNN — overfitting config | Phase 2 |
| 4 | Standard CNN — regularization sweep | Phase 3 |
| 5 | Non-standard CNN — underfitting config | Phase 1 |
| 6 | Non-standard CNN — overfitting config | Phase 2 |
| 7 | Non-standard CNN — regularization sweep | Phase 3 |
| 8 | (Optional) Transfer learning baseline | Comparison |
| 9 | (Optional) Transfer learning fine-tuned | Best result |
| 10 | Final evaluation on test set for best configs | Report numbers |

> Run steps 1–3 first before committing to a full architecture for Phase 3.

---

## 8. BSC Job Management

- Each experiment = one `sbatch` job
- Use job arrays for hyperparameter sweeps (`#SBATCH --array=0-N`)
- Save all outputs (model checkpoints, logs, plots) to a named experiment folder
- Monitor with `squeue -u nct01204`
- Estimate time per epoch on BSC before launching long jobs

### Launcher Template Variables to Set
- `--gres=gpu:1` (one GPU per job)
- `--time=HH:MM:SS` (set conservatively — estimate from short test run)
- `--ntasks-per-node=1`
- Environment: load the appropriate Python/PyTorch module

---

## 9. Report Structure (9 pages max)

1. **Introduction** (0.5 page) — dataset summary, task definition, metrics used
2. **Data Preprocessing & Augmentation** (0.5 page) — normalization, augmentation choices, justification
3. **Standard Architecture** (2 pages) — diagram or table of architecture, underfitting/overfitting evidence (curves), regularization choices, final results
4. **Non-Standard Architecture** (2 pages) — same structure as above, plus justification for choosing this type
5. **(Optional) Transfer Learning** (1 page) — backbone used, fine-tuning strategy, results
6. **Comparison Table** (0.5 page) — all final models side by side: val acc, test acc, #params, training time
7. **Discussion & Conclusions** (1 page) — what worked, what didn't, lessons learned, open questions
8. **LLM Declaration** (1–2 lines) — which parts used LLM assistance (required by statement)
9. **References** (remaining space)
10. **Appendix** (if needed) — extra plots, full hyperparameter tables

---

## 10. Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| BSC queue wait times | Submit jobs early; have local debug runs ready |
| Training too slow | Reduce input resolution to 128×128 for early experiments |
| Overfitting hard to achieve | Increase model size, train longer, remove all regularization |
| Underfitting hard to confirm | Use very small model (1–2 blocks), few epochs (5–10) |
| Test set leakage | Only evaluate on test set for final selected configs |
| Unbalanced test set | Report per-class accuracy alongside overall accuracy |
| Report over 9 pages | Write section drafts early; trim figures to the most informative |
