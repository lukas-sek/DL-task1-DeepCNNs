# 3. Standard CNN Architecture

<!-- STATUS: WAITING FOR EXPERIMENTS — std_underfit, std_overfit, std_best -->

## Architecture Description
A standard CNN following the classic `Conv → BN → ReLU → MaxPool` stack.
Input: 128×128×3. The overfitting and best configurations use `standard_large`;
the underfitting configuration uses `standard_small`.

**standard_large** (overfitting / best config):

| Block | Output size | Filters | Kernel | BN |
|-------|------------|---------|--------|----|
| Conv1 + Pool | 64×64 | 32 | 3×3 | ✓ |
| Conv2 + Pool | 32×32 | 64 | 3×3 | ✓ |
| Conv3 + Pool | 16×16 | 128 | 3×3 | ✓ |
| Conv4 + Pool | 8×8 | 256 | 3×3 | ✓ |
| Conv5 + Pool | 4×4 | 256 | 3×3 | ✓ |
| FC1 (ReLU + Dropout) | 512 | — | — | — |
| FC2 / output | 29 | — | — | — |

Flattened size entering FC: 256 × 4 × 4 = 4,096.
Total parameters (standard_large, no dropout): ~4.6 M  <!-- verify with print from train.py -->

**standard_small** (underfitting config): channels [32, 64, 128], FC 256 — ~0.6 M params.

## Phase 1 — Underfitting
<!-- POPULATED BY RALPH LOOP after std_underfit results are in -->
Config: <!-- TODO -->
- Training accuracy: <!-- TODO -->
- Validation accuracy: <!-- TODO -->
- Observation: <!-- TODO -->

![Loss curve](../figures/std_underfit_curves.pdf)
*Figure X: Training and validation loss/accuracy for the underfitting configuration.*

## Phase 2 — Overfitting
<!-- POPULATED BY RALPH LOOP after std_overfit results are in -->
Config: <!-- TODO -->
- Training accuracy: <!-- TODO -->
- Validation accuracy: <!-- TODO -->
- Train/val gap: <!-- TODO -->
- Observation: <!-- TODO -->

![Loss curve](../figures/std_overfit_curves.pdf)

## Phase 3 — Regularization & Best Result
<!-- POPULATED BY RALPH LOOP after std_best results are in -->
Techniques applied: BatchNorm, Dropout (p=<!-- TODO -->), data augmentation, weight decay=<!-- TODO -->

- Best validation accuracy: <!-- TODO --> (epoch <!-- TODO -->)
- Test accuracy: <!-- TODO -->

![Loss curve](../figures/std_best_curves.pdf)
![Confusion matrix](../figures/std_best_confusion.pdf)
