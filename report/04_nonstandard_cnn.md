# 4. Non-Standard CNN Architecture (Residual)

<!-- STATUS: WAITING FOR EXPERIMENTS — res_underfit, res_overfit, res_best -->

## Architecture Description & Justification
<!-- TODO: fill in after architecture is finalised in train.py -->
We implement a custom residual network. Unlike the standard stack, residual blocks add a skip
connection from input to output: `out = F(x) + x`. This allows gradients to flow directly through
the network, enabling deeper architectures without vanishing gradients.

Justification for this choice: MAMe images exhibit fine-grained texture differences (e.g., oil on
canvas vs. woven fabric). Deeper networks with skip connections can learn hierarchical texture
features more effectively than a shallow standard CNN.

Input: 128×128×3. The `residual` config is used for the best run; `residual_large` for overfitting; `residual_small` for underfitting.

**residual** (best config — channels [64, 128, 256, 512]):

| Block | Type | Stride | Output size | Channels |
|-------|------|--------|------------|---------|
| Stem Conv3×3 + BN + ReLU | Standard | 1 | 128×128 | 64 |
| ResBlock ×2 | Residual | 2,1 | 64×64 | 128 |
| ResBlock ×2 | Residual | 2,1 | 32×32 | 256 |
| ResBlock ×2 | Residual | 2,1 | 16×16 | 512 |
| Global Avg Pool | — | — | 1×1 | 512 |
| Dropout + FC / output | — | — | 29 | — |

Each ResBlock: `Conv3×3 → BN → ReLU → Conv3×3 → BN → (+shortcut) → ReLU`.
Shortcut uses 1×1 conv when channels or spatial dims change.
Total parameters: ~6.6 M  <!-- verify from train.py output -->

## Phase 1 — Underfitting
<!-- POPULATED BY RALPH LOOP after res_underfit results are in -->
- Training accuracy: <!-- TODO -->
- Validation accuracy: <!-- TODO -->

## Phase 2 — Overfitting
<!-- POPULATED BY RALPH LOOP after res_overfit results are in -->
- Training accuracy: <!-- TODO -->
- Validation accuracy: <!-- TODO -->
- Train/val gap: <!-- TODO -->

## Phase 3 — Regularization & Best Result
<!-- POPULATED BY RALPH LOOP after res_best results are in -->
- Best validation accuracy: <!-- TODO -->
- Test accuracy: <!-- TODO -->

![Loss curve](../figures/res_best_curves.pdf)
![Confusion matrix](../figures/res_best_confusion.pdf)
