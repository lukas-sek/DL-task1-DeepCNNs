# 2. Data Preprocessing

<!-- STATUS: READY TO WRITE — no experiment data needed -->

## Splits
Official train/val/test splits are used without modification. The test set is unbalanced and is only
evaluated for the final selected configuration per architecture.

## Normalization
All images are resized to a fixed input size (128×128 or 224×224 depending on model) using
bilinear interpolation. Pixel values are normalized per channel using the ImageNet statistics:
mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]. This is applied to all splits.

<!-- TODO: if we compute dataset-specific mean/std, update here -->

## Data Augmentation (regularization phase only)
Applied exclusively to the training set:
- Random horizontal flip (p=0.5)
- Random resized crop (scale 0.8–1.0)
- Color jitter (brightness ±0.3, contrast ±0.3, saturation ±0.2)
- Random rotation ±15°

Augmentation is disabled for underfitting and overfitting phases to isolate model capacity effects.

## Class Imbalance
Training and validation sets are balanced (700 and 50 per class respectively). The test set is
unbalanced — we report both overall accuracy and per-class accuracy on the test set.
