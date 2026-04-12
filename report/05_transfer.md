# 5. Transfer Learning (Optional)

<!-- STATUS: OPTIONAL — populate if transfer_frozen / transfer_finetune experiments run -->

## Setup
Pretrained backbone: ResNet50 (ImageNet weights via torchvision)
Input size: 224×224 (required by ResNet)

**Strategy A — Feature extraction:** backbone frozen, only the classification head trained.
**Strategy B — Fine-tuning:** last 2 residual blocks unfrozen, trained with lr=1e-4; head trained with lr=1e-3.

## Results
<!-- POPULATED BY RALPH LOOP after transfer experiments are in -->

| Strategy | Val acc | Test acc | Epochs | Notes |
|----------|---------|----------|--------|-------|
| Frozen | <!-- TODO --> | — | <!-- TODO --> | |
| Fine-tuned | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | |

## Discussion
<!-- TODO: compare against from-scratch models — does ImageNet pretraining help for art textures? -->
