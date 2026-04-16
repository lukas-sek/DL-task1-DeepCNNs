# 5. Transfer Learning (Optional)

<!-- STATUS: OPTIONAL — populate if transfer_frozen / transfer_finetune experiments run -->

## Setup
Pretrained backbone: ResNet18 (ImageNet weights via torchvision)
Input size: 224×224 (typical for ResNet)

**Strategy A — Feature extraction:** backbone frozen, only the classification head trained.
**Strategy B — Full fine-tuning:** all layers unfrozen, trained with lr=1e-4.

## Results
<!-- POPULATED BY RALPH LOOP after transfer experiments are in -->

| Strategy | Val acc | Test acc | Epochs | Notes |
|----------|---------|----------|--------|-------|
| Frozen | <!-- TODO --> | — | <!-- TODO --> | |
| Fine-tuned | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | |

## Discussion
<!-- TODO: compare against from-scratch models — does ImageNet pretraining help for art textures? -->
