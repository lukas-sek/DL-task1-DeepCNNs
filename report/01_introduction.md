# 1. Introduction

The MAMe (Museum Art Medium) dataset [1] contains 37,407 images from three major art museums
(Metropolitan Museum of Art, Los Angeles County Museum of Art, Cleveland Museum of Art),
categorised by art experts into 29 medium classes representing materials and techniques such as
oil on canvas, marble, etching, and woven fabric. All images are provided in a downsampled
256×256 version. The dataset uses a balanced training split (20,300 images, 700 per class),
a balanced validation split (1,450 images, 50 per class), and an unbalanced test split (15,657
images). The task is 29-class image classification evaluated by top-1 accuracy.

This report studies Deep Convolutional Neural Network (CNN) configurations on this task.
For each architecture we follow a three-phase methodology: (1) select hyperparameters that
produce underfitting, confirming the model is too small; (2) select hyperparameters that
produce overfitting, confirming the model has sufficient capacity; (3) apply regularisation
techniques to maximise generalisation. We test:

- **Standard CNN** (Section 3): a custom Conv→BN→ReLU→Pool stack designed from scratch.
- **Non-standard CNN** (Section 4): a custom residual network with skip connections.
- **Transfer learning** (Section 5, optional): a pretrained ResNet50 backbone fine-tuned on MAMe.

All experiments run on BSC MareNostrum5 (NVIDIA GPU nodes) using PyTorch. Code is included in
the submission.
