#!/usr/bin/env python
"""
MAMe CNN Trainer
----------------
Main training script for Practical Work 1.
Supports standard and non-standard CNN architectures with configurable hyperparameters.

Usage examples:
    # Pipeline test (1 epoch, tiny model)
    python train.py --data_dir data --epochs 1 --model tiny --batch_size 64

    # Standard CNN — underfitting config
    python train.py --data_dir data --epochs 30 --model standard_small

    # Standard CNN — overfitting config
    python train.py --data_dir data --epochs 60 --model standard_large --no_augment

    # Standard CNN — regularized
    python train.py --data_dir data --epochs 60 --model standard_large --augment --dropout 0.5 --weight_decay 1e-4

    # Non-standard CNN (residual blocks)
    python train.py --data_dir data --epochs 60 --model residual --augment --dropout 0.4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from mame_dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD


# ════════════════════════════════════════════════════════════════════════════
# Architectures
# ════════════════════════════════════════════════════════════════════════════

class StandardCNN(nn.Module):
    """
    Standard CNN: Conv → BN → ReLU → Pool repeated N times, then FC layers.

    Parameters
    ----------
    channels : list of int
        Number of filters per conv block, e.g. [32, 64, 128].
    fc_size : int
        Size of the hidden fully-connected layer.
    dropout : float
        Dropout probability after the FC hidden layer (0 = disabled).
    input_size : int
        Spatial input resolution (square). Used to infer FC input dimension.
    use_batchnorm : bool
        Whether to add BatchNorm after each conv layer.
    """

    def __init__(self, channels, fc_size, num_classes, dropout=0.0, input_size=128, use_batchnorm=True):
        super().__init__()

        in_ch = 3
        conv_blocks = []
        for out_ch in channels:
            block = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if use_batchnorm:
                block.append(nn.BatchNorm2d(out_ch))
            block.append(nn.ReLU(inplace=True))
            block.append(nn.MaxPool2d(2, 2))
            conv_blocks.append(nn.Sequential(*block))
            in_ch = out_ch

        self.features = nn.Sequential(*conv_blocks)

        # Compute flattened size after all pooling
        spatial = input_size // (2 ** len(channels))
        flat_size = channels[-1] * spatial * spatial

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ─── Residual block ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Simple residual block: two 3×3 conv layers with a skip connection.
    If in_channels != out_channels a 1×1 conv adapts the shortcut.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResidualCNN(nn.Module):
    """
    Non-standard CNN using residual blocks.
    Architecture: stem conv → N residual stages (each doubles channels, halves spatial) → GAP → FC
    """

    def __init__(self, channels, num_classes, dropout=0.0, input_size=128):
        super().__init__()

        # Stem: initial conv to get to first channel size
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Residual stages — each stage has 2 residual blocks; first block uses stride=2 to downsample
        stages = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            stages.append(ResidualBlock(in_ch, out_ch, stride=2))
            stages.append(ResidualBlock(out_ch, out_ch, stride=1))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # Global Average Pooling collapses spatial dims → no fixed FC input size
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x)
        return self.classifier(x)


# ─── Model factory ──────────────────────────────────────────────────────────

def build_model(name: str, dropout: float, input_size: int, use_batchnorm: bool, num_classes: int = 29) -> nn.Module:
    """
    name options:
        tiny            — 2-block standard CNN, minimal width (underfitting baseline)
        standard_small  — 3-block standard CNN, small width  (underfitting config)
        standard_large  — 5-block standard CNN, large width  (overfitting config)
        residual_small  — residual CNN, small  (underfitting config)
        residual        — residual CNN, medium (main non-standard model)
        residual_large  — residual CNN, large  (overfitting config)
    """
    configs = {
        "tiny":            {"type": "standard", "channels": [16, 32],                "fc": 128},
        "standard_small":  {"type": "standard", "channels": [32, 64, 128],           "fc": 256},
        "standard_large":  {"type": "standard", "channels": [32, 64, 128, 256, 256], "fc": 512},
        "residual_small":  {"type": "residual", "channels": [32, 64, 128]},
        "residual":        {"type": "residual", "channels": [64, 128, 256, 512]},
        "residual_large":  {"type": "residual", "channels": [64, 128, 256, 512, 512]},
    }
    if name not in configs:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(configs.keys())}")

    cfg = configs[name]
    if cfg["type"] == "standard":
        return StandardCNN(
            channels=cfg["channels"],
            fc_size=cfg["fc"],
            num_classes=num_classes,
            dropout=dropout,
            input_size=input_size,
            use_batchnorm=use_batchnorm,
        )
    else:
        return ResidualCNN(channels=cfg["channels"], num_classes=num_classes, dropout=dropout, input_size=input_size)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════════════
# Training & evaluation loops  (same structure as MNIST example)
# ════════════════════════════════════════════════════════════════════════════

def model_train(device, model, data_loader, loss_fn, optimizer, epoch):
    model.train()
    train_loss = 0.0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        scores  = model(inputs)
        loss_b  = loss_fn(scores, targets)

        optimizer.zero_grad()
        loss_b.backward()
        optimizer.step()

        train_loss += loss_b.item()
        preds       = scores.argmax(dim=1)
        correct    += preds.eq(targets).sum().item()

        if batch_idx % 50 == 49:
            print(f"    [epoch {epoch+1}, batch {batch_idx+1}] "
                  f"running loss: {train_loss / (batch_idx+1):.4f}", flush=True)

    train_loss    /= len(data_loader)
    train_accuracy = 100.0 * correct / len(data_loader.dataset)
    return train_loss, train_accuracy


def model_eval(device, model, data_loader, loss_fn, return_preds=False):
    model.eval()
    eval_loss = 0.0
    correct   = 0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            scores  = model(inputs)
            eval_loss += loss_fn(scores, targets).item()
            preds = scores.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            if return_preds:
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

    eval_loss    /= len(data_loader)
    eval_accuracy = 100.0 * correct / len(data_loader.dataset)
    if return_preds:
        return eval_loss, eval_accuracy, all_targets, all_preds
    return eval_loss, eval_accuracy


def model_fit(device, model, loaders, loss_fn, optimizer, scheduler, num_epochs, out_dir):
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss, train_acc = model_train(device, model, loaders["train"], loss_fn, optimizer, epoch)
        val_loss,   val_acc   = model_eval(device, model, loaders["val"],   loss_fn)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  "
              f"({elapsed:.1f}s)", flush=True)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pth")

    print(f"\nBest val accuracy: {best_val_acc:.2f}%")
    return history


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train a CNN on the MAMe dataset.")

    # Data
    p.add_argument("--data_dir",    type=str,   default="data",   help="Root data directory")
    p.add_argument("--input_size",  type=int,   default=128,      help="Input image size (square)")
    p.add_argument("--augment",     action="store_true",          help="Enable data augmentation")
    p.add_argument("--num_workers", type=int,   default=4)

    # Model
    p.add_argument("--model",       type=str,   default="standard_small",
                   choices=["tiny", "standard_small", "standard_large",
                             "residual_small", "residual", "residual_large"])
    p.add_argument("--dropout",     type=float, default=0.0,      help="Dropout probability (0=off)")
    p.add_argument("--no_batchnorm",action="store_true",          help="Disable BatchNorm (standard CNN only)")

    # Optimizer
    p.add_argument("--optimizer",   type=str,   default="adam",   choices=["adam", "sgd", "adamw"])
    p.add_argument("--lr",          type=float, default=1e-3,     help="Learning rate")
    p.add_argument("--momentum",    type=float, default=0.9,      help="SGD momentum")
    p.add_argument("--weight_decay",type=float, default=0.0,      help="L2 weight decay")
    p.add_argument("--scheduler",   type=str,   default="none",   choices=["none", "cosine"])

    # Training
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--epochs",      type=int,   default=30)

    # Output
    p.add_argument("--out_dir",     type=str,   default="runs/default",
                   help="Directory to save checkpoints and logs")

    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("\nLoading data ...")
    loaders, _, class_names = get_dataloaders(
        data_dir    = args.data_dir,
        input_size  = args.input_size,
        batch_size  = args.batch_size,
        augment     = args.augment,
        num_workers = args.num_workers,
        mean        = IMAGENET_MEAN,
        std         = IMAGENET_STD,
    )
    num_classes = len(class_names)
    print(f"  train: {len(loaders['train'].dataset)} images")
    print(f"  val:   {len(loaders['val'].dataset)} images")
    print(f"  test:  {len(loaders['test'].dataset)} images")
    print(f"  classes ({num_classes}): {class_names[:5]} ...")

    # Model
    model = build_model(
        name         = args.model,
        dropout      = args.dropout,
        input_size   = args.input_size,
        use_batchnorm= not args.no_batchnorm,
        num_classes  = num_classes,
    ).to(device)

    n_params = count_parameters(model)
    print(f"\nModel: {args.model}  |  parameters: {n_params:,}")

    # Optimizer
    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), momentum=args.momentum, **opt_kwargs)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), **opt_kwargs)
    else:
        optimizer = optim.Adam(model.parameters(), **opt_kwargs)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = nn.CrossEntropyLoss()

    # Training
    print(f"\nTraining for {args.epochs} epoch(s) ...")
    t_start = time.time()
    history = model_fit(device, model, loaders, loss_fn, optimizer, scheduler, args.epochs, out_dir)
    t_total = time.time() - t_start

    print(f"\nTotal training time: {t_total:.1f}s  ({t_total/args.epochs:.1f}s/epoch)")

    # Load best model for final evaluation
    best_ckpt = out_dir / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print("\nLoaded best checkpoint for final evaluation.")

    # Final test evaluation with confusion matrix
    print("\nEvaluating on test set ...")
    test_loss, test_acc, true_labels, pred_labels = model_eval(
        device, model, loaders["test"], loss_fn, return_preds=True
    )
    print(f"  test_loss={test_loss:.4f}  test_acc={test_acc:.2f}%")

    # Per-class accuracy
    num_cls = len(class_names)
    class_correct = [0] * num_cls
    class_total   = [0] * num_cls
    for t, p in zip(true_labels, pred_labels):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1
    per_class_acc = {
        class_names[i]: round(100.0 * class_correct[i] / class_total[i], 2)
        if class_total[i] > 0 else 0.0
        for i in range(num_cls)
    }

    # Confusion matrix (num_classes × num_classes list of lists)
    conf_matrix = [[0] * num_cls for _ in range(num_cls)]
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t][p] += 1

    # Save results
    results = {
        "config":           vars(args),
        "n_parameters":     n_params,
        "history":          history,
        "best_val_acc":     max(history["val_accuracy"]),
        "test_acc":         test_acc,
        "training_time_s":  t_total,
        "time_per_epoch_s": t_total / args.epochs,
        "class_names":      class_names,
        "per_class_acc":    per_class_acc,
        "confusion_matrix": conf_matrix,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    print(f"  best val acc : {max(history['val_accuracy']):.2f}%")
    print(f"  test acc     : {test_acc:.2f}%")
    print(f"  per-class acc (top 5 worst):")
    worst = sorted(per_class_acc.items(), key=lambda x: x[1])[:5]
    for cls, acc in worst:
        print(f"    {cls:<30} {acc:.1f}%")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
