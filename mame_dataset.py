import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

NUM_CLASSES = 29 


def get_transforms(input_size: int, augment: bool, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    
    normalize = transforms.Normalize(mean=mean, std=std)

    eval_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_size + 20, input_size + 20)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = eval_transform

    return train_transform, eval_transform


def get_dataloaders(
    data_dir: str | Path,
    input_size: int = 128,
    batch_size: int = 64,
    augment: bool = False,
    num_workers: int = 4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    data_dir = Path(data_dir)
    train_tf, eval_tf = get_transforms(input_size, augment, mean, std)

    datasets_dict = {
        "train": datasets.ImageFolder(data_dir / "train", transform=train_tf),
        "val":   datasets.ImageFolder(data_dir / "val",   transform=eval_tf),
        "test":  datasets.ImageFolder(data_dir / "test",  transform=eval_tf),
    }

    class_names = datasets_dict["train"].classes
    if len(class_names) != NUM_CLASSES:
        print(f"  INFO: found {len(class_names)} classes (full MAMe has {NUM_CLASSES})")

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for split, ds in datasets_dict.items()
    }

    return loaders, datasets_dict, class_names

def compute_normalization_stats(data_dir: str | Path, input_size: int = 128, batch_size: int = 64):
    data_dir = Path(data_dir)
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir / "train", transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Computing normalization stats over training set ...")
    t0 = time.time()
    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n    = 0
    for images, _ in loader:
        # images: (B, C, H, W)
        mean += images.mean(dim=(0, 2, 3)) * images.size(0)
        std  += images.std(dim=(0, 2, 3))  * images.size(0)
        n    += images.size(0)
    mean /= n
    std  /= n
    print(f"  mean={mean.tolist()}  std={std.tolist()}  ({time.time()-t0:.1f}s)")
    return mean.tolist(), std.tolist()

def dataset_summary(data_dir: str | Path):
    """Print a quick summary of the dataset directory."""
    data_dir = Path(data_dir)
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  {split}: NOT FOUND")
            continue
        classes = [d for d in split_dir.iterdir() if d.is_dir()]
        n_images = sum(
            1 for c in classes
            for f in c.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"  {split}: {len(classes)} classes, {n_images} images")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    print(f"Dataset summary for: {data_dir}")
    dataset_summary(data_dir)
