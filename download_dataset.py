import argparse
import os
import zipfile
from pathlib import Path


DATASET_SLUG = "ferranpares/mame-dataset"
DEFAULT_DEST = Path(__file__).parent / "data"


def download(dest: Path):
    # Import here so the error message is clear if kaggle is not installed
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        raise SystemExit("kaggle package not found. Run: pip install kaggle")

    api = KaggleApiExtended()
    api.authenticate()

    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{DATASET_SLUG}' to {dest} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(dest), unzip=False, quiet=False)

    # Find the downloaded zip (kaggle names it after the dataset slug)
    zips = list(dest.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("Download finished but no zip file found in destination.")

    for zip_path in zips:
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink()
        print(f"  Removed {zip_path.name}")

    print(f"\nDone. Dataset extracted to: {dest}")
    print("Expected structure:")
    print("  data/")
    print("    train/  (20,300 images)")
    print("    val/    (1,450 images)")
    print("    test/   (15,657 images)")

    # Quick sanity check
    for split in ("train", "val", "test"):
        split_dir = dest / split
        if split_dir.exists():
            n = sum(1 for _ in split_dir.rglob("*.jpg")) + \
                sum(1 for _ in split_dir.rglob("*.png")) + \
                sum(1 for _ in split_dir.rglob("*.jpeg"))
            print(f"  {split}: {n} images found")
        else:
            print(f"  WARNING: {split}/ directory not found — check extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the MAMe 256x256 dataset from Kaggle.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()
    download(args.dest)
