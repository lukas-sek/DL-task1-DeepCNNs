import csv
import shutil
from pathlib import Path

SRC_IMAGES = Path("/tmp/mame_raw/data")
CSV_FILE   = Path("/tmp/mame_raw/MAMe_dataset.csv")
DST_ROOT   = Path.home() / "DL-Lab-CNN" / "data"

if not CSV_FILE.exists():
    # Try alternative location
    CSV_FILE = Path("/tmp/mame_raw/MAMe_dataset.csv")

print(f"Source images : {SRC_IMAGES}")
print(f"CSV file      : {CSV_FILE}")
print(f"Destination   : {DST_ROOT}")

if not SRC_IMAGES.exists():
    raise FileNotFoundError(f"Source images not found: {SRC_IMAGES}")
if not CSV_FILE.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

moved   = 0
skipped = 0

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"\nTotal rows in CSV: {len(rows)}")
print("Reorganizing ...")

for row in rows:
    fname  = row["Image file"].strip()
    medium = row["Medium"].strip()
    subset = row["Subset"].strip()

    src = SRC_IMAGES / fname
    if not src.exists():
        skipped += 1
        continue

    dst_dir = DST_ROOT / subset / medium
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), dst_dir / fname)
    moved += 1
    if moved % 5000 == 0:
        print(f"  {moved} / {len(rows)} ...")

print(f"\nDone. Moved: {moved}  |  Skipped (not found): {skipped}")
print("\nVerifying ...")
for split in ("train", "val", "test"):
    split_dir = DST_ROOT / split
    if split_dir.exists():
        n = sum(1 for _ in split_dir.rglob("*.jpg"))
        print(f"  {split}: {n} images")
    else:
        print(f"  {split}: NOT FOUND")
