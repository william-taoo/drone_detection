"""Validate image/label pairing and YOLO bounding-box rows before training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Counts:
    images: int = 0
    boxes: int = 0
    negatives: int = 0
    warnings: int = 0
    errors: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset.yaml")
    return parser.parse_args()


def label_path_for(image: Path, image_root: Path, label_root: Path) -> Path:
    return label_root / image.relative_to(image_root).with_suffix(".txt")


def validate_label(label: Path, class_ids: set[int], counts: Counts) -> None:
    rows = label.read_text(encoding="utf-8").splitlines()
    if not rows:
        counts.negatives += 1
        return

    for line_number, raw in enumerate(rows, start=1):
        parts = raw.split()
        if len(parts) != 5:
            print(f"ERROR {label}:{line_number}: expected 5 fields, got {len(parts)}")
            counts.errors += 1
            continue
        try:
            class_value = float(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            print(f"ERROR {label}:{line_number}: contains a non-numeric value")
            counts.errors += 1
            continue

        class_id = int(class_value)
        if class_value != class_id or class_id not in class_ids:
            print(f"ERROR {label}:{line_number}: invalid class id {parts[0]}")
            counts.errors += 1
        if any(value < 0.0 or value > 1.0 for value in coords):
            print(f"ERROR {label}:{line_number}: coordinates must be within [0, 1]")
            counts.errors += 1
        if coords[2] <= 0.0 or coords[3] <= 0.0:
            print(f"ERROR {label}:{line_number}: width and height must be positive")
            counts.errors += 1
        counts.boxes += 1


def main() -> None:
    args = parse_args()
    config_path = args.data.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = (config_path.parent / config["path"]).resolve()
    class_ids = {int(key) for key in config["names"]}
    total = Counts()

    for split in ("train", "val", "test"):
        image_root = dataset_root / config[split]
        label_root = dataset_root / "labels" / split
        split_counts = Counts()
        if not image_root.is_dir():
            print(f"ERROR missing {split} image directory: {image_root}")
            total.errors += 1
            continue

        images = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        for image in images:
            split_counts.images += 1
            label = label_path_for(image, image_root, label_root)
            if label.is_file():
                validate_label(label, class_ids, split_counts)
            else:
                split_counts.negatives += 1

        print(
            f"{split:>5}: {split_counts.images} images, {split_counts.boxes} boxes, "
            f"{split_counts.negatives} negative/unlabeled images, {split_counts.errors} errors"
        )
        total.images += split_counts.images
        total.boxes += split_counts.boxes
        total.negatives += split_counts.negatives
        total.errors += split_counts.errors

    print(
        f"TOTAL: {total.images} images, {total.boxes} boxes, "
        f"{total.negatives} negative/unlabeled images, {total.errors} errors"
    )
    if total.images == 0:
        raise SystemExit("Dataset contains no images yet.")
    if total.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

