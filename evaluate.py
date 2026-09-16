"""Evaluate trained weights on the held-out YOLO test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    data_path = args.data.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(ROOT / "runs" / "evaluation"),
        name=weights.stem,
        plots=True,
    )

    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()

