"""Fine-tune a YOLO11 detector on the one-class drone dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained checkpoint")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA index such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="yolo11n_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs" / "drone"),
        name=args.name,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        close_mosaic=10,
        plots=True,
    )


if __name__ == "__main__":
    main()

