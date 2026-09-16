"""Run a trained detector on a video, image directory, or webcam."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Video/image path, directory, or webcam index")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    results = model.predict(
        source=parse_source(args.source),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        save=not args.no_save,
        project=str(ROOT / "runs" / "predictions"),
        name=weights.stem,
        stream=True,
    )

    # Consume the streaming generator so inference actually runs.
    for _ in results:
        pass


if __name__ == "__main__":
    main()
