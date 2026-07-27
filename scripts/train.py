#!/usr/bin/env python3
"""Fine-tune a YOLO detector for the single steel_ball class."""

from __future__ import annotations

import argparse
from pathlib import Path


def require_cuda(device: str) -> None:
    if device.lower() == "cpu":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build or pass --device cpu explicitly."
        )
    print(f"using CUDA GPU: {torch.cuda.get_device_name(0)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML")
    parser.add_argument("--model", default="yolo26n.pt", help="yolo26n.pt or yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="steel-ball")
    args = parser.parse_args()
    if args.epochs <= 0 or args.imgsz <= 0:
        parser.error("epochs and imgsz must be positive")
    if not args.data.is_file():
        parser.error(f"dataset YAML not found: {args.data}")

    try:
        require_cuda(args.device)
    except RuntimeError as error:
        parser.error(str(error))

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
