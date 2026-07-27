#!/usr/bin/env python3
"""Export a trained YOLO26 or YOLO11 detector to fixed-shape ONNX."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_export_options(fallback_yolo11: bool) -> dict[str, object]:
    options: dict[str, object] = {
        "format": "onnx",
        "imgsz": 640,
        "batch": 1,
        "dynamic": False,
    }
    if fallback_yolo11:
        options["end2end"] = False
    return options


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fallback-yolo11",
        action="store_true",
        help="export a separately trained YOLO11 model with traditional NMS output",
    )
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"weights not found: {args.weights}")
    if args.output.suffix.lower() != ".onnx":
        parser.error("output must end in .onnx")

    from ultralytics import YOLO

    exported_path = Path(YOLO(str(args.weights)).export(**build_export_options(args.fallback_yolo11)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != args.output.resolve():
        shutil.copy2(exported_path, args.output)
    print(f"exported ONNX: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
