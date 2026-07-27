#!/usr/bin/env python3
"""Run one image through ONNX Runtime and validate fixed-shape finite output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# PyTorch loads the CUDA 13 runtime shipped in its nvidia-cu13 dependency.
# It must be imported before onnxruntime-gpu on this CUDA-only host.
import torch  # noqa: F401
import onnxruntime as ort


def select_providers(available: list[str]) -> list[str]:
    """Prefer CUDA while retaining CPU as a portable fallback."""
    return [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]


def preprocess(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((640, 640))
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_file() or not args.image.is_file():
        parser.error("--model and --image must name existing files")

    providers = select_providers(ort.get_available_providers())
    if not providers:
        parser.error("ONNX Runtime exposes neither CUDA nor CPU execution provider")
    print(f"ONNX Runtime providers: {', '.join(providers)}")
    session = ort.InferenceSession(str(args.model), providers=providers)
    input_info = session.get_inputs()[0]
    if tuple(input_info.shape) != (1, 3, 640, 640):
        parser.error(f"expected ONNX input (1, 3, 640, 640), got {input_info.shape}")
    outputs = session.run(None, {input_info.name: preprocess(args.image)})
    if not outputs or any(not np.isfinite(output).all() for output in outputs):
        parser.error("ONNX output is empty or contains non-finite values")
    print("ONNX validation passed:", ", ".join(str(output.shape) for output in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
