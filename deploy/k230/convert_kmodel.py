#!/usr/bin/env python3
"""Quantize a fixed-shape ONNX detector into a K230 kmodel with nncase."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((640, 640))
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def calibration_samples(directory: Path, limit: int) -> list[np.ndarray]:
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)[:limit]
    if not paths:
        raise ValueError(f"no calibration images found in {directory}")
    return [preprocess(path) for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, default=Path("artifacts/nncase-dump"))
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if not args.onnx.is_file() or not args.calibration_dir.is_dir() or args.samples <= 0:
        parser.error("--onnx must be a file, --calibration-dir a directory, and --samples positive")

    try:
        import nncase
    except ModuleNotFoundError:
        parser.error("nncase is missing; install it with: uv pip install nncase nncase-kpu onnxsim scikit-learn")

    samples = calibration_samples(args.calibration_dir, args.samples)
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import_options = nncase.ImportOptions()
    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"
    compile_options.dump_ir = False
    compile_options.dump_asm = True
    compile_options.dump_dir = str(args.dump_dir)
    compile_options.preprocess = False

    ptq_options = nncase.PTQTensorOptions()
    ptq_options.quant_type = "uint8"
    ptq_options.w_quant_type = "uint8"
    ptq_options.calibrate_method = "NoClip"
    ptq_options.finetune_weights_method = "NoFineTuneWeights"
    ptq_options.samples_count = len(samples)
    ptq_options.set_tensor_data([samples])

    compiler = nncase.Compiler(compile_options)
    compiler.import_onnx(args.onnx.read_bytes(), import_options)
    compiler.use_ptq(ptq_options)
    compiler.compile()
    args.output.write_bytes(compiler.gencode_tobytes())
    print(f"generated K230 kmodel: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
