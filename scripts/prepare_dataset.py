#!/usr/bin/env python3
"""Validate YOLO labels and build a reproducible train/validation split."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_label_line(line: str, line_number: int) -> tuple[int, float, float, float, float]:
    """Parse one normalized YOLO box, rejecting class and bounds violations."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"line {line_number}: expected 5 values")
    try:
        class_id = int(parts[0])
        center_x, center_y, width, height = (float(value) for value in parts[1:])
    except ValueError as error:
        raise ValueError(f"line {line_number}: values must be numeric") from error

    if class_id != 0:
        raise ValueError(f"line {line_number}: expected class 0, got {class_id}")
    if not all(0 < value <= 1 for value in (center_x, center_y, width, height)):
        raise ValueError(f"line {line_number}: normalized coordinates must be in (0, 1]")
    if center_x - width / 2 < 0 or center_x + width / 2 > 1:
        raise ValueError(f"line {line_number}: box exceeds image width")
    if center_y - height / 2 < 0 or center_y + height / 2 > 1:
        raise ValueError(f"line {line_number}: box exceeds image height")
    return class_id, center_x, center_y, width, height


def validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"damaged image: {path}") from error


def validate_labels(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"missing label: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            parse_label_line(line, line_number)


def collect_images(images_dir: Path, labels_dir: Path) -> list[Path]:
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"no supported images found in {images_dir}")
    for image_path in images:
        validate_image(image_path)
        validate_labels(labels_dir / f"{image_path.stem}.txt")
    return images


def split_images(images: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def copy_split(images: list[Path], labels_dir: Path, output_dir: Path, split: str) -> None:
    image_target = output_dir / "images" / split
    label_target = output_dir / "labels" / split
    image_target.mkdir(parents=True, exist_ok=True)
    label_target.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        shutil.copy2(image_path, image_target / image_path.name)
        shutil.copy2(labels_dir / f"{image_path.stem}.txt", label_target / f"{image_path.stem}.txt")


def write_dataset_yaml(output_dir: Path) -> Path:
    destination = output_dir / "dataset.yaml"
    destination.write_text(
        yaml.safe_dump(
            {
                "path": str(output_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "steel_ball"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    try:
        images = collect_images(args.images_dir, args.labels_dir)
        train_images, val_images = split_images(images, args.val_ratio, args.seed)
        copy_split(train_images, args.labels_dir, args.output_dir, "train")
        copy_split(val_images, args.labels_dir, args.output_dir, "val")
        yaml_path = write_dataset_yaml(args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"prepared {len(train_images)} train and {len(val_images)} val images: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
