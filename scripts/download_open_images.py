#!/usr/bin/env python3
"""Download images from a reviewed, traceable JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


MAX_BYTES = 20 * 1024 * 1024
REQUIRED_FIELDS = ("url", "source", "license")


def validate_record(record: dict[str, str]) -> None:
    """Reject a manifest entry that cannot be traced or downloaded safely."""
    for field in REQUIRED_FIELDS:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"record requires a non-empty {field}")

    scheme = urlparse(record["url"]).scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("url must use http or https")


def load_manifest(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} must be an object")
        validate_record(record)
        records.append(record)
    return records


def detect_image_type(payload: bytes) -> str | None:
    """Return a normalized image format only after Pillow verifies the bytes."""
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image_type = image.format
            image.verify()
    except (UnidentifiedImageError, OSError):
        return None
    return image_type.lower() if image_type else None


def download_record(record: dict[str, str], images_dir: Path) -> dict[str, str] | None:
    """Download one record and return its persisted metadata, or None if skipped."""
    request = Request(record["url"], headers={"User-Agent": "steel-ball-yolo/0.1"})
    try:
        with urlopen(request, timeout=20) as response:
            content_type = response.headers.get_content_type()
            if not content_type.startswith("image/"):
                print(f"skip non-image: {record['url']} ({content_type})", file=sys.stderr)
                return None
            payload = response.read(MAX_BYTES + 1)
    except (HTTPError, URLError, TimeoutError) as error:
        print(f"skip download error: {record['url']} ({error})", file=sys.stderr)
        return None

    if len(payload) > MAX_BYTES:
        print(f"skip oversized image: {record['url']}", file=sys.stderr)
        return None
    image_type = detect_image_type(payload)
    if image_type is None:
        print(f"skip invalid image bytes: {record['url']}", file=sys.stderr)
        return None

    digest = hashlib.sha256(payload).hexdigest()
    filename = f"{digest}.{image_type}"
    destination = images_dir / filename
    if destination.exists():
        print(f"skip duplicate: {record['url']}", file=sys.stderr)
        return None
    destination.write_bytes(payload)
    return {
        **record,
        "sha256": digest,
        "filename": filename,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL source manifest")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="directory for images and the persisted manifest",
    )
    args = parser.parse_args()

    try:
        records = load_manifest(args.manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_dir / "manifest.jsonl"
    downloaded = 0
    with output_manifest.open("a", encoding="utf-8") as output:
        for record in records:
            result = download_record(record, images_dir)
            if result is not None:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                downloaded += 1
    print(f"downloaded {downloaded} image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
