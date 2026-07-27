import io

import pytest
from PIL import Image

from scripts.download_open_images import detect_image_type, validate_record


def test_validate_record_rejects_missing_source() -> None:
    """A downloader change that stops requiring provenance must fail here."""
    with pytest.raises(ValueError, match="source"):
        validate_record(
            {
                "url": "https://example.test/steel-ball.jpg",
                "license": "CC-BY-4.0",
            }
        )


def test_detect_image_type_identifies_png_bytes() -> None:
    """A broken image parser must not accept arbitrary bytes as a PNG."""
    output = io.BytesIO()
    Image.new("RGB", (1, 1)).save(output, format="PNG")
    assert detect_image_type(output.getvalue()) == "png"
