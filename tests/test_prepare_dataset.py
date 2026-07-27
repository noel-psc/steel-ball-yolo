import pytest

from scripts.prepare_dataset import parse_label_line


def test_parse_label_line_rejects_box_outside_image() -> None:
    """A parser change that accepts out-of-frame boxes must fail here."""
    with pytest.raises(ValueError, match="line 1"):
        parse_label_line("0 1.1 0.5 0.2 0.2", 1)
