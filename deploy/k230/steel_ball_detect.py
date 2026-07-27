"""Pure-Python reference decoder for YOLO26 End-to-End output rows."""

from __future__ import annotations

from collections.abc import Sequence


def decode_end2end_row(row: Sequence[float], threshold: float) -> dict[str, float] | None:
    """Decode one `[x1, y1, x2, y2, confidence, class_id]` YOLO26 row."""
    if len(row) != 6:
        raise ValueError("expected six output values")
    left, top, right, bottom, confidence, class_id = (float(value) for value in row)
    if class_id != 0 or confidence < threshold or right <= left or bottom <= top:
        return None
    return {
        "confidence": confidence,
        "center_x": (left + right) / 2,
        "center_y": (top + bottom) / 2,
    }
