from deploy.k230.steel_ball_detect import decode_end2end_row


def test_end_to_end_row_returns_center_for_confident_detection() -> None:
    """A decoder change that swaps box coordinates must fail here."""
    assert decode_end2end_row([100, 200, 140, 260, 0.9, 0], 0.5) == {
        "confidence": 0.9,
        "center_x": 120.0,
        "center_y": 230.0,
    }
