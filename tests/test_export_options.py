from scripts.export_onnx import build_export_options
from scripts.validate_onnx import select_providers


def test_default_export_uses_fixed_shape_end_to_end_options() -> None:
    """An export change that enables dynamic shapes must fail here."""
    assert build_export_options(False) == {
        "format": "onnx",
        "imgsz": 640,
        "batch": 1,
        "dynamic": False,
    }


def test_select_providers_prefers_cuda_before_cpu() -> None:
    """A provider-order regression must not silently move validation back to CPU."""
    assert select_providers(["CPUExecutionProvider", "CUDAExecutionProvider"]) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
