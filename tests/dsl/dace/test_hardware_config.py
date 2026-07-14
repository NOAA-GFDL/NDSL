import pytest

from ndsl.dsl.dace.hardware_config import get_gpu_hardware_defaults
from ndsl.optional_imports import cupy


@pytest.mark.gpu
def test_gpu_detection() -> None:
    assert cupy is not None

    defaults = get_gpu_hardware_defaults()
    assert defaults.vendor != "Unknown"


def test_gpu_detection_no_crash_on_cpu() -> None:
    if cupy is not None:
        pytest.skip("This test only make sense when access to GPU is not available.")

    defaults = get_gpu_hardware_defaults()
    assert defaults.vendor == "Unknown"
