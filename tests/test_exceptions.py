import pytest

from ndsl import OutOfBoundsError


def test_OutOfBoundsError_is_deprecation() -> None:
    with pytest.deprecated_call():
        OutOfBoundsError("This should trigger a deprecation warning.")
