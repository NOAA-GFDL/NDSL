import pytest

from ndsl.units import UnitsError, ensure_equal_units, units_are_equal


def test_UnitsError_is_deprecated() -> None:
    with pytest.deprecated_call():
        UnitsError()


def test_units_are_equal_is_deprecated() -> None:
    with pytest.deprecated_call():
        units_are_equal("asdf", "asdf")


def test_ensure_equal_units_is_deprecated() -> None:
    with pytest.deprecated_call():
        ensure_equal_units("asdf", "asdf")
