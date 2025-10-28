import pytest

from ndsl import Namelist


def test_ndsl_namelist_deprecation() -> None:
    with pytest.deprecated_call():
        my_namelist = Namelist()
