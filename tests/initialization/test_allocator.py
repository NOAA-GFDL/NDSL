import pytest

from ndsl import QuantityFactory


def test_QuantityFactory_from_backend_warns() -> None:
    with pytest.deprecated_call():
        QuantityFactory.from_backend(None, backend="numpy")
