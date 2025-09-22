import warnings

import numpy as np
import pytest

from ndsl import QuantityFactory


def test_QuantityFactory_constructor_warns() -> None:
    with pytest.warns(
        DeprecationWarning,
        match="Usage of QuantityFactory.* is discouraged and will change",
    ):
        QuantityFactory(None, np)

    # Make sure no warnings are emitted if users use `QuantityFactory.from_backend()`
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        QuantityFactory.from_backend(None, "numpy")
