import numpy as np
import pytest

from ndsl import xumpy as xp
from ndsl.config import Backend


@pytest.mark.parametrize("dtype", [None, np.float32, np.float64])
def test_random(dtype) -> None:
    shape = (2, 3, 5)
    rand_array = xp.random(shape, Backend.python())
    assert rand_array.shape == shape
    assert (rand_array != xp.random(shape, Backend.python())).all()


def test_ones() -> None:
    shape = (2, 3, 5)
    assert (np.ones(shape) == xp.ones(shape, Backend.python())).all()


def test_zeros() -> None:
    shape = (2, 3, 5)
    assert (np.zeros(shape) == xp.zeros(shape, Backend.python())).all()


def test_full() -> None:
    shape = (2, 3, 5)
    value = 42.42
    assert (
        np.full(shape, value) == xp.full(shape, value=value, backend=Backend.python())
    ).all()
