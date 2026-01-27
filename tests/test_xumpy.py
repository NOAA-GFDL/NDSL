import numpy as np

import ndsl.xumpy as xp
from ndsl.config import Backend


shape = (2, 2, 5)


def test_xumpy_alloc():
    rand_array = xp.random(shape, Backend.debug())
    assert rand_array.shape == shape
    (rand_array != xp.random(shape, Backend.debug())).all()

    assert (np.ones(shape) == xp.ones(shape, Backend.debug())).all()
    assert (np.zeros(shape) == xp.zeros(shape, Backend.debug())).all()
    assert (
        np.full(shape, 42.42) == xp.full(shape, value=42.42, backend=Backend.debug())
    ).all()


def test_xumpy_minmax():
    rand_array = xp.random(shape, Backend.debug())

    assert (np.max(rand_array, axis=1) == xp.max(rand_array, axis=1)).all()
    assert (np.min(rand_array, axis=1) == xp.min(rand_array, axis=1)).all()

    out_buffer = xp.empty(shape, Backend.debug())
    xp.max_on_horizontal_plane(rand_array, out_buffer)

    assert (np.max(rand_array, axis=(0, 1)) == out_buffer).all()


def test_xumpy_counts():
    rand_array = xp.random(shape, Backend.debug())
    rand_array[1, 1, :] = 0

    assert np.count_nonzero(rand_array) == xp.count_nonzero(rand_array)
