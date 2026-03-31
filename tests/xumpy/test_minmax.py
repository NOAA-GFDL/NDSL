import numpy as np

from ndsl import xumpy as xp
from ndsl.config import Backend


def test_minmax():
    shape = (2, 3, 5)
    rand_array = xp.random(shape, Backend.python())

    assert (np.max(rand_array, axis=1) == xp.max(rand_array, axis=1)).all()
    assert (np.min(rand_array, axis=1) == xp.min(rand_array, axis=1)).all()

    out_buffer = xp.empty(shape, Backend.python())
    xp.max_on_horizontal_plane(rand_array, out_buffer)

    assert (np.max(rand_array, axis=(0, 1)) == out_buffer).all()
