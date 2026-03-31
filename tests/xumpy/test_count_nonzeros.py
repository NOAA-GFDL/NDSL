import numpy as np

from ndsl import xumpy as xp
from ndsl.config import Backend


def test_count_nonzero():
    shape = (2, 3, 5)
    rand_array = xp.random(shape, Backend.python())
    rand_array[1, 1, :] = 0

    assert np.count_nonzero(rand_array) == xp.count_nonzero(rand_array)
