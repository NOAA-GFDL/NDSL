from functools import singledispatch

import numpy as np
import numpy.typing as npt

from ndsl import Quantity
from ndsl.optional_imports import cupy as cp


@singledispatch
def count_nonzero(
    in_buffer: npt.NDArray | Quantity,
    axis: int | tuple[int, ...] | None = None,
) -> np.integer:
    """Count non zero element in buffer."""
    raise NotImplementedError("`count_nonzero` called with not supported type")


@count_nonzero.register(np.ndarray)
def _(
    buffer: npt.NDArray,
    axis: int | tuple[int, ...] | None = None,
) -> np.integer:
    return np.count_nonzero(buffer, axis)


@count_nonzero.register(Quantity)
def _(
    in_quantity: Quantity,
    axis: int | tuple[int, ...] | None = None,
) -> np.integer:
    return count_nonzero(in_quantity.field, axis)


if cp is not None:

    @count_nonzero.register(cp.ndarray)
    def _(
        buffer: npt.NDArray,
        axis: int | tuple[int, ...] | None = None,
    ) -> np.integer:
        assert cp
        return cp.count_nonzero(buffer, axis)
