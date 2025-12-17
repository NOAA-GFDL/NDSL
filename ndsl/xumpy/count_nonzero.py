import numpy as np
import numpy.typing as npt

from ndsl.optional_imports import cupy as cp


if cp is None:
    cp = np
from functools import singledispatch

from ndsl import Quantity


@singledispatch
def count_nonzero(
    in_buffer: npt.NDArray | Quantity,
    axis: int | tuple[int, ...] | None = None,
) -> int:
    """Count non zero element in buffer."""
    raise NotImplementedError("")


@count_nonzero.register(np.ndarray)
def _(
    buffer: npt.NDArray,
    axis: int | tuple[int, ...] | None = None,
) -> int:
    return np.count_nonzero(buffer, axis)


@count_nonzero.register(cp.ndarray)
def _(
    buffer: npt.NDArray,
    axis: int | tuple[int, ...] | None = None,
) -> int:
    return cp.count_nonzero(buffer, axis)


@count_nonzero.register(Quantity)
def _(
    in_quantity: Quantity,
    axis: int | tuple[int, ...] | None = None,
) -> int:
    return count_nonzero(in_quantity.field, axis)
