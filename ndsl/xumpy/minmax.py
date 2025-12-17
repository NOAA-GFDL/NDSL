import numpy as np
import numpy.typing as npt

from ndsl.dsl.typing import Float
from ndsl.optional_imports import cupy as cp


if cp is None:
    cp = np
from functools import singledispatch

from ndsl import Quantity


@singledispatch
def max_on_horizontal_plane(
    in_buffer: npt.NDArray | Quantity,
    out_buffer: npt.NDArray | Quantity,
) -> None:
    """Find maximum value on each horizontal plane, e.g. for the
    first two dimensions of the array."""
    raise NotImplementedError("")


@max_on_horizontal_plane.register(np.ndarray | cp.ndarray)
def _(in_buffer: npt.NDArray, out_buffer: npt.NDArray) -> None:
    print("numpy/cupy")
    out_buffer[:] = in_buffer.max(axis=(0, 1))


@max_on_horizontal_plane.register(Quantity)
def _(in_quantity: Quantity, out_quantity: Quantity) -> None:
    print("qty")

    out_quantity.field[:] = in_quantity.field.max(axis=(0, 1))


@singledispatch
def max(
    buffer: npt.NDArray | Quantity | int,
    axis: int | tuple[int, ...] | None = None,
) -> Float:
    """Find maximum value on a 1D field."""
    raise NotImplementedError(f"Missing implementation for {type(buffer)}")


@max.register(np.ndarray | cp.ndarray)
def _(
    buffer: npt.NDArray,
    axis: int | tuple[int, ...] | None = None,
) -> Float:
    return buffer.max(axis=axis)


@max.register(Quantity)
def _(
    quantity: Quantity,
    axis: int | tuple[int, ...] | None = None,
) -> Float:
    return quantity.field.max(axis=axis)


@singledispatch
def min(
    buffer: npt.NDArray | Quantity | int,
    axis: int | tuple[int, ...] | None = None,
) -> Float:
    """Find minimum value on a 1D field."""
    raise NotImplementedError("")


@min.register(np.ndarray | cp.ndarray)
def _(
    buffer: npt.NDArray,
    axis: int | tuple[int, ...] | None = None,
) -> Float:
    return buffer.min(axis=axis)


@min.register(Quantity)
def _(
    quantity: Quantity,
    axis: int | tuple[int, ...] | None = None,
) -> Float:
    return quantity.field.min(axis=axis)
