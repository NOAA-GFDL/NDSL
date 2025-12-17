import numpy as np
import numpy.typing as npt

from ndsl.dsl.gt4py_utils import is_gpu_backend
from ndsl.dsl.typing import Float
from ndsl.optional_imports import cupy as cp


if cp is None:
    cp = np


def zeros(
    shape: tuple[int, ...],
    backend: str,
    dtype: npt.DTypeLike = Float,
) -> np.ndarray | cp.ndarray:
    if is_gpu_backend(backend):
        return cp.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def ones(
    shape: tuple[int, ...],
    backend: str,
    dtype: npt.DTypeLike = Float,
) -> np.ndarray | cp.ndarray:
    if is_gpu_backend(backend):
        return cp.ones(shape, dtype=dtype)
    return np.ones(shape, dtype=dtype)


def empty(
    shape: tuple[int, ...],
    backend: str,
    dtype: npt.DTypeLike = Float,
) -> np.ndarray | cp.ndarray:
    if is_gpu_backend(backend):
        return cp.empty(shape, dtype=dtype)
    return np.empty(shape, dtype=dtype)


def full(
    shape: tuple[int, ...],
    backend: str,
    value: np.generic,
    dtype: npt.DTypeLike = Float,
) -> np.ndarray | cp.ndarray:
    if is_gpu_backend(backend):
        return cp.full(shape, value, dtype=dtype)
    return np.full(shape, value, dtype=dtype)


def random(
    shape: tuple[int, ...],
    backend: str,
    dtype: npt.DTypeLike = Float,
) -> np.ndarray | cp.ndarray:
    if is_gpu_backend(backend):
        cp.random.rand(*shape)
    return np.random.rand(*shape)
