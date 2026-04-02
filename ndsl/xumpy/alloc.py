from typing import Sequence, SupportsIndex

import numpy as np
import numpy.typing as npt

from ndsl.config import Backend
from ndsl.dsl.typing import Float
from ndsl.optional_imports import cupy as cp


if cp is None:
    cp = np

# Taking a page from cupy's playbook to have tuple & ndarray
_ShapeLike = SupportsIndex | Sequence[SupportsIndex]


def zeros(
    shape: _ShapeLike,
    backend: Backend,
    dtype: npt.DTypeLike = Float,
) -> np.ndarray | cp.ndarray:
    if backend.is_gpu_backend():
        return cp.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def ones(
    shape: _ShapeLike,
    backend: Backend,
    dtype: npt.DTypeLike = Float,
) -> np.ndarray | cp.ndarray:
    if backend.is_gpu_backend():
        return cp.ones(shape, dtype=dtype)
    return np.ones(shape, dtype=dtype)


def empty(
    shape: _ShapeLike,
    backend: Backend,
    dtype: npt.DTypeLike = Float,
) -> np.ndarray | cp.ndarray:
    if backend.is_gpu_backend():
        return cp.empty(shape, dtype=dtype)
    return np.empty(shape, dtype=dtype)


def full(
    shape: _ShapeLike,
    backend: Backend,
    value: np.ScalarType,
    dtype: npt.DTypeLike = Float,
) -> np.ndarray | cp.ndarray:
    if backend.is_gpu_backend():
        return cp.full(shape, value, dtype=dtype)
    return np.full(shape, value, dtype=dtype)


def random(
    shape: _ShapeLike,
    backend: Backend,
    dtype: npt.DTypeLike = Float,
) -> np.ndarray | cp.ndarray:
    if backend.is_gpu_backend():
        cp.random.rand(*shape)
    return np.random.rand(*shape)
