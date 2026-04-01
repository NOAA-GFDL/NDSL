from typing import Sequence, SupportsIndex

import numpy as np
import numpy.typing as npt
from numpy._typing import _SupportsDType

from ndsl.config import Backend
from ndsl.dsl.typing import Float
from ndsl.optional_imports import cupy as cp


if cp is None:
    cp = np

# Taking a page from cupy's playbook to have tuple & ndarray
_ShapeLike = SupportsIndex | Sequence[SupportsIndex]
_DTypeLikeFloat32 = (
    np.dtype[np.float32] | _SupportsDType[np.dtype[np.float32]] | type[np.float32]
)
_DTypeLikeFloat64 = (
    np.dtype[np.float64] | _SupportsDType[np.dtype[np.float64]] | type[np.float64]
)


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
    dtype: _DTypeLikeFloat32 | _DTypeLikeFloat64 = Float,  # type: ignore [valid-type]
) -> np.ndarray | cp.ndarray:
    if backend.is_gpu_backend():
        gen = cp.random.default_rng()
        return gen.random(shape, dtype, None)

    gen = np.random.default_rng()
    return gen.random(shape, dtype, None)
