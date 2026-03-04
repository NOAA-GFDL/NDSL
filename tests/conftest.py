import numpy as np
import pytest

from ndsl.config import Backend
from ndsl.optional_imports import cupy


@pytest.fixture(params=["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def backend(request):
    if request.param == "cupy" and cupy is None:
        raise ModuleNotFoundError("cupy must be installed to run gpu tests")

    return request.param


@pytest.fixture
def ndsl_backend(backend: str):
    if backend == "numpy":
        return Backend("st:numpy:cpu:IJK")

    if backend == "cupy":
        return Backend("st:dace:gpu:KJI")

    raise ValueError(f"Test backend {backend} cannot be translated into Backend")


@pytest.fixture
def numpy(backend: str):
    if backend == "numpy":
        return np

    if backend == "cupy":
        return cupy

    raise NotImplementedError(f"Unsupported backend {backend} found in test setup.")
