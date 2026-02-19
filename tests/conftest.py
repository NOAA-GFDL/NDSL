import numpy as np
import pytest

from ndsl.optional_imports import cupy


@pytest.fixture(params=["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def backend(request):
    if request.param == "cupy" and cupy is None:
        raise ModuleNotFoundError("cupy must be installed to run gpu tests")

    return request.param


@pytest.fixture
def gt4py_backend(backend):
    if backend == "numpy":
        return "numpy"

    if backend == "cupy":
        return "gt:gpu"

    return None


@pytest.fixture
def numpy(backend):
    if backend == "numpy":
        return np

    if backend == "cupy":
        return cupy

    raise NotImplementedError(f"Unsupported backend {backend} found in test setup.")
