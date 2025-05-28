import numpy as np
import pytest


try:
    import ndsl.dsl  # noqa: F401
except ModuleNotFoundError:
    raise ModuleNotFoundError("NDSL cannot be loaded")
try:
    import cupy
except ModuleNotFoundError:
    cupy = None


@pytest.fixture(params=["numpy", "cupy"])
def backend(request):
    if cupy is None and request.param.endswith("cupy"):
        if request.config.getoption("--gpu-only"):
            raise ModuleNotFoundError("cupy must be installed to run gpu tests")
        else:
            pytest.skip("cupy is not available for GPU backend")
    elif request.config.getoption("--gpu-only") and not request.param.endswith("cupy"):
        pytest.skip("running gpu tests only")
    else:
        return request.param


@pytest.fixture
def gt4py_backend(backend):
    if backend in ("numpy"):
        return "numpy"
    elif backend in ("cupy"):
        return "gt:gpu"
    else:
        return None


@pytest.fixture
def fast(pytestconfig):
    return pytestconfig.getoption("--fast")


@pytest.fixture
def numpy(backend):
    if backend == "numpy":
        return np
    elif backend == "cupy":
        return cupy
    else:
        raise NotImplementedError()


def pytest_addoption(parser):
    parser.addoption(
        "--gpu-only", action="store_true", default=False, help="only run gpu tests"
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="run a limited suite of tests which completes quickly",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cpu_only: mark test as not using a gpu")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu-only"):
        skip_cpu_only = pytest.mark.skip(reason="running gpu tests only")
        for item in items:
            if "cpu_only" in item.keywords:
                item.add_marker(skip_cpu_only)
