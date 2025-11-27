import numpy as np
import pytest

from ndsl import Local


def test_dace_data_descriptor_is_transient() -> None:
    nx = 5
    shape = (nx,)
    local = Local(
        data=np.empty(shape),
        origin=(0,),
        extent=(nx,),
        dims=("dim_X",),
        units="n/a",
        backend="debug",
    )
    array = local.__descriptor__()
    assert array.transient


def test_gt4py_backend_is_deprecated() -> None:
    nx = 5
    shape = (nx,)
    backend = "debug"
    with pytest.deprecated_call(match="gt4py_backend is deprecated"):
        local = Local(
            data=np.empty(shape),
            origin=(0,),
            extent=(nx,),
            dims=("dim_X",),
            units="n/a",
            gt4py_backend=backend,
        )

    # make sure we assign backend
    assert local.backend == backend

    # make sure we are backwards compatible (for now)
    with pytest.deprecated_call(match="gt4py_backend is deprecated"):
        assert local.gt4py_backend == backend


def test_backend_will_be_required() -> None:
    nx = 5
    shape = (nx,)
    with pytest.deprecated_call(match="`backend` will be a required argument"):
        local = Local(
            data=np.empty(shape),
            origin=(0,),
            extent=(nx,),
            dims=("dim_X",),
            units="n/a",
        )
