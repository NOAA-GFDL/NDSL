import numpy as np
import pytest

from ndsl import QuantityFactory, StencilFactory
from ndsl.config import Backend
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField


def _copy_ops(stencil_factory: StencilFactory, quantity_factory: QuantityFactory):
    # Allocate data and fill input
    qty_out = quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a")
    qty_in = quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a")
    qty_in.view[:] = np.indices(
        dimensions=quantity_factory.sizer.get_extent([X_DIM, Y_DIM, Z_DIM]),
        dtype=np.float64,
    ).sum(
        axis=0
    )  # Value of each entry is sum of the I and J index at each point

    # Define a stencil
    def copy_stencil(input_field: FloatField, output_field: FloatField):
        with computation(PARALLEL), interval(...):
            output_field = input_field

    # Execute
    copy = stencil_factory.from_dims_halo(
        func=copy_stencil, compute_dims=[X_DIM, Y_DIM, Z_DIM]
    )
    copy(qty_in, qty_out)
    assert (qty_in.view[:] == qty_out.view[:]).all()


def test_boilerplate_import_numpy():
    """Test make sure the basic numpy boilerplate works as expected.

    Dev Note: the import inside the function are part of the test.
    """
    from ndsl.boilerplate import get_factories_single_tile

    # Boilerplate
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=5, ny=5, nz=2, nhalo=1
    )

    # Ensure backend is propagated to StencilFactory and QuantityFactory
    assert stencil_factory.backend == Backend.python()
    assert quantity_factory.backend == Backend.python()

    _copy_ops(stencil_factory, quantity_factory)


def test_boilerplate_import_orchestrated_cpu():
    """Test make sure the basic orchestrate boilerplate works as expected.

    Dev Note: the import inside the function are part of the test.
    """
    from ndsl.boilerplate import get_factories_single_tile_orchestrated

    # Boilerplate
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        nx=5, ny=5, nz=2, nhalo=1
    )

    # Ensure backend is propagated to StencilFactory and QuantityFactory
    assert stencil_factory.backend == "dace:cpu"
    assert quantity_factory.backend == "dace:cpu"

    _copy_ops(stencil_factory, quantity_factory)


def test_boilerplate_non_dace_based_orchestration_raises():
    from ndsl.boilerplate import get_factories_single_tile_orchestrated

    with pytest.raises(ValueError, match="Only .* backends can be orchestrated."):
        get_factories_single_tile_orchestrated(
            nx=5, ny=5, nz=2, nhalo=1, backend=Backend.python()
        )
