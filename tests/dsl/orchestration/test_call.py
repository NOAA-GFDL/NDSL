import pytest

from ndsl import QuantityFactory, StencilFactory
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.gt4py import PARALLEL, Field, computation, interval


def _stencil(out: Field[float]):
    with computation(PARALLEL), interval(...):
        out = out + 1


@pytest.fixture
def factories() -> tuple[StencilFactory, QuantityFactory]:
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 2, 0
    )
    return (stencil_factory, quantity_factory)


class OrchestratedProgram:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self.stencil = stencil_factory.from_dims_halo(_stencil, [X_DIM, Y_DIM, Z_DIM])

    def __call__(self, out_qty):
        self.stencil(out_qty)


def test_memory_reallocation(factories):
    qty_factory = factories[1]
    code = OrchestratedProgram(factories[0], qty_factory)
    qty_A = qty_factory.ones([X_DIM, Y_DIM, Z_DIM], "A")
    qty_B = qty_factory.ones([X_DIM, Y_DIM, Z_DIM], "B")

    code(qty_A)
    assert (qty_A.field[0, 0, :] == 2).all()

    code(qty_A)
    assert (qty_A.field[0, 0, :] == 3).all()

    code(qty_B)
    assert (qty_A.field[0, 0, :] == 3).all()
    assert (qty_B.field[0, 0, :] == 2).all()
