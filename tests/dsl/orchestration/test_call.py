from ndsl import QuantityFactory, StencilFactory
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.gt4py import PARALLEL, Field, computation, interval


def _stencil(out: Field[float]):
    with computation(PARALLEL), interval(...):
        out = out + 1


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


def test_memory_reallocation():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 2, 0
    )
    code = OrchestratedProgram(stencil_factory, quantity_factory)
    qty_A = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "A")
    qty_B = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "B")

    code(qty_A)
    assert (qty_A.field[0, 0, :] == 2).all()

    code(qty_A)
    assert (qty_A.field[0, 0, :] == 3).all()

    code(qty_B)
    assert (qty_A.field[0, 0, :] == 3).all()
    assert (qty_B.field[0, 0, :] == 2).all()
