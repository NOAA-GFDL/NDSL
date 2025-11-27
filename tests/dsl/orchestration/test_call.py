import dataclasses

from ndsl import NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Float
from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.gt4py import PARALLEL, Field, computation, interval
from ndsl.quantity import Quantity, State


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


@dataclasses.dataclass
class AState(State):
    the_quantity: Quantity = dataclasses.field(
        metadata={
            "name": "A",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg kg-1",
            "intent": "?",
            "dtype": Float,
        }
    )


class DefaultTypeProgram(NDSLRuntime):
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        super().__init__(stencil_factory.config.dace_config)
        self.stencil = stencil_factory.from_dims_halo(_stencil, [X_DIM, Y_DIM, Z_DIM])

    def __call__(self, a_quantity: Quantity, a_state: AState):
        self.stencil(a_quantity)
        self.stencil(a_state.the_quantity)


def test_default_types_are_compiletime():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 2, 0
    )
    qty_A = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "A")
    state_A = AState.zeros(quantity_factory)
    code = DefaultTypeProgram(stencil_factory, quantity_factory)
    code(qty_A, state_A)
