import dataclasses

import pytest

from ndsl import NDSLRuntime, Quantity, State, StencilFactory
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.config import Backend
from ndsl.constants import I_DIM, J_DIM, K_DIM, Float
from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.gt4py import PARALLEL, Field, computation, interval
from ndsl.dsl.typing import FloatField


def _stencil(out: Field[float]):
    with computation(PARALLEL), interval(...):
        out = out + 1


@dataclasses.dataclass
class AState(State):
    the_quantity: Quantity = dataclasses.field(
        metadata={
            "name": "A",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg kg-1",
            "intent": "?",
            "dtype": Float,
        }
    )


class OrchestratedProgram:
    def __init__(self, stencil_factory: StencilFactory):
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self.stencil = stencil_factory.from_dims_halo(_stencil, [I_DIM, J_DIM, K_DIM])

    def __call__(self, out_qty):
        self.stencil(out_qty)


class DSLTypeProgram(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
        self.stencil = stencil_factory.from_dims_halo(_stencil, [I_DIM, J_DIM, K_DIM])

    def __call__(self, a_quantity: Quantity, a_state: AState):
        self.stencil(a_quantity)
        self.stencil(a_state.the_quantity)


class GTTypeProgram(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
        self.stencil = stencil_factory.from_dims_halo(_stencil, [I_DIM, J_DIM, K_DIM])

    def __call__(self, a_quantity: FloatField):
        self.stencil(a_quantity)


def test_memory_reallocation_blind_type():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 2, 0
    )
    code = OrchestratedProgram(stencil_factory)
    qty_A = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "A")
    qty_B = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "B")

    code(qty_A)
    assert (qty_A.field[0, 0, :] == 2).all()

    code(qty_A)
    assert (qty_A.field[0, 0, :] == 3).all()

    code(qty_B)
    assert (qty_A.field[0, 0, :] == 3).all()
    assert (qty_B.field[0, 0, :] == 2).all()


@pytest.mark.xfail(reason="See https://github.com/NOAA-GFDL/NDSL/issues/436")
def test_memory_reallocation_dsl_typehint():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 2, 0
    )
    typed_code = DSLTypeProgram(stencil_factory)
    qty_C = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "A")
    state_A = AState.ones(quantity_factory)
    state_B = AState.ones(quantity_factory)

    typed_code(qty_C, state_A)
    assert (qty_C.field[0, 0, :] == 2).all()
    assert (state_A.the_quantity.field[0, 0, :] == 2).all()

    typed_code(qty_C, state_B)
    assert (state_A.the_quantity.field[0, 0, :] == 2).all()
    assert (state_B.the_quantity.field[0, 0, :] == 2).all()


def test_memory_reallocation_gt4py_typehint():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 2, 0
    )
    typed_code = GTTypeProgram(stencil_factory)
    qty_D = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "A")
    qty_E = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "A")

    typed_code(qty_D)
    assert (qty_D.field[0, 0, :] == 2).all()

    typed_code(qty_E)
    assert (qty_D.field[0, 0, :] == 2).all()
    assert (qty_E.field[0, 0, :] == 2).all()


def test_default_types_are_compiletime():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 2, 0
    )
    qty_A = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "A")
    state_A = AState.zeros(quantity_factory)
    code = DSLTypeProgram(stencil_factory)
    code(qty_A, state_A)


def test_dace_call_argument_caching():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 2, 0, backend=Backend.cpu()
    )
    dconfig = stencil_factory.config.dace_config

    quantity_A = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "A")
    state_A = AState.zeros(quantity_factory)
    code = DSLTypeProgram(stencil_factory)
    code(quantity_A, state_A)

    assert len(dconfig.loaded_dace_executables.values()) == 1

    hash_A = list(dconfig.loaded_dace_executables.values())[0].arguments_hash

    code(quantity_A, state_A)

    # Same call - no hash recompute
    assert list(dconfig.loaded_dace_executables.values())[0].arguments_hash == hash_A

    qty_B = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "B")
    code(qty_B, state_A)

    # New call - hash recompute
    assert list(dconfig.loaded_dace_executables.values())[0].arguments_hash != hash_A
    hash_B = list(dconfig.loaded_dace_executables.values())[0].arguments_hash

    # Back to original call - recompute to first hash
    code(quantity_A, state_A)
    assert list(dconfig.loaded_dace_executables.values())[0].arguments_hash != hash_B
    assert list(dconfig.loaded_dace_executables.values())[0].arguments_hash == hash_A

    # Check that inner quantity data swap recomputes
    quantity_A.swap_buffer(quantity_factory.ones([I_DIM, J_DIM, K_DIM], "Abis")._data)
    code(quantity_A, state_A)
    assert list(dconfig.loaded_dace_executables.values())[0].arguments_hash != hash_A
    hash_Abis = list(dconfig.loaded_dace_executables.values())[0].arguments_hash

    # Check that state quantity swap recomputes
    state_A.the_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "InnerA")
    code(quantity_A, state_A)
    assert list(dconfig.loaded_dace_executables.values())[0].arguments_hash != hash_Abis
