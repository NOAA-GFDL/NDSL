import dataclasses

from ndsl import (
    Quantity,
    QuantityFactory,
    State,
    StencilFactory,
    orchestrate,
)
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Float
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField


@dataclasses.dataclass
class CodeState(State):
    @dataclasses.dataclass
    class Inner:
        A: Quantity = dataclasses.field(
            metadata={
                "name": "A",
                "dims": [X_DIM, Y_DIM, Z_DIM],
                "units": "kg kg-1",
                "intent": "?",
                "dtype": Float,
            }
        )

    inner: Inner
    C: Quantity = dataclasses.field(
        metadata={
            "name": "C",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg kg-1",
            "intent": "?",
            "dtype": Float,
        }
    )


def the_copy_stencil(from_: FloatField, to: FloatField):
    with computation(PARALLEL), interval(...):
        to = from_


class Code:
    def __init__(
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ) -> None:
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["state", "tmps"],
        )

        self.copy = stencil_factory.from_dims_halo(
            the_copy_stencil, compute_dims=[X_DIM, Y_DIM, Z_DIM]
        )
        self.local = quantity_factory.empty(
            [X_DIM, Y_DIM, Z_DIM], units="n/a", is_local=True
        )

    def __call__(self, state: CodeState):
        self.copy(state.inner.A, self.local)
        self.copy(self.local, state.C)


def test_locals():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 3, 0, backend="dace:cpu_kfirst"
    )

    state = CodeState.full(quantity_factory, 42.42)

    c = Code(stencil_factory, quantity_factory)
    c(state)

    assert c.local._transient
    assert not state.inner.A._transient
    assert not state.C._transient
    assert (state.inner.A.data[:] == state.C.data[:]).all()
