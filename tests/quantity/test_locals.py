from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField


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
            dace_compiletime_args=["A", "B"],
        )

        self.copy = stencil_factory.from_dims_halo(
            the_copy_stencil, compute_dims=[X_DIM, Y_DIM, Z_DIM]
        )
        self.local = quantity_factory.empty(
            [X_DIM, Y_DIM, Z_DIM], units="n/a", is_local=True
        )

    def __call__(self, A, B):
        self.copy(A, self.local)
        self.copy(self.local, B)


def test_locals():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 3, 0, backend="dace:cpu_kfirst"
    )

    A_ = quantity_factory.ones(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a")
    B_ = quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a")

    c = Code(stencil_factory, quantity_factory)
    c(A_, B_)

    assert c.local._transient
    assert not A_._transient
    assert not B_._transient
    assert (A_.field[:] == B_.field[:]).all()
