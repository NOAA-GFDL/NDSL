import pytest

from ndsl import NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField


def the_copy_stencil(from_: FloatField, to: FloatField):
    with computation(PARALLEL), interval(...):
        to = from_


class Code(NDSLRuntime):
    def __init__(
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ) -> None:
        super().__init__(dace_config=stencil_factory.config.dace_config)
        self.copy = stencil_factory.from_dims_halo(
            the_copy_stencil, compute_dims=[X_DIM, Y_DIM, Z_DIM]
        )
        self.local = self.make_local(quantity_factory, [X_DIM, Y_DIM, Z_DIM])

    def test_check(self):
        assert self.local.__descriptor__().transient

    def __call__(self, A, B):
        self.copy(A, self.local)
        self.copy(self.local, B)


def test_local_and_transient_flags():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 3, 0, backend="dace:cpu_kfirst"
    )

    A_ = quantity_factory.ones(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a")
    B_ = quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a")

    code = Code(stencil_factory, quantity_factory)
    code(A_, B_)

    # Check that local is not reachable outside of Code
    with pytest.raises(RuntimeError, match="Forbidden Local access:"):
        assert code.local._transient

    # Check the local is properly transient - with access in Code
    code.test_check()

    # Check regular quantity are not transient
    assert not A_.__descriptor__().transient
    assert not B_.__descriptor__().transient

    # Check numerics
    assert (A_.field[:] == B_.field[:]).all()
