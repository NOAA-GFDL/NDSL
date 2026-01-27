from typing import Any

import pytest

from ndsl import NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.boilerplate import (
    get_factories_single_tile,
    get_factories_single_tile_orchestrated,
)
from ndsl.config import Backend
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField


def the_copy_stencil(from_: FloatField, to: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        to = from_


class Code(NDSLRuntime):
    def __init__(
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ) -> None:
        super().__init__(stencil_factory)
        self.copy = stencil_factory.from_dims_halo(
            the_copy_stencil, compute_dims=[I_DIM, J_DIM, K_DIM]
        )
        self.local = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])

    def test_check(self) -> None:
        assert self.local.__descriptor__().transient

    def __call__(self, A, B) -> None:  # type: ignore[no-untyped-def]
        self.copy(A, self.local)
        self.copy(self.local, B)


class BadCode_NoSuperInit(NDSLRuntime):
    def __init__(self) -> None:
        # Forget to init
        pass


class Code_NoCall(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory) -> None:
        super().__init__(stencil_factory)
        pass

    def run(self, A: Any, B: Any) -> None:
        pass


def test_runtime_make_local() -> None:
    stencil_factory, quantity_factory = get_factories_single_tile(5, 5, 3, 0)
    A_ = quantity_factory.ones(dims=[I_DIM, J_DIM, K_DIM], units="n/a")
    B_ = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="n/a")

    code = Code(stencil_factory, quantity_factory)

    # Check that local is not reachable outside of Code
    with pytest.raises(RuntimeError, match="Forbidden Local access:"):
        assert code.local.__descriptor__().transient

    # Check the local is properly transient - with access in Code
    code.test_check()

    # Check regular quantity are not transient
    assert not A_.__descriptor__().transient
    assert not B_.__descriptor__().transient


def test_runtime_has_orchestracted_call() -> None:
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 3, 0, backend=Backend.performance_cpu()
    )
    A_ = quantity_factory.ones(dims=[I_DIM, J_DIM, K_DIM], units="n/a")
    B_ = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="n/a")
    code = Code(stencil_factory, quantity_factory)
    code(A_, B_)

    # We monkey patch the class, a __name__ attribute is now available
    # and the original Class name is postfixed with "_patched"
    assert hasattr(code, "__name__")
    assert code.__name__ == "Code_patched"
    assert (A_.field[:] == B_.field[:]).all()


def test_runtime_does_not_orchestrate_when_call_is_not_present() -> None:
    stencil_factory, _ = get_factories_single_tile_orchestrated(
        5, 5, 3, 0, backend=Backend.performance_cpu()
    )
    code = Code_NoCall(stencil_factory)

    # We didn't monkey patch the class, no __name__ on object
    # and the original Class name is intact
    assert not hasattr(code, "__name__")
    assert type(code).__name__ == "Code_NoCall"


def test_runtime_fail_when_not_super_init() -> None:
    with pytest.raises(
        RuntimeError, match="inherit from NDSLRuntime but didn't call super()"
    ):
        bad_code = BadCode_NoSuperInit()
