from typing import Any

import pytest

from ndsl import (
    NDSLRuntime,
    OptimizationConfig,
    QuantityFactory,
    StencilFactory,
    stencils,
)
from ndsl.boilerplate import (
    get_factories_single_tile,
    get_factories_single_tile_orchestrated,
)
from ndsl.config import Backend
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, FloatField
from ndsl.stencils.basic_operations import copy


class Code(NDSLRuntime):
    def __init__(
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ) -> None:
        super().__init__(stencil_factory)
        self.copy = stencil_factory.from_dims_halo(
            copy, compute_dims=[I_DIM, J_DIM, K_DIM]
        )
        self.local = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])

    def test_check(self) -> None:
        # HACK Temporarily mark this function as orchestrated to allow
        # testing the descriptor (from the python side).
        self._ndsl_orchestrated_methods.append("test_check")

        # Test the descriptor (with the override in place).
        assert self.local.__descriptor__().transient

        # Restore test override.
        self._ndsl_orchestrated_methods.remove("test_check")

    def __call__(self, A, B) -> None:  # type: ignore[no-untyped-def]
        self.copy(A, self.local)
        self.copy(self.local, B)


def test_runtime_make_local() -> None:
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=5, ny=5, nz=3, nhalo=0, backend=Backend.python()
    )
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


def test_runtime_has_orchestrated_call() -> None:
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        nx=5, ny=5, nz=3, nhalo=0, backend=Backend.cpu()
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
    class Code_NoCall(NDSLRuntime):
        def __init__(self, stencil_factory: StencilFactory) -> None:
            super().__init__(stencil_factory)
            pass

        def run(self, A: Any, B: Any) -> None:
            pass

    stencil_factory, _ = get_factories_single_tile_orchestrated(
        nx=5, ny=5, nz=3, nhalo=0, backend=Backend.cpu()
    )
    code = Code_NoCall(stencil_factory)

    # We didn't monkey patch the class, no __name__ on object
    # and the original Class name is intact
    assert not hasattr(code, "__name__")
    assert type(code).__name__ == "Code_NoCall"


def test_runtime_fails_when_not_super_init() -> None:
    class BadCode_NoSuperInit(NDSLRuntime):
        def __init__(self) -> None:
            # Forget to call super().__init__(...)
            pass

    with pytest.raises(
        RuntimeError, match="inherit from NDSLRuntime but didn't call super()"
    ):
        BadCode_NoSuperInit()


def test_runtime_with_performance_config() -> None:
    class CustomPerformanceConfig(NDSLRuntime):
        def __init__(
            self,
            stencil_factory: StencilFactory,
            optimization_config: OptimizationConfig,
        ) -> None:
            super().__init__(stencil_factory, optimization_config)
            self.copy = stencil_factory.from_dims_halo(
                stencils.copy, compute_dims=[I_DIM, J_DIM, K_DIM]
            )

        def __call__(self, src, dst) -> None:  # type: ignore[no-untyped-def]
            self.copy(src, dst)

    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        nx=5, ny=5, nz=3, nhalo=0, backend=Backend.cpu()
    )

    # setup code
    config = OptimizationConfig()
    code = CustomPerformanceConfig(stencil_factory, config)

    # setup inputs/outputs
    src = quantity_factory.ones(dims=[I_DIM, J_DIM, K_DIM], units="n/a")
    dst = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="n/a")

    # call code with inputs/outputs
    code(src, dst)

    assert (src.field[:] == dst.field[:]).all()


def test_runtime_fails_local_in_non_orchestrated_code_path() -> None:
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=5, ny=5, nz=3, nhalo=0, backend=Backend.python()
    )

    class NonOrchestratedLocalUsage(NDSLRuntime):
        def __init__(
            self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
        ) -> None:
            super().__init__(stencil_factory)
            self._copy_local = stencil_factory.from_dims_halo(
                func=copy,
                compute_dims=[I_DIM, J_DIM, K_DIM],
            )
            self.local = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])

            # Bad usage of Local in non-orchestrated code path
            self.local[:] = 2.0

        def __call__(self, in_field: FloatField) -> None:  # type: ignore
            self._copy_local(self._my_local, in_field)

    with pytest.raises(
        RuntimeError,
        match="Forbidden Local access: .* called in non-orchestrated method",
    ):
        NonOrchestratedLocalUsage(stencil_factory, quantity_factory)


@pytest.mark.parametrize(
    "backend",
    ["st:python:cpu:IJK", "st:dace:cpu:IJK", "orch:dace:cpu:IJK", "orch:dace:cpu:KJI"],
)
def test_runtime_use_local_in_orchestrated_code_path(backend: str) -> None:
    class Code(NDSLRuntime):
        def __init__(
            self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
        ) -> None:
            super().__init__(stencil_factory)
            self._copy_local = stencil_factory.from_dims_halo(
                func=copy, compute_dims=[I_DIM, J_DIM, K_DIM]
            )
            self._my_local = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])

        def __call__(self, field: FloatField, value: Float) -> None:  # type: ignore
            self._my_local[:] = value
            self._copy_local(self._my_local, field)

    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=5, ny=5, nz=3, nhalo=0, backend=Backend(backend)
    )

    array = quantity_factory.ones(dims=[I_DIM, J_DIM, K_DIM], units="n/a")
    value = 2.0

    code = Code(stencil_factory, quantity_factory)
    code(array, value)

    assert (array.field == value).all()
