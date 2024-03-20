from gt4py.storage import full, ones, zeros

from ndsl import (
    CompilationConfig,
    DaceConfig,
    DaCeOrchestration,
    GridIndexing,
    RunMode,
    StencilConfig,
    StencilFactory,
)
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.stencils import basic_operations as basic


nx = 20
ny = 20
nz = 79
nhalo = 3
backend = "numpy"

dace_config = DaceConfig(
    communicator=None, backend=backend, orchestration=DaCeOrchestration.Python
)

compilation_config = CompilationConfig(
    backend=backend,
    rebuild=True,
    validate_args=True,
    format_source=False,
    device_sync=False,
    run_mode=RunMode.BuildAndRun,
    use_minimal_caching=False,
)

stencil_config = StencilConfig(
    compare_to_numpy=False,
    compilation_config=compilation_config,
    dace_config=dace_config,
)

grid_indexing = GridIndexing(
    domain=(nx, ny, nz),
    n_halo=nhalo,
    south_edge=True,
    north_edge=True,
    west_edge=True,
    east_edge=True,
)

stencil_factory = StencilFactory(config=stencil_config, grid_indexing=grid_indexing)


class Copy:
    def __init__(self, stencil_factory: StencilFactory):
        grid_indexing = stencil_factory.grid_indexing
        self._copy_stencil = stencil_factory.from_origin_domain(
            basic.copy_defn,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        f_in: FloatField,
        f_out: FloatField,
    ):
        self._copy_stencil(f_in, f_out)


class AdjustmentFactor:
    def __init__(self, stencil_factory: StencilFactory):
        grid_indexing = stencil_factory.grid_indexing
        self._adjustmentfactor_stencil = stencil_factory.from_origin_domain(
            basic.adjustmentfactor_stencil_defn,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        factor: FloatFieldIJ,
        f_out: FloatField,
    ):
        self._adjustmentfactor_stencil(factor, f_out)


class SetValue:
    def __init__(self, stencil_factory: StencilFactory):
        grid_indexing = stencil_factory.grid_indexing
        self._set_value_stencil = stencil_factory.from_origin_domain(
            basic.set_value_defn,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        f_out: FloatField,
        value: Float,
    ):
        self._set_value_stencil(f_out, value)


class AdjustDivide:
    def __init__(self, stencil_factory: StencilFactory):
        grid_indexing = stencil_factory.grid_indexing
        self._adjust_divide_stencil = stencil_factory.from_origin_domain(
            basic.adjust_divide_stencil,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        factor: FloatField,
        f_out: FloatField,
    ):
        self._adjust_divide_stencil(factor, f_out)


def test_copy():
    copy = Copy(stencil_factory)

    infield = zeros(
        backend=backend, dtype=Float, shape=(nx + 2 * nhalo, ny + 2 * nhalo, nz)
    )

    outfield = ones(
        backend=backend, dtype=Float, shape=(nx + 2 * nhalo, ny + 2 * nhalo, nz)
    )

    copy(f_in=infield, f_out=outfield)

    assert infield.all() == outfield.all()


def test_adjustmentfactor():
    adfact = AdjustmentFactor(stencil_factory)

    factorfield = ones(
        backend=backend, dtype=Float, shape=(nx + 2 * nhalo, ny + 2 * nhalo)
    )

    outfield = ones(
        backend=backend, dtype=Float, shape=(nx + 2 * nhalo, ny + 2 * nhalo, nz)
    )

    testfield = full(
        backend=backend,
        dtype=Float,
        shape=(nx + 2 * nhalo, ny + 2 * nhalo),
        fill_value=26.0,
    )

    adfact(factor=factorfield, f_out=outfield)
    assert outfield.any() == testfield.any()


def test_setvalue():
    setvalue = SetValue(stencil_factory)

    outfield = zeros(
        backend=backend,
        dtype=Float,
        shape=(nx + 2 * nhalo, ny + 2 * nhalo, nz),
    )

    testfield = full(
        backend=backend,
        dtype=Float,
        shape=(nx + 2 * nhalo, ny + 2 * nhalo, nz),
        fill_value=2.0,
    )

    setvalue(f_out=outfield, value=2.0)

    assert outfield.any() == testfield.any()


def test_adjustdivide():
    addiv = AdjustDivide(stencil_factory)

    factorfield = full(
        backend=backend,
        dtype=Float,
        shape=(nx + 2 * nhalo, ny + 2 * nhalo, nz),
        fill_value=2.0,
    )

    outfield = ones(
        backend=backend,
        dtype=Float,
        shape=(nx + 2 * nhalo, ny + 2 * nhalo, nz),
    )

    testfield = full(
        backend=backend,
        dtype=Float,
        shape=(nx + 2 * nhalo, ny + 2 * nhalo),
        fill_value=13.0,
    )

    addiv(factor=factorfield, f_out=outfield)
    assert outfield.any() == testfield.any()
