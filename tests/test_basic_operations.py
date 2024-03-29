import numpy as np

from ndsl import (
    CompilationConfig,
    DaceConfig,
    DaCeOrchestration,
    GridIndexing,
    Quantity,
    RunMode,
    StencilConfig,
    StencilFactory,
)
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.stencils import basic_operations as basic


nx = 20
ny = 20
nz = 79
nhalo = 0
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

    infield = Quantity(
        data=np.zeros([20, 20, 79]),
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    outfield = Quantity(
        data=np.ones([20, 20, 79]),
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    copy(f_in=infield.data, f_out=outfield.data)

    assert (infield.data == outfield.data).any()


def test_adjustmentfactor():
    adfact = AdjustmentFactor(stencil_factory)

    factorfield = Quantity(
        data=np.full(shape=[20, 20], fill_value=2.0),
        dims=[X_DIM, Y_DIM],
        units="m",
    )

    outfield = Quantity(
        data=np.full(shape=[20, 20, 79], fill_value=2.0),
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    testfield = Quantity(
        data=np.full(shape=[20, 20, 79], fill_value=4.0),
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    adfact(factor=factorfield.data, f_out=outfield.data)
    assert (outfield.data == testfield.data).any()


def test_setvalue():
    setvalue = SetValue(stencil_factory)

    outfield = Quantity(
        data=np.zeros(shape=[20, 20, 79]),
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    testfield = Quantity(
        data=np.full(shape=[20, 20, 79], fill_value=2.0),
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    setvalue(f_out=outfield.data, value=2.0)

    assert (outfield.data == testfield.data).any()


def test_adjustdivide():
    addiv = AdjustDivide(stencil_factory)

    factorfield = Quantity(
        data=np.full(shape=[20, 20, 79], fill_value=2.0),
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    outfield = Quantity(
        data=np.full(shape=[20, 20, 79], fill_value=2.0),
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    testfield = Quantity(
        data=np.full(shape=[20, 20, 79], fill_value=1.0),
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    addiv(factor=factorfield.data, f_out=outfield.data)

    assert (outfield.data == testfield.data).any()
