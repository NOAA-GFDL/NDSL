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
from ndsl.dsl.gt4py import FORWARD, computation, interval
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from ndsl.stencils.column_operations import column_max, column_min


nx = 1
ny = 1
nz = 10
nhalo = 0
backend = "dace:cpu"

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


class ColumnOperations:
    def __init__(self, stencil_factory: StencilFactory):
        grid_indexing = stencil_factory.grid_indexing

        def column_max_stencil(
            data: FloatField, max_value: FloatFieldIJ, max_index: FloatFieldIJ
        ):
            from __externals__ import k_end

            with computation(FORWARD), interval(0, 1):
                max_value, max_index = column_max(data, 0, k_end)

        def column_min_stencil(
            data: FloatField, min_value: FloatFieldIJ, min_index: FloatFieldIJ
        ):
            from __externals__ import k_end

            with computation(FORWARD), interval(0, 1):
                min_value, min_index = column_min(data, 5, k_end)

        self._column_max_stencil = stencil_factory.from_dims_halo(
            func=column_max_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self._column_min_stencil = stencil_factory.from_dims_halo(
            func=column_min_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(
        self,
        data: FloatField,
        max_value: FloatFieldIJ,
        max_index: FloatFieldIJ,
        min_value: FloatFieldIJ,
        min_index: FloatFieldIJ,
    ):
        self._column_max_stencil(data, max_value, max_index)
        self._column_min_stencil(data, min_value, min_index)


def test_column_operations():
    data = Quantity(
        data=np.zeros([nx, ny, nz]),
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="n/a",
    )
    data.field[:] = [
        47.3821,
        2.9157,
        88.6034,
        71.9275,
        53.1412,
        19.4783,
        94.2258,
        36.8099,
        64.0175,
        7.3504,
    ]

    max_value = Quantity(
        data=np.zeros([nx, ny]),
        dims=[X_DIM, Y_DIM],
        units="n/a",
    )
    max_index = Quantity(
        data=np.zeros([nx, ny]),
        dims=[X_DIM, Y_DIM],
        units="n/a",
    )
    min_value = Quantity(
        data=np.zeros([nx, ny]),
        dims=[X_DIM, Y_DIM],
        units="n/a",
    )
    min_index = Quantity(
        data=np.zeros([nx, ny]),
        dims=[X_DIM, Y_DIM],
        units="n/a",
    )

    code = ColumnOperations(stencil_factory)
    print("initalized the class")
    code(data, max_value, max_index, min_value, min_index)

    assert max_value.field[:] == np.max(data.field[:], axis=2)
    assert max_index.field[:] == np.argmax(data.field[:], axis=2)
    assert min_value.field[:] == np.min(data.field[:, :, 5:], axis=2)
    assert min_index.field[:] == 5 + np.argmin(data.field[:, :, 5:], axis=2)
