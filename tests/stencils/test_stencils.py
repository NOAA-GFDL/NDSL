import numpy as np

from ndsl import StencilFactory
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import FORWARD, computation, interval
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from ndsl.stencils.column_operations import column_max, column_min


nx = 1
ny = 1
nz = 10
nhalo = 0
backend = "dace:cpu"

stencil_factory, quantity_factory = get_factories_single_tile(
    nx, ny, nz, nhalo, backend
)


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
    data = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
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

    max_value = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
    max_index = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
    min_value = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
    min_index = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")

    code = ColumnOperations(stencil_factory)
    code(data, max_value, max_index, min_value, min_index)

    assert max_value.field[:] == np.max(data.field[:], axis=2)
    assert max_index.field[:] == np.argmax(data.field[:], axis=2)
    assert min_value.field[:] == np.min(data.field[:, :, 5:], axis=2)
    assert min_index.field[:] == 5 + np.argmin(data.field[:, :, 5:], axis=2)
