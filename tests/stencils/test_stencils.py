import numpy as np
import pytest

from ndsl import QuantityFactory, StencilFactory
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import FORWARD, computation, interval
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, set_4d_field_size
from ndsl.stencils import CopyCornersXY
from ndsl.stencils.column_operations import (
    column_max,
    column_max_ddim,
    column_min,
    column_min_ddim,
)


FloatField_ddim = set_4d_field_size(2, Float)


@pytest.fixture
def boilerplate() -> tuple[StencilFactory, QuantityFactory]:
    return get_factories_single_tile(nx=1, ny=1, nz=10, nhalo=0, backend="dace:cpu")


class ColumnOperations:
    def __init__(self, stencil_factory: StencilFactory):

        def column_max_stencil(
            data: FloatField, max_value: FloatFieldIJ, max_index: FloatFieldIJ
        ):
            from __externals__ import k_end

            with computation(FORWARD), interval(0, 1):
                max_value, max_index = column_max(data, 0, k_end)

        def column_max_ddim_stencil(
            data: FloatField_ddim, max_value: FloatFieldIJ, max_index: FloatFieldIJ
        ):
            from __externals__ import k_end

            with computation(FORWARD), interval(0, 1):
                max_value, max_index = column_max_ddim(data, 1, 0, k_end)

        def column_min_stencil(
            data: FloatField, min_value: FloatFieldIJ, min_index: FloatFieldIJ
        ):
            from __externals__ import k_end

            with computation(FORWARD), interval(0, 1):
                min_value, min_index = column_min(data, 5, k_end)

        def column_min_ddim_stencil(
            data: FloatField_ddim, min_value: FloatFieldIJ, min_index: FloatFieldIJ
        ):
            from __externals__ import k_end

            with computation(FORWARD), interval(0, 1):
                min_value, min_index = column_min_ddim(data, 1, 5, k_end)

        self._column_max_stencil = stencil_factory.from_dims_halo(
            func=column_max_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self._column_max_ddim_stencil = stencil_factory.from_dims_halo(
            func=column_max_ddim_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self._column_min_stencil = stencil_factory.from_dims_halo(
            func=column_min_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self._column_min_ddim_stencil = stencil_factory.from_dims_halo(
            func=column_min_ddim_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(
        self,
        data: FloatField,
        max_value: FloatFieldIJ,
        max_index: FloatFieldIJ,
        min_value: FloatFieldIJ,
        min_index: FloatFieldIJ,
        data_ddim: FloatField_ddim,
        max_value_ddim: FloatFieldIJ,
        max_index_ddim: FloatFieldIJ,
        min_value_ddim: FloatFieldIJ,
        min_index_ddim: FloatFieldIJ,
    ):
        self._column_max_stencil(data, max_value, max_index)
        self._column_max_ddim_stencil(data_ddim, max_value_ddim, max_index_ddim)
        self._column_min_stencil(data, min_value, min_index)
        self._column_min_ddim_stencil(data_ddim, min_value_ddim, min_index_ddim)


def test_column_operations(boilerplate):
    stencil_factory, quantity_factory = boilerplate
    quantity_factory.add_data_dimensions({"ddim": 2})
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

    data_ddim = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM, "ddim"], "n/a")
    data_ddim.field[:] = [
        [
            [
                [47.3821, 27.4825],
                [2.9157, 93.1242],
                [88.6034, 14.6347],
                [71.9275, 58.2094],
                [53.1412, 6.2369],
                [19.4783, 71.5457],
                [94.2258, 42.3091],
                [36.8099, 89.7718],
                [64.0175, 63.1910],
                [7.3504, 3.4991],
            ]
        ]
    ]

    max_value = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
    max_index = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
    min_value = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
    min_index = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
    max_value_ddim = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
    max_index_ddim = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
    min_value_ddim = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
    min_index_ddim = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")

    code = ColumnOperations(stencil_factory)
    code(
        data,
        max_value,
        max_index,
        min_value,
        min_index,
        data_ddim,
        max_value_ddim,
        max_index_ddim,
        min_value_ddim,
        min_index_ddim,
    )

    # 3d field tests
    assert max_value.field[:] == np.max(data.field[:], axis=2)
    assert max_index.field[:] == np.argmax(data.field[:], axis=2)
    assert min_value.field[:] == np.min(data.field[:, :, 5:], axis=2)
    assert min_index.field[:] == 5 + np.argmin(data.field[:, :, 5:], axis=2)

    # 4d field tests
    assert max_value_ddim.field[:] == np.max(data_ddim.field[:, :, :, 1], axis=2)
    assert max_index_ddim.field[:] == np.argmax(data_ddim.field[:, :, :, 1], axis=2)
    assert min_value_ddim.field[:] == np.min(data_ddim.field[:, :, 5:, 1], axis=2)
    assert min_index_ddim.field[:] == 5 + np.argmin(
        data_ddim.field[:, :, 5:, 1], axis=2
    )


def test_CopyCornersXY_deprecation(boilerplate) -> None:
    stencil_factory, _ = boilerplate

    with pytest.deprecated_call(match="Usage of CopyCornersXY is deprecated"):
        CopyCornersXY(stencil_factory, [X_DIM, Y_DIM, Z_DIM], None)
