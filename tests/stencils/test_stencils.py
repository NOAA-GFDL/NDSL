import numpy as np
from ndsl import Quantity
from ndsl.stencils.column_operations import column_max, column_min


def test_column_operations():
    shape = (1, 1, 10)
    data = np.zeros(shape, dtype=float)
    quantity = Quantity(
        data,
        origin=(0, 0),
        extent=shape,
        dims=["dim1", "dim2", "dim3"],
        units="units",
        gt4py_backend="numpy",
    )
    quantity.field[:] = [
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

    max_value, max_index = column_max(quantity, 0, 9)
    min_value, min_index = column_min(quantity, 0, 9)

    assert max_value == 94.2258
    assert max_index == 5
    assert min_value == 2.9157
    assert min_index == 1
