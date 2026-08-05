import importlib

import numpy as np

import ndsl.dsl


def test_type_precision_select():
    original_precision = ndsl.dsl.NDSL_GLOBAL_PRECISION

    ndsl.dsl.NDSL_GLOBAL_PRECISION = 32
    importlib.reload(ndsl.dsl.typing)
    assert ndsl.dsl.typing.Float == np.float32
    assert ndsl.dsl.typing.Int == np.int32

    ndsl.dsl.NDSL_GLOBAL_PRECISION = 64
    importlib.reload(ndsl.dsl.typing)
    assert ndsl.dsl.typing.Float == np.float64
    assert ndsl.dsl.typing.Int == np.int64

    ndsl.dsl.NDSL_GLOBAL_PRECISION = original_precision
