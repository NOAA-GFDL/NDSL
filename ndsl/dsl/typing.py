import os
from typing import Tuple, Union, cast

import gt4py.cartesian.gtscript as gtscript
import numpy as np


# A Field
Field = gtscript.Field
"""A gt4py field"""

# Axes
IJK = gtscript.IJK
IJ = gtscript.IJ
IK = gtscript.IK
JK = gtscript.JK
I = gtscript.I  # noqa: E741
J = gtscript.J  # noqa: E741
K = gtscript.K  # noqa: E741

# Union of valid data types (from gt4py.cartesian.gtscript)
DTypes = Union[bool, np.bool_, int, np.int32, np.int64, float, np.float32, np.float64]


def floating_point_precision() -> int:
    return int(os.getenv("PACE_FLOAT_PRECISION", "64"))


# We redefine the type as a way to distinguish
# the model definition of a float to other usage of the
# common numpy type in the rest of the code.
NDSL_32BIT_FLOAT_TYPE = np.float32
NDSL_64BIT_FLOAT_TYPE = np.float64


def global_set_floating_point_precision():
    """Set the global floating point precision for all reference
    to Float in the codebase. Defaults to 64 bit."""
    global Float
    precision_in_bit = floating_point_precision()
    if precision_in_bit == 64:
        return NDSL_64BIT_FLOAT_TYPE
    elif precision_in_bit == 32:
        return NDSL_32BIT_FLOAT_TYPE
    else:
        NotImplementedError(
            f"{precision_in_bit} bit precision not implemented or tested"
        )
    return None


# Default float and int types
Float = global_set_floating_point_precision()
Int = np.int_
Bool = np.bool_

FloatField = Field[gtscript.IJK, Float]
FloatFieldI = Field[gtscript.I, Float]
FloatFieldJ = Field[gtscript.J, Float]
FloatFieldIJ = Field[gtscript.IJ, Float]
FloatFieldK = Field[gtscript.K, Float]
IntField = Field[gtscript.IJK, Int]
IntFieldIJ = Field[gtscript.IJ, Int]
IntFieldK = Field[gtscript.K, Int]
BoolField = Field[gtscript.IJK, Bool]
BoolFieldIJ = Field[gtscript.IJ, Bool]

Index3D = Tuple[int, int, int]


def set_4d_field_size(n, dtype):
    """
    Defines a 4D field with a given size and type
    The extra data dimension is not parallel
    """
    return Field[gtscript.IJK, (dtype, (n,))]


def cast_to_index3d(val: Tuple[int, ...]) -> Index3D:
    if len(val) != 3:
        raise ValueError(f"expected 3d index, received {val}")
    return cast(Index3D, val)
