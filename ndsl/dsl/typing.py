import os
from typing import Tuple, TypeAlias, Union, cast

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


# Depreciated version of get_precision, but retained for a PACE dependency
def floating_point_precision() -> int:
    return int(os.getenv("PACE_FLOAT_PRECISION", "64"))


def get_precision() -> int:
    return int(os.getenv("PACE_FLOAT_PRECISION", "64"))


# We redefine the type as a way to distinguish
# the model definition of a float to other usage of the
# common numpy type in the rest of the code.
NDSL_32BIT_FLOAT_TYPE: TypeAlias = np.float32
NDSL_64BIT_FLOAT_TYPE: TypeAlias = np.float64
NDSL_32BIT_INT_TYPE: TypeAlias = np.int32
NDSL_64BIT_INT_TYPE: TypeAlias = np.int64


def global_set_precision() -> Tuple[TypeAlias, TypeAlias]:
    """Set the global precision for all references of
    Float and Int in the codebase. Defaults to 64 bit."""
    global Float, Int
    precision_in_bit = get_precision()
    if precision_in_bit == 64:
        return NDSL_64BIT_FLOAT_TYPE, NDSL_64BIT_INT_TYPE
    elif precision_in_bit == 32:
        return NDSL_32BIT_FLOAT_TYPE, NDSL_32BIT_INT_TYPE
    else:
        raise NotImplementedError(
            f"{precision_in_bit} bit precision not implemented or tested"
        )


# Default float and int types
Float, Int = global_set_precision()
Bool = np.bool_

FloatField = Field[gtscript.IJK, Float]
FloatField64 = Field[gtscript.IJK, np.float64]
FloatField32 = Field[gtscript.IJK, np.float32]
FloatFieldI = Field[gtscript.I, Float]
FloatFieldI64 = Field[gtscript.I, np.float64]
FloatFieldI32 = Field[gtscript.I, np.float32]
FloatFieldJ = Field[gtscript.J, Float]
FloatFieldJ64 = Field[gtscript.J, np.float64]
FloatFieldJ32 = Field[gtscript.J, np.float32]
FloatFieldIJ = Field[gtscript.IJ, Float]
FloatFieldIJ64 = Field[gtscript.IJ, np.float64]
FloatFieldIJ32 = Field[gtscript.IJ, np.float32]
FloatFieldK = Field[gtscript.K, Float]
FloatFieldK64 = Field[gtscript.K, np.float64]
FloatFieldK32 = Field[gtscript.K, np.float32]

IntField = Field[gtscript.IJK, Int]
IntField64 = Field[gtscript.IJK, np.int64]
IntField32 = Field[gtscript.IJK, np.int32]
IntFieldI = Field[gtscript.I, Int]
IntFieldI64 = Field[gtscript.I, np.int64]
IntFieldI32 = Field[gtscript.I, np.int32]
IntFieldJ = Field[gtscript.J, Int]
IntFieldJ64 = Field[gtscript.J, np.int64]
IntFieldJ32 = Field[gtscript.J, np.int32]
IntFieldIJ = Field[gtscript.IJ, Int]
IntFieldIJ64 = Field[gtscript.IJ, np.int64]
IntFieldIJ32 = Field[gtscript.IJ, np.int32]
IntFieldK = Field[gtscript.K, Int]
IntFieldK64 = Field[gtscript.K, np.int64]
IntFieldK32 = Field[gtscript.K, np.int32]

BoolField = Field[gtscript.IJK, Bool]
BoolFieldI = Field[gtscript.I, Bool]
BoolFieldJ = Field[gtscript.J, Bool]
BoolFieldK = Field[gtscript.K, Bool]
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


def is_float(dtype: type):
    """Expected floating point type"""
    return dtype in [
        Float,
        float,
        np.float16,
        np.float32,
        np.float64,
    ]
