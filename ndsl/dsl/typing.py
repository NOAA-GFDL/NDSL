from typing import Tuple, TypeAlias, Union, cast

import gt4py.cartesian.gtscript as gtscript
import numpy as np

from ndsl.dsl import NDSL_GLOBAL_PRECISION


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

# We redefine the type as a way to distinguish
# the model definition of a float to other usage of the
# common numpy type in the rest of the code.
NDSL_32BIT_FLOAT_TYPE: TypeAlias = np.float32
NDSL_32BIT_INT_TYPE: TypeAlias = np.int32
NDSL_64BIT_FLOAT_TYPE: TypeAlias = np.float64
NDSL_64BIT_INT_TYPE: TypeAlias = np.int64


def global_set_floating_point_precision() -> Tuple[TypeAlias, TypeAlias]:
    """Set the global floating point precision for all reference
    to Float in the codebase. Defaults to 64 bit."""
    global Float, Int
    if NDSL_GLOBAL_PRECISION == 64:
        return NDSL_64BIT_FLOAT_TYPE, NDSL_64BIT_INT_TYPE
    elif NDSL_GLOBAL_PRECISION == 32:
        return NDSL_32BIT_FLOAT_TYPE, NDSL_32BIT_INT_TYPE
    raise NotImplementedError(
        f"{NDSL_GLOBAL_PRECISION} bit precision not implemented or tested"
    )


# Default float and int types
Float, Int = global_set_floating_point_precision()
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
