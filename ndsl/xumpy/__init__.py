"""
This package presents a unified API for `numpy` and `cupy` common operations
while natively handling NDSL memory concepts (Quantity, etc.) and mixed precision
as defined by NDSL_LITERAL_PRECISION

It's aim is to provide a performance portable API for writing common numerics
in Python.

Development scheme is to "add feature as needed".
"""

from ndsl.xumpy.alloc import empty, full, ones, random, zeros
from ndsl.xumpy.count_nonzero import count_nonzero
from ndsl.xumpy.minmax import max, max_on_horizontal_plane, min


__all__ = [
    "max",
    "min",
    "max_on_horizontal_plane",
    "count_nonzero",
    "zeros",
    "ones",
    "empty",
    "full",
    "random",
]
