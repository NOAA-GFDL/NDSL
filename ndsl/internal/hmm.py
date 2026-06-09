"""
This module's sole role is to attempt to setup DSL-wide usage of Heterogeneous
Memory Management (HMM) by leveraging CuPy's experimental support.

We rely on Nvidia's docs and CuPy's evolving support (reference in inline comment). It boils down to:

HMM requires:
    - a device that can access paged RAM on an OS that has an HMM service running on kernel
    - the `CUPY_ENABLE_UMP` environement variable set to 1

⚠️ If `CUPY_ENABLE_UMP` is set _but_ the device/OS cannot support pageable memory, the upload/download
will fail as CuPy does blind pointer-binding. ⚠️

If HMM is availbale we:
    - flip `cupy` malloc managed allocator to the system one
    - set the `numpy` allocator to the system one (using the `numpy_allocator` package)

Once set globally - every allocation will happen on paged memory and every cupy initiated transfer will bypass
calls for upload/download and do pointer-mapping.

In addition, we make a good effort to log when HMM could be used hardware-wise but isn't due to a software
or configuration limitation.
"""

import ctypes
import os

from ndsl.logging import ndsl_log
from ndsl.optional_imports import cupy as cp
from ndsl.optional_imports import numpy_allocator as np_allocator


def _is_hmm_available() -> bool:
    if cp is None:
        return False

    if not cp.cuda.runtime.deviceGetAttribute(
        cp.cuda.runtime.cudaDevAttrPageableMemoryAccess, 0
    ):
        return False

    if "CUPY_ENABLE_UMP" not in os.environ or os.environ["CUPY_ENABLE_UMP"] != "1":
        ndsl_log.info("HMM possible but OFF: set `CUPY_ENABLE_UMP=1` to activate HMM")
        return False

    if np_allocator is None:
        ndsl_log.info("HMM possible - but `numpy_allocator` is not installed")

    ndsl_log.info("HMM is ON")
    return True


if _is_hmm_available():
    # Based on https://github.com/cupy/cupy/issues/8711 and
    # https://docs.cupy.dev/en/stable/user_guide/memory.html#unified-memory-programming-ump-support-experimental
    import cupy._core.numpy_allocator as ac

    # System allocator for cupy
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.memory.malloc_system).malloc)

    # System allocator for numpy
    lib = ctypes.CDLL(ac.__file__)

    class my_allocator(metaclass=np_allocator.type):
        _calloc_ = ctypes.addressof(lib._calloc)
        _malloc_ = ctypes.addressof(lib._malloc)
        _realloc_ = ctypes.addressof(lib._realloc)
        _free_ = ctypes.addressof(lib._free)

    my_allocator.__enter__()  # flip the allocator
