from typing import Any


class RaiseWhenAccessed:
    def __init__(self, err: ModuleNotFoundError) -> None:
        self._err = err

    def __getattr__(self, _: Any) -> None:
        raise self._err

    def __call__(self, *args: Any, **kwargs: dict) -> None:
        raise self._err


try:
    import zarr
except ModuleNotFoundError as err:
    zarr = RaiseWhenAccessed(err)

try:
    import cupy
except ImportError:
    cupy = None

if cupy is not None:
    # Cupy might be available - but not the device
    try:
        cupy.cuda.runtime.deviceSynchronize()
    except cupy.cuda.runtime.CUDARuntimeError:
        cupy = None
