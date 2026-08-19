from collections.abc import Sequence
from typing import Any

import dace
import numpy as np

from ndsl.config import Backend
from ndsl.optional_imports import cupy
from ndsl.quantity import Quantity

if cupy is None:
    import numpy as cupy


class Local(Quantity):
    """Local is a Quantity that cannot be used outside of the class
    it was allocated in."""

    def __init__(
        self,
        data: np.ndarray | cupy.ndarray,
        dims: Sequence[str],
        units: str,
        *,
        backend: Backend,
        origin: Sequence[int] | None = None,
        extent: Sequence[int] | None = None,
        allow_mismatch_float_precision: bool = False,
    ):
        # Initialize memory to obviously wrong value - Local should _not_ be expected
        # to be zero'ed.
        data[:] = 123456789
        self._on_gpu = backend.is_gpu_backend()

        super().__init__(
            data,
            dims,
            units,
            origin=origin,
            extent=extent,
            allow_mismatch_float_precision=allow_mismatch_float_precision,
            backend=backend,
        )

    def __descriptor__(self) -> Any:
        """Locals uses `Quantity.__descriptor__` and flag itself as transient."""
        data = dace.data.create_datadescriptor(self._data)
        data.transient = True if not self._on_gpu else False
        return data

    def __repr__(self) -> str:
        return (
            f"Local(\n    data=\n{self._data},\n    dims={self.dims},\n"
            f"    units={self.units},\n    origin={self.origin},\n"
            f"    extent={self.extent}\n)"
        )


class ConstantQuantity(Quantity):
    """ConstantQuantity is a Quantity that cannot be written to outside of it's allocation
    (notably with the `NDSLRuntime.make_constant_quantity` method)."""

    def __init__(
        self,
        data: np.ndarray | cupy.ndarray | Quantity,
        dims: Sequence[str],
        units: str,
        *,
        backend: Backend,
        origin: Sequence[int] | None = None,
        extent: Sequence[int] | None = None,
        allow_mismatch_float_precision: bool = False,
    ):
        super().__init__(
            data,
            dims,
            units,
            origin=origin,
            extent=extent,
            allow_mismatch_float_precision=allow_mismatch_float_precision,
            backend=backend,
        )

    def __setitem__(self, subscript: Any, value: Any) -> None:
        raise RuntimeError(
            "ConstantQuantity cannot be written too after initialization"
        )
