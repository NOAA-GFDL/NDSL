from typing import Any, Sequence

import dace
import numpy as np

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
        origin: Sequence[int] | None = None,
        extent: Sequence[int] | None = None,
        gt4py_backend: str | None = None,
        allow_mismatch_float_precision: bool = False,
    ):
        super().__init__(
            data,
            dims,
            units,
            origin,
            extent,
            gt4py_backend,
            allow_mismatch_float_precision,
        )
        self._transient = True

    def __descriptor__(self) -> Any:
        """Locals uses `Quantity.__descriptor__` and flag itself as transient."""
        data = dace.data.create_datadescriptor(self.data)
        data.transient = True
        return data
