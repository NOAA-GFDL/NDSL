import warnings
from collections.abc import Sequence
from typing import Any

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
        *,
        backend: str | None = None,
        origin: Sequence[int] | None = None,
        extent: Sequence[int] | None = None,
        gt4py_backend: str | None = None,
        allow_mismatch_float_precision: bool = False,
    ):
        if gt4py_backend is not None:
            warnings.warn(
                "gt4py_backend is deprecated. Use `backend` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if backend is None:
                backend = gt4py_backend

        if backend is None:
            warnings.warn(
                "`backend` will be a required argument starting with the next version of NDSL.",
                DeprecationWarning,
                stacklevel=2,
            )

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
        data = dace.data.create_datadescriptor(self.data)
        data.transient = True
        return data
