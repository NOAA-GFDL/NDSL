from typing import Sequence

import numpy as np

from ndsl.optional_imports import cupy
from ndsl.quantity import Quantity


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
