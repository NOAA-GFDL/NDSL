import functools
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
from typing_extensions import Protocol


Number: TypeAlias = int | float | np.int32 | np.int64 | np.float32 | np.float64


class Allocator(Protocol):
    def __call__(self, shape: Iterable[int], dtype: type) -> None:
        pass


class AsyncRequest(Protocol):
    """Define the result of an over-the-network capable communication API"""

    def wait(self) -> None:
        """Block the current thread waiting for the request to be completed"""
        ...
