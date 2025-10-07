# flake8: noqa
import warnings

from ndsl.comm.local_comm import ConcurrencyError
from ndsl.units import UnitsError


class OutOfBoundsError(ValueError):
    def __init__(self, *args) -> None:
        warnings.warn(
            "Usage of `OutOfBoundsError` is discouraged. The class will be "
            "removed in the next version in favor of using the built-in `IndexError`.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args)
