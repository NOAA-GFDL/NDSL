# flake8: noqa
from ndsl.comm.local_comm import ConcurrencyError
from ndsl.units import UnitsError


class OutOfBoundsError(ValueError):
    pass
