from __future__ import annotations

import logging
import os
import sys
from typing import Annotated

from ndsl.comm.mpi import MPI


# Python log levels are hierarchical. The following dict is sorted by
# severity. Setting the log level to "info" means that "info" and everything
# more severe (e.g. "warning") will be logged.
AVAILABLE_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LogLowerLevelsOnRankZeroOnly(logging.Filter):
    """Allow logging on rank 0 - all other logs are cancelled
    unless:
    - `NDSL_LOG_ALL` is `True`
    - OR the log level is >= `Error`
    """

    def filter(self, record: logging.LogRecord) -> bool:
        log_all = os.getenv("NDSL_LOG_ALL", "False").lower() == "true"
        if log_all:
            return True

        rank = MPI.COMM_WORLD.Get_rank()
        if record.levelno >= logging.ERROR:
            return True

        if rank == 0:
            return True

        return False


def _get_log_level(default: str = "info") -> str:
    loglevel = os.getenv("NDSL_LOGLEVEL", default).lower()

    if loglevel in AVAILABLE_LOG_LEVELS.keys():
        return loglevel

    logging.warning(
        f"Unknown log level '{loglevel}', falling back to '{default}'. Valid values are: {AVAILABLE_LOG_LEVELS.keys()}."
    )
    return default


def _ndsl_logger() -> logging.Logger:
    log_level = _get_log_level()

    name_log = logging.getLogger(__name__)
    name_log.setLevel(AVAILABLE_LOG_LEVELS[log_level])

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(AVAILABLE_LOG_LEVELS[log_level])
    formatter = logging.Formatter(
        fmt=(
            f"%(asctime)s|%(levelname)s|rank {MPI.COMM_WORLD.Get_rank()}|"
            "%(name)s:%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    name_log.addHandler(handler)
    name_log.addFilter(LogLowerLevelsOnRankZeroOnly())
    return name_log


def _ndsl_logger_on_rank_0() -> logging.Logger:
    log_level = _get_log_level()

    name_log = logging.getLogger(f"{__name__}_on_rank_0")
    name_log.setLevel(AVAILABLE_LOG_LEVELS[log_level])

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(AVAILABLE_LOG_LEVELS[log_level])
        formatter = logging.Formatter(
            fmt=(
                f"%(asctime)s|%(levelname)s|rank {MPI.COMM_WORLD.Get_rank()}|"
                ": %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        name_log.addHandler(handler)
    else:
        name_log.disabled = True
    return name_log


ndsl_log: Annotated[logging.Logger, "NDSL Python logger, logs on all rank"] = (
    _ndsl_logger()
)
ndsl_log.info(f"Log level: {_get_log_level()}")

ndsl_log_on_rank_0: Annotated[
    logging.Logger, "NDSL Python logger, logs on rank 0 only"
] = _ndsl_logger_on_rank_0()
