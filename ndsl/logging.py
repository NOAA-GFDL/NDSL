from __future__ import annotations

import logging
import os
import sys
from typing import Annotated

from mpi4py import MPI


LOGLEVEL = os.environ.get("PACE_LOGLEVEL", "INFO").upper()

# Python log levels are hierarchical, therefore setting INFO
# means DEBUG and everything lower will be logged.
AVAILABLE_LOG_LEVELS = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _ndsl_logger() -> logging.Logger:
    name_log = logging.getLogger(__name__)
    name_log.setLevel(LOGLEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(LOGLEVEL)
    formatter = logging.Formatter(
        fmt=(
            f"%(asctime)s|%(levelname)s|rank {MPI.COMM_WORLD.Get_rank()}|"
            "%(name)s:%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    name_log.addHandler(handler)
    return name_log


def _ndsl_logger_on_rank_0() -> logging.Logger:
    name_log = logging.getLogger(f"{__name__}_on_rank_0")
    name_log.setLevel(LOGLEVEL)

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(LOGLEVEL)
        formatter = logging.Formatter(
            fmt=(
                f"%(asctime)s|%(levelname)s|rank {MPI.COMM_WORLD.Get_rank()}|"
                "%(name)s:%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        name_log.addHandler(handler)
    else:
        name_log.disabled = True
    return name_log


ndsl_log: Annotated[
    logging.Logger, "NDSL Python logger, logs on all rank"
] = _ndsl_logger()

ndsl_log_on_rank_0: Annotated[
    logging.Logger, "NDSL Python logger, logs on rank 0 only"
] = _ndsl_logger_on_rank_0()
