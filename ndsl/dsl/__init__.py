# Literal precision for both GT4Py & NDSL
import os
import sys
import warnings
from typing import Literal

from ndsl.comm.mpi import MPI
from ndsl.logging import ndsl_log


gt4py_config_module = "gt4py.cartesian.config"
if gt4py_config_module in sys.modules:
    raise RuntimeError(
        "`GT4Py` config imported before `ndsl` imported."
        " Please import `ndsl.dsl` or any `ndsl` module "
        " before any `gt4py` imports."
    )

# Literal precision handling


def _get_literal_precision(default: Literal["32", "64"] = "64") -> Literal["32", "64"]:
    precision = os.getenv("NDSL_LITERAL_PRECISION", default)

    expected: list[Literal["32", "64"]] = ["32", "64"]
    if precision in expected:
        return precision  # type: ignore

    ndsl_log.warning(
        f"Unexpected literal precision '{precision}', falling back to '{default}'. Valid values are {expected}."
    )
    return default


NDSL_GLOBAL_PRECISION = int(_get_literal_precision())
os.environ["GT4PY_LITERAL_INT_PRECISION"] = str(NDSL_GLOBAL_PRECISION)
os.environ["GT4PY_LITERAL_FLOAT_PRECISION"] = str(NDSL_GLOBAL_PRECISION)


# Set cache names for default gt backends workflow
import gt4py.cartesian.config  # noqa: E402


if MPI is not None:
    import os

    gt4py.cartesian.config.cache_settings["dir_name"] = os.environ.get(
        "GT_CACHE_DIR_NAME", f".gt_cache_{MPI.COMM_WORLD.Get_rank():06}"
    )

# Ensure DaCe backends are registered in GT4Py
with warnings.catch_warnings(record=True) as caught:
    # Cause all warnings to always be triggered.
    warnings.simplefilter("always")

    # Load GT4Py backends
    import gt4py.cartesian.backend  # noqa: E402

    # If DaCe backends aren't registered in GT4Py, escalate the warning with a descriptive error message.
    for warning in caught:
        if "GT4Py was unable to load DaCe" in str(warning.message):
            raise RuntimeError(
                "NDSL installation is incomplete. GT4Py was unable to load DaCe. Contact the team."
            )


ndsl_log.info(f"Literal precision: {NDSL_GLOBAL_PRECISION}")
