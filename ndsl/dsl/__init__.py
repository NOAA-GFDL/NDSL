from ndsl.comm.mpi import MPI

# Literal precision for both GT4Py & NDSL
import os
import sys

gt4py_config_module = "gt4py.cartesian.config"
if gt4py_config_module in sys.modules:
    raise RuntimeError(
        "`GT4Py` config imported before `ndsl` imported."
        " Please import `ndsl.dsl` or any `ndsl` module "
        " before any `gt4py` imports."
    )
NDSL_GLOBAL_PRECISION = int(os.getenv("PACE_FLOAT_PRECISION", "64"))
os.environ["GT4PY_LITERAL_PRECISION"] = str(NDSL_GLOBAL_PRECISION)


# Set cache names for default gt backends workflow
import gt4py.cartesian.config  # noqa: E402

from ndsl.comm.mpi import MPI  # noqa: E402


if MPI is not None:
    import os

    gt4py.cartesian.config.cache_settings["dir_name"] = os.environ.get(
        "GT_CACHE_DIR_NAME", f".gt_cache_{MPI.COMM_WORLD.Get_rank():06}"
    )
