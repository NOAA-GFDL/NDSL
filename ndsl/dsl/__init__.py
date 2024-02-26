import gt4py.cartesian.config

from ndsl.comm.mpi import MPI

from . import dace
from .dace import DaceConfig, DaCeOrchestration, orchestrate, orchestrate_function
from .stencil import CompareToNumpyStencil, FrozenStencil, GridIndexing, StencilFactory
from .stencil_config import CompilationConfig, RunMode, StencilConfig


if MPI is not None:
    import os

    gt4py.cartesian.config.cache_settings["dir_name"] = os.environ.get(
        "GT_CACHE_DIR_NAME", f".gt_cache_{MPI.COMM_WORLD.Get_rank():06}"
    )

__version__ = "0.2.0"
