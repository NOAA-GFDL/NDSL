import gt4py.cartesian.config

from ndsl.comm.mpi import MPI


if MPI is not None:
    import os

    gt4py.cartesian.config.cache_settings["dir_name"] = os.environ.get(
        "GT_CACHE_DIR_NAME", f".gt_cache_{MPI.COMM_WORLD.Get_rank():06}"
    )
