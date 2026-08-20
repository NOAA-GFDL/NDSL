from ndsl import ndsl_log
from ndsl.config import Backend
from ndsl.dsl.caches.cache_location import get_cache_fullpath
from ndsl.dsl.dace.dace_config import DaceConfig, DaCeOrchestration

DACE_BUILD_INFO_FILENAME = "build_info.txt"
"""File dropped alongside .dacecache and keeping track of NDSL configuration nformation
that led to this build."""


def get_sdfg_path(
    daceprog_name: str,
    config: DaceConfig,
    sdfg_file_path: str | None = None,
    override_run_only: bool = False,
) -> str | None:
    """Utility to make an SDFG path from the qualified program name or it's direct path to .sdfg

    Args:
        daceprog_name: qualified name in the form module_qualname if module is not locals
        sdfg_file_path: absolute path to a .sdfg file
    """
    import os

    # TODO: check DaceConfig for cache.strategy == name
    # Guarding against bad usage of this function
    if not override_run_only and config.get_orchestrate() != DaCeOrchestration.Run:
        return None

    # Case of a .sdfg file given by the user to be compiled
    if sdfg_file_path is not None:
        if not os.path.isfile(sdfg_file_path):
            raise RuntimeError(
                f"SDFG filepath {sdfg_file_path} cannot be found or is not a file"
            )
        return sdfg_file_path

    # Case of loading a precompiled .so - lookup using GT_CACHE
    cache_fullpath = get_cache_fullpath(config.code_path)
    sdfg_dir_path = f"{cache_fullpath}/dacecache/{daceprog_name}"
    if not os.path.isdir(sdfg_dir_path):
        raise RuntimeError(f"Precompiled SDFG is missing at {sdfg_dir_path}")

    # Check layout in build time matches layout now
    import ast

    with open(f"{sdfg_dir_path}/{DACE_BUILD_INFO_FILENAME}") as build_info_file:
        # Jump over schema comment
        build_info_file.readline()
        # Read in
        build_backend = build_info_file.readline().rstrip()
        if config.get_backend() != Backend(build_backend):
            raise RuntimeError(
                f"SDFG build for {build_backend}, {config._backend} has been asked"
            )
        # Check resolution per tile
        build_layout = ast.literal_eval(build_info_file.readline())
        build_resolution = ast.literal_eval(build_info_file.readline())
        if (config.tile_resolution[0] / config.layout[0]) != (
            build_resolution[0] / build_layout[0]
        ):
            raise RuntimeError(
                f"SDFG build for resolution {build_resolution}, "
                f"cannot be run with current resolution {config.tile_resolution}"
            )

    ndsl_log.debug(f"[DaCe Config] Rank {config.my_rank} loading SDFG {sdfg_dir_path}")

    return sdfg_dir_path
