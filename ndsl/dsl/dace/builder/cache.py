import os
from pathlib import Path

from dace.sdfg.sdfg import SDFG

from ndsl import ndsl_log
from ndsl.config import Backend
from ndsl.dsl.caches.cache_location import get_cache_fullpath
from ndsl.dsl.dace.dace_config import DaceConfig


class BuildInfo:
    """Drop build and memory info inside the `.dacecache`,
    which keeps track of NDSL configuration information that led to this build."""

    _BUILD_FILENAME: Path = Path("build_info.txt")
    _MEMORY_FILENAME: Path = Path("memory_report.txt")

    @classmethod
    def save(
        cls,
        sdfg: SDFG,
        layout: tuple[int, int],
        resolution_per_tile: list[int],
        memory_report: str,
        backend: Backend,
    ) -> None:
        """Write down all relevant information on the build to identify
        it at load time."""
        # Dev NOTE: we should be able to leverage sdfg.make_key to get a hash or
        # even go to a complete hash base system and read the data from the SDFG itself

        path_to_sdfg_dir = Path(os.path.abspath(sdfg.build_folder))
        with open(path_to_sdfg_dir / cls._BUILD_FILENAME, "w") as build_info_read:
            build_info_read.write("#Schema: Backend Layout Resolution per tile\n")
            build_info_read.write(f"{backend}\n")
            build_info_read.write(f"{layout}\n")
            build_info_read.write(f"{resolution_per_tile}\n")

        with open(path_to_sdfg_dir / cls._MEMORY_FILENAME, "w") as f:
            f.write(memory_report)

    @classmethod
    def check(cls, config: DaceConfig, cache_directory: Path) -> None:
        # Check layout in build time matches layout now
        import ast

        with open(cache_directory / cls._BUILD_FILENAME) as build_info_file:
            # Jump over schema comment
            build_info_file.readline()

            # Read in
            backend = build_info_file.readline().rstrip()
            if config.get_backend() != Backend(backend):
                raise RuntimeError(
                    f"SDFG build for {backend}, {config._backend} has been asked"
                )

            # Check resolution per tile
            layout = ast.literal_eval(build_info_file.readline())
            resolution = ast.literal_eval(build_info_file.readline())
            if (config.tile_resolution[0] / config.layout[0]) != (
                resolution[0] / layout[0]
            ):
                raise RuntimeError(
                    f"SDFG build for resolution {resolution}, "
                    f"cannot be run with current resolution {config.tile_resolution}"
                )


def get_sdfg_path_from_cache(daceprog_name: str, config: DaceConfig) -> Path:
    """Utility to get an SDFG path from the qualified program name

    Args:
        daceprog_name: qualified name in the form module_qualname if module is not locals
    """

    # Case of loading a precompiled .so - lookup using GT_CACHE
    cache_fullpath = get_cache_fullpath(config.code_path)
    sdfg_dir_path = Path(f"{cache_fullpath}/dacecache/{daceprog_name}")
    if not sdfg_dir_path.is_dir():
        raise RuntimeError(
            f"Precompiled SDFG is missing at {sdfg_dir_path}.\n"
            "Are you running `DaCeOrchestration.Run` without a pre-built cache folder? "
            "Try `DacCeOrchestration.BuildAndRun` instead."
        )

    BuildInfo.check(config, sdfg_dir_path)

    ndsl_log.debug(f"[DaCe Config] Rank {config.my_rank} loading SDFG {sdfg_dir_path}")

    return sdfg_dir_path
