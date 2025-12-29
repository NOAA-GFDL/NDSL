from __future__ import annotations

import enum
import os
from typing import TYPE_CHECKING, Any, Self

import dace.config
from gt4py.cartesian.config import GT4PY_COMPILE_OPT_LEVEL

from ndsl import LocalComm
from ndsl.comm.communicator import Communicator
from ndsl.comm.partitioner import Partitioner
from ndsl.dsl.caches.cache_location import identify_code_path
from ndsl.dsl.caches.codepath import FV3CodePath
from ndsl.dsl.gt4py_utils import is_gpu_backend
from ndsl.dsl.typing import get_precision
from ndsl.optional_imports import cupy as cp
from ndsl.performance.collector import NullPerformanceCollector, PerformanceCollector


if TYPE_CHECKING:
    from ndsl.dsl.dace.dace_executable import DaceExecutables

# This can be turned on to revert compilation for orchestration
# in a rank-compile-itself more, instead of the distributed top-tile
# mechanism.
DEACTIVATE_DISTRIBUTED_DACE_COMPILE = False


def _debug_dace_orchestration() -> bool:
    """
    Debugging Dace orchestration deeper can be done by turning on `syncdebug`.
    We control this Dace configuration below with our own override.
    """
    return os.getenv("NDSL_DACE_DEBUG", "False") == "True"


def _is_corner(rank: int, partitioner: Partitioner) -> bool:
    if partitioner.tile.on_tile_bottom(rank):
        if partitioner.tile.on_tile_left(rank):
            return True
        if partitioner.tile.on_tile_right(rank):
            return True
    if partitioner.tile.on_tile_top(rank):
        if partitioner.tile.on_tile_left(rank):
            return True
        if partitioner.tile.on_tile_right(rank):
            return True
    return False


def _smallest_rank_bottom(x: int, y: int, layout: tuple[int, int]) -> bool:
    return y == 0 and x == 1


def _smallest_rank_top(x: int, y: int, layout: tuple[int, int]) -> bool:
    return y == layout[1] - 1 and x == 1


def _smallest_rank_left(x: int, y: int, layout: tuple[int, int]) -> bool:
    return x == 0 and y == 1


def _smallest_rank_right(x: int, y: int, layout: tuple[int, int]) -> bool:
    return x == layout[0] - 1 and y == 1


def _smallest_rank_middle(x: int, y: int, layout: tuple[int, int]) -> bool:
    return layout[0] > 1 and layout[1] > 1 and x == 1 and y == 1


def _determine_compiling_ranks(
    config: DaceConfig,
    partitioner: Partitioner,
) -> bool:
    """
    We try to map every layout to a 3x3 layout which MPI ranks
    looks like
        6 7 8
        3 4 5
        0 1 2
    Using the partitioner we find mapping of the given layout
    to all of those. For example on 4x4 layout
        12 13 14 15
        8  9  10 11
        4  5  6  7
        0  1  2  3
    therefore we map
        0 -> 0
        1 -> 1
        2 -> NOT COMPILING
        3 -> 2
        4 -> 3
        5 -> 4
        6 -> NOT COMPILING
        7 -> 5
        8 -> NOT COMPILING
        9 -> NOT COMPILING
        10 -> NOT COMPILING
        11 -> NOT COMPILING
        12 -> 6
        13 -> 7
        14 -> NOT COMPILING
        15 -> 8
    """

    if config._single_code_path:
        return config.my_rank == 0

    # Tile 0 compiles
    if partitioner.tile_index(config.my_rank) != 0:
        return False

    # Corners compile
    if _is_corner(config.my_rank, partitioner):
        return True

    y, x = partitioner.tile.subtile_index(config.my_rank)

    # If edge or center tile, we give way to the smallest rank
    return (
        _smallest_rank_left(x, y, config.layout)
        or _smallest_rank_bottom(x, y, config.layout)
        or _smallest_rank_middle(x, y, config.layout)
        or _smallest_rank_right(x, y, config.layout)
        or _smallest_rank_top(x, y, config.layout)
    )


class DaCeOrchestration(enum.Enum):
    """
    Orchestration mode for DaCe

        Python: python orchestration
        Build: compile & save SDFG only
        BuildAndRun: compile & save SDFG, then run
        Run: load from .so and run, will fail if .so is not available
    """

    Python = 0
    Build = 1
    BuildAndRun = 2
    Run = 3


class DaceConfig:
    def __init__(
        self,
        communicator: Communicator | None,
        backend: str,
        tile_nx: int = 0,
        tile_nz: int = 0,
        orchestration: DaCeOrchestration | None = None,
        time: bool = False,
        single_code_path: bool = False,
    ):
        """Specialize the DaCe configuration for NDSL use.

        Dev note: This class wrongly carries two runtime values:
            - `loaded_dace_executables`: cache of loaded SDFG & cached arguments
            - `performance_collector`: runtime timer shared for all runtime call
                of orchestrate code

        Args:
            communicator: used for setting the distributed caches
            backend: string for the backend
            tile_nx: x/y domain size for a single time
            tile_nz: z domain size for a single time
            orchestration: orchestration mode from DaCeOrchestration
            time: trigger performance collection, available to user with
                `performance_collector`
            single_codepath: code is expected to be the same on every rank (case
                of column-physics) and therefore can be compiled once
        """

        self._single_code_path = single_code_path
        # Recording SDFG loaded for fast re-access
        # ToDo: DaceConfig becomes a bit more than a read-only config
        #       with this. Should be refactored into a DaceExecutor carrying a config
        self.loaded_dace_executables: DaceExecutables = {}
        self.performance_collector = (
            PerformanceCollector(
                "InternalOrchestrationTimer",
                comm=(
                    LocalComm(0, 6, {}) if communicator is None else communicator.comm
                ),
            )
            if time
            else NullPerformanceCollector()
        )

        # Temporary. This is a bit too out of the ordinary for the common user.
        # We should refactor the architecture to allow for a `gtc:orchestrated:dace:X`
        # backend that would signify both the `CPU|GPU` split and the orchestration mode
        if orchestration is None:
            fv3_dacemode_env_var = os.getenv("FV3_DACEMODE", "Python")
            # The below condition guards against defining empty FV3_DACEMODE and
            # awkward behavior of os.getenv returning "" even when not defined
            if fv3_dacemode_env_var is None or fv3_dacemode_env_var == "":
                fv3_dacemode_env_var = "Python"
            self._orchestrate = DaCeOrchestration[fv3_dacemode_env_var]
        else:
            self._orchestrate = orchestration

        # We hijack the optimization level of GT4Py because we don't
        # have the configuration at NDSL level, but we do use the GT4Py
        # level
        # TODO: if GT4PY opt level is funneled via NDSL - use it here
        optimization_level = int(GT4PY_COMPILE_OPT_LEVEL)

        # Set the configuration of DaCe to a rigid & tested set of divergence
        # from the defaults when orchestrating
        if self.is_dace_orchestrated():
            # Detecting neoverse-v1/2 requires an external package, we swap it
            # for a read on GH200 nodes themselves.
            is_arm_neoverse = (
                cp is not None
                and cp.cuda.runtime.getDeviceProperties(0)["name"]
                == b"NVIDIA GH200 480GB"
            )

            if optimization_level == 0:
                dace.config.Config.set("compiler", "build_type", value="Debug")
            elif optimization_level == 2 or optimization_level == 1:
                dace.config.Config.set("compiler", "build_type", value="RelWithDebInfo")
            else:
                dace.config.Config.set("compiler", "build_type", value="Release")

            # Required to True for gt4py storage/memory
            dace.config.Config.set(
                "compiler",
                "allow_view_arguments",
                value=True,
            )
            # Resolve "march/mtune" option for GPU
            # - turn on numeric-centric SSE by default
            # - Neoverse-V2 Grace CPU is too new for GCC 14 and -march=native will fail
            # - use alternative march=armv8-a instead
            march_cpu = "armv8-a" if is_arm_neoverse else "native"
            # Removed --fmath
            dace.config.Config.set(
                "compiler",
                "cpu",
                "args",
                value=f"-march={march_cpu} -std=c++17 -fPIC -Wall -Wextra -O{optimization_level}",
            )
            # Potentially buggy - deactivate
            dace.config.Config.set(
                "compiler",
                "cpu",
                "openmp_sections",
                value=0,
            )
            # Resolve "march/mtune" option for GPU
            # - turn on numeric-centric SSE by default
            # - Neoverse-V2 Grace CPU will fail
            # - use alternative mcpu=native instead
            march_option = "-mcpu=native" if is_arm_neoverse else "-march=native"
            # Removed --fast-math
            dace.config.Config.set(
                "compiler",
                "cuda",
                "args",
                value=f"-std=c++14 -Xcompiler -fPIC -O3 -Xcompiler {march_option}",
            )

            cuda_sm = cp.cuda.Device(0).compute_capability if cp else 60
            dace.config.Config.set("compiler", "cuda", "cuda_arch", value=f"{cuda_sm}")
            # Block size/thread count is defaulted to an average value for recent
            # hardware (Pascal and upward). The problem of setting an optimized
            # block/thread is both hardware and problem dependant. Fine tuners
            # available in DaCe should be relied on for further tuning of this value.
            dace.config.Config.set(
                "compiler", "cuda", "default_block_size", value="64,8,1"
            )
            # Potentially buggy - deactivate
            dace.config.Config.set(
                "compiler",
                "cuda",
                "max_concurrent_streams",
                value=-1,  # no concurrent streams, every kernel on defaultStream
            )
            # Speed up built time
            dace.config.Config.set(
                "compiler",
                "cuda",
                "unique_functions",
                value="none",
            )
            # Required for HaloEx callbacks and general code sanity
            dace.config.Config.set(
                "frontend",
                "dont_fuse_callbacks",
                value=True,
            )
            # Unroll all loop - outer loop should be exempted with dace.nounroll
            dace.config.Config.set(
                "frontend",
                "unroll_threshold",
                value=False,
            )
            # Allow for a longer stack dump when parsing fails
            dace.config.Config.set(
                "frontend",
                "verbose_errors",
                value=True,
            )
            # Build speed up by removing some deep copies
            dace.config.Config.set(
                "store_history",
                value=False,
            )

            # Enable to debug GPU failures
            dace.config.Config.set(
                "compiler", "cuda", "syncdebug", value=_debug_dace_orchestration()
            )

            if get_precision() == 32:
                # When using 32-bit float, we flip the default dtypes to be all
                # C, e.g. 32 bit.
                dace.Config.set(
                    "compiler",
                    "default_data_types",
                    value="c",
                )

        # Attempt to kill the dace.conf to avoid confusion
        if dace.config.Config._cfg_filename:
            try:
                os.remove(dace.config.Config._cfg_filename)
            except OSError:
                pass

        self._backend = backend
        self.tile_resolution = [tile_nx, tile_nx, tile_nz]
        from ndsl.dsl.dace.build import set_distributed_caches

        # Distributed build required info
        if communicator:
            self.my_rank = communicator.rank
            self.rank_size = communicator.comm.Get_size()
            self.code_path = identify_code_path(
                self.my_rank,
                communicator.partitioner,
                self._single_code_path,
            )
            self.layout = communicator.partitioner.layout
            self.do_compile = (
                DEACTIVATE_DISTRIBUTED_DACE_COMPILE
                or _determine_compiling_ranks(self, communicator.partitioner)
            )
        else:
            self.my_rank = 0
            self.rank_size = 1
            self.code_path = FV3CodePath.All
            self.layout = (1, 1)
            self.do_compile = True

        set_distributed_caches(self)

        if self.is_dace_orchestrated() and "dace" not in self._backend:
            raise RuntimeError(
                "DaceConfig: orchestration can only be leveraged "
                f"with the `dace:*` backends, not with {self._backend}."
            )

    def is_dace_orchestrated(self) -> bool:
        return self._orchestrate != DaCeOrchestration.Python

    def is_gpu_backend(self) -> bool:
        return is_gpu_backend(self._backend)

    def get_backend(self) -> str:
        return self._backend

    def get_orchestrate(self) -> DaCeOrchestration:
        return self._orchestrate

    def get_sync_debug(self) -> bool:
        return dace.config.Config.get_bool("compiler", "cuda", "syncdebug")

    def as_dict(self) -> dict[str, Any]:
        return {
            "_orchestrate": str(self._orchestrate.name),
            "_backend": self._backend,
            "my_rank": self.my_rank,
            "rank_size": self.rank_size,
            "layout": self.layout,
            "tile_resolution": self.tile_resolution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        config = cls(
            None,
            backend=data["_backend"],
            orchestration=DaCeOrchestration[data["_orchestrate"]],
        )
        config.my_rank = data["my_rank"]
        config.rank_size = data["rank_size"]
        config.layout = data["layout"]
        config.tile_resolution = data["tile_resolution"]
        return config
