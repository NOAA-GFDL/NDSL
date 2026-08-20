from __future__ import annotations

import copy
import sys
from typing import Any

from dace import SDFG
from dace.frontend.python.parser import DaceProgram

import ndsl.dsl.dace.replacements  # noqa # We load in the DaCe replacements
from ndsl import OptimizationConfig, ndsl_log
from ndsl.comm.mpi import MPI
from ndsl.dsl.dace.builder.cache import get_sdfg_path
from ndsl.dsl.dace.builder.optimize import optimize_full_program_sdfg
from ndsl.dsl.dace.dace_config import (
    DaceConfig,
    DaCeOrchestration,
)
from ndsl.dsl.dace.dace_executable import DACE_EXECUTABLE_CACHE, DaceExecutable
from ndsl.dsl.dace.utils import DaCeProgress


def get_dace_executable(
    dace_program: DaceProgram,
    sdfg: SDFG,
    config: DaceConfig,
    optimization_config: OptimizationConfig | None,
    args: Any,
    kwargs: Any,
) -> DaceExecutable:
    """Get DaceExecutable by either building or fetching it from cache.

    This function is multi-process safe."""

    mode = config.get_orchestrate()
    if (
        mode in [DaCeOrchestration.Build, DaCeOrchestration.BuildAndRun]
        and dace_program not in DACE_EXECUTABLE_CACHE  # already cached
    ):
        parsed_sdfg = copy.copy(sdfg)
        compiled_sdfg = optimize_full_program_sdfg(
            sdfg, config, optimization_config, args, kwargs
        )

        MPI.COMM_WORLD.Barrier()

        if not compiled_sdfg:
            ndsl_log.info(
                f"{DaCeProgress.default_prefix(mode)} Rank is not compiling. "
                "Waiting for compilation to end on all other ranks..."
            )

            with DaCeProgress(mode, "Loading"):
                sdfg_path = get_sdfg_path(
                    dace_program.name, config, override_run_only=True
                )
                if sdfg_path is None:
                    raise ValueError("Couldn't load SDFG post build")
                compiled_sdfg, _ = dace_program.load_precompiled_sdfg(
                    sdfg_path, *args, **kwargs
                )

        MPI.COMM_WORLD.Barrier()

        exe = DaceExecutable.from_compiled(
            dace_program=dace_program,
            config=config,
            compiled_sdfg=compiled_sdfg,
            original_unoptimized_sdfg=parsed_sdfg,
        )
        DACE_EXECUTABLE_CACHE[dace_program] = exe

        # This is a build only, we can now cleanly exit
        if config.get_orchestrate() == DaCeOrchestration.Build:
            ndsl_log.info(f"{DaCeProgress.default_prefix(mode)} Build only, exiting.")
            sys.exit(0)

    if dace_program not in DACE_EXECUTABLE_CACHE:
        raise RuntimeError(
            "Dace program not found in cache. Are you running `DaCeOrchestration.Run` "
            "without a pre-filled cache folder? Try `DacCeOrchestration.BuildAndRun` instead."
        )

    return DACE_EXECUTABLE_CACHE[dace_program]
