import copy
import sys
from typing import Any

from dace.frontend.python.parser import DaceProgram

from ndsl import OptimizationConfig, ndsl_log
from ndsl.comm.mpi import MPI
from ndsl.dsl.dace.builder.cache import get_sdfg_path_from_cache
from ndsl.dsl.dace.builder.optimize import optimize_full_program_sdfg
from ndsl.dsl.dace.builder.parse import parse_sdfg
from ndsl.dsl.dace.dace_config import (
    DaceConfig,
    DaCeOrchestration,
)
from ndsl.dsl.dace.dace_executable import DACE_EXECUTABLE_CACHE, DaceExecutable
from ndsl.dsl.dace.utils import DaCeProgress


def get_dace_executable(
    dace_program: DaceProgram,
    config: DaceConfig,
    optimization_config: OptimizationConfig | None,
    args: Any,
    kwargs: Any,
) -> DaceExecutable:
    """Get DaceExecutable by either building or fetching it from cache.

    This function is multi-process safe."""

    if not config.get_backend().is_orchestrated():
        raise RuntimeError(
            f"Cannot fetch full program executable from non-orchestrated ({config.get_backend()}) backend."
        )

    mode = config.get_orchestrate()
    is_compiling = config.is_compiling()

    if mode == DaCeOrchestration.Run:
        with DaCeProgress(
            config.get_orchestrate(), f"Load {dace_program.name} executable..."
        ):
            sdfg_path = get_sdfg_path_from_cache(dace_program.name, config)
            compiled_sdfg, _ = dace_program.load_precompiled_sdfg(
                str(sdfg_path), *args, **kwargs
            )
        DACE_EXECUTABLE_CACHE[dace_program] = DaceExecutable.from_compiled(
            dace_program=dace_program,
            config=config,
            compiled_sdfg=compiled_sdfg,
            original_unoptimized_sdfg=None,
        )
    elif (
        mode in [DaCeOrchestration.Build, DaCeOrchestration.BuildAndRun]
        and dace_program not in DACE_EXECUTABLE_CACHE  # already cached
    ):
        if is_compiling:
            with DaCeProgress(
                config.get_orchestrate(), f"Make {dace_program.name} executable..."
            ):
                parsed_sdfg = parse_sdfg(
                    dace_program,
                    config,
                    optimization_config,
                    True,  # top most code, since it's the call fetching the executable
                    *args,
                    **kwargs,
                )
                original_sdfg = copy.copy(parsed_sdfg)
                compiled_sdfg = optimize_full_program_sdfg(
                    parsed_sdfg, config, optimization_config, args, kwargs
                )
        else:
            original_sdfg = None

        if not is_compiling:
            ndsl_log.info(
                f"{DaCeProgress.default_prefix(mode)} Rank is not compiling. "
                "Waiting for compilation to end on all other ranks..."
            )

        MPI.COMM_WORLD.Barrier()

        if not is_compiling:
            with DaCeProgress(
                config.get_orchestrate(), f"Load {dace_program.name} executable..."
            ):
                sdfg_path = get_sdfg_path_from_cache(dace_program.name, config)
                compiled_sdfg, _ = dace_program.load_precompiled_sdfg(
                    str(sdfg_path), *args, **kwargs
                )

        MPI.COMM_WORLD.Barrier()

        exe = DaceExecutable.from_compiled(
            dace_program=dace_program,
            config=config,
            compiled_sdfg=compiled_sdfg,
            original_unoptimized_sdfg=original_sdfg,
        )
        DACE_EXECUTABLE_CACHE[dace_program] = exe

        # This is a build only, we can now cleanly exit
        if mode == DaCeOrchestration.Build:
            ndsl_log.info(f"{DaCeProgress.default_prefix(mode)} Build only, exiting.")
            sys.exit(0)

    if dace_program not in DACE_EXECUTABLE_CACHE:
        raise RuntimeError("Dace program not found in cache. Abort.")

    return DACE_EXECUTABLE_CACHE[dace_program]
