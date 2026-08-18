from __future__ import annotations

from typing import Any

from dace import SDFG
from dace.frontend.python.parser import DaceProgram

import ndsl.dsl.dace.replacements  # noqa # We load in the DaCe replacements
from ndsl import OptimizationConfig
from ndsl.dsl.dace.builder.optimize import optimize_full_program_sdfg
from ndsl.dsl.dace.dace_config import (
    DaceConfig,
    DaCeOrchestration,
)
from ndsl.dsl.dace.dace_executable import DACE_EXECUTABLE_CACHE, DaceExecutable


def get_dace_executable(
    dace_program: DaceProgram,
    sdfg: SDFG,
    config: DaceConfig,
    optimization_config: OptimizationConfig | None,
    args: Any,
    kwargs: Any,
) -> DaceExecutable:
    """Get DaceExecutable by either building or fetching it from cache."""

    mode = config.get_orchestrate()
    if (
        mode in [DaCeOrchestration.Build, DaCeOrchestration.BuildAndRun]
        and dace_program not in DACE_EXECUTABLE_CACHE  # already cached
    ):
        optimize_full_program_sdfg(
            dace_program, sdfg, config, optimization_config, args, kwargs
        )

    if dace_program not in DACE_EXECUTABLE_CACHE:
        raise RuntimeError(
            "Dace program not found in cache. Are you running `DaCeOrchestration.Run` "
            "without a pre-filled cache folder? Try `DacCeOrchestration.BuildAndRun` instead."
        )

    return DACE_EXECUTABLE_CACHE[dace_program]
