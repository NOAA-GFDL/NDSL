from __future__ import annotations

import os
from typing import Any

from dace import SDFG, CompiledSDFG
from dace.frontend.python.parser import DaceProgram

import ndsl.dsl.dace.replacements  # noqa # We load in the DaCe replacements
from ndsl import OptimizationConfig, ndsl_log
from ndsl.dsl.dace.builder.cache import get_sdfg_path
from ndsl.dsl.dace.dace_config import (
    DEACTIVATE_DISTRIBUTED_DACE_COMPILE,
    DaceConfig,
)
from ndsl.dsl.dace.dace_executable import DACE_EXECUTABLE_CACHE, DaceExecutable
from ndsl.dsl.dace.labeler import set_label
from ndsl.dsl.dace.utils import DaCeProgress


def parse_sdfg(
    dace_program: DaceProgram,
    config: DaceConfig,
    optimization: OptimizationConfig | None,
    *args: Any,
    **kwargs: Any,
) -> SDFG | CompiledSDFG | None:
    """Return an SDFG depending on cache existence.
    Either parses, load a .sdfg or load .so (as a compiled sdfg)

    Attributes:
        dace_program: the DaceProgram carrying reference to the original method/function
        config: the DaceConfig configuration for this execution
    """

    # Check cache for already loaded SDFG
    if dace_program in DACE_EXECUTABLE_CACHE:
        return DACE_EXECUTABLE_CACHE[dace_program].compiled_sdfg

    ndsl_log.info(f"Building DaCe orchestration for {dace_program.f.__qualname__}")
    mode = config.get_orchestrate()

    # Build expected path
    sdfg_path = get_sdfg_path(dace_program.name, config)
    if sdfg_path is None:
        is_compiling = (
            True if DEACTIVATE_DISTRIBUTED_DACE_COMPILE else config.do_compile
        )

        if not is_compiling:
            # We can not parse the SDFG since we will load the proper
            # compiled SDFG from the compiling rank
            return None

        with DaCeProgress(mode, f"Parse code of {dace_program.name} to SDFG"):
            sdfg = dace_program.to_sdfg(
                *args,
                **dace_program.__sdfg_closure__(),
                **kwargs,
                save=False,
                simplify=False,
                validate=False,  # TODO: should we have a "debug flag" to turn this on?
            )

        # Label the code (this is the topmost code)
        if sdfg is not None and optimization is not None and optimization.stree.enabled:
            set_label(
                sdfg,
                dace_program.f.__qualname__,
                is_top_sdfg=True,
                local_optimizations=optimization,
            )

        return sdfg

    if os.path.isfile(sdfg_path):
        with DaCeProgress(mode, "Load .sdfg"):
            sdfg, _ = dace_program.load_sdfg(sdfg_path, *args, **kwargs)
        return sdfg

    with DaCeProgress(mode, "Load precompiled .sdfg (.so)"):
        compiled_sdfg, _ = dace_program.load_precompiled_sdfg(
            sdfg_path, *args, **kwargs
        )
        DACE_EXECUTABLE_CACHE[dace_program] = DaceExecutable.from_compiled(
            dace_program=dace_program,
            config=config,
            compiled_sdfg=compiled_sdfg,
        )

    return compiled_sdfg
