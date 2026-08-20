from __future__ import annotations

from typing import Any

from dace import SDFG
from dace.frontend.python.parser import DaceProgram

from ndsl import OptimizationConfig
from ndsl.dsl.dace.builder.labeler import set_label
from ndsl.dsl.dace.dace_config import (
    DaceConfig,
)
from ndsl.dsl.dace.utils import DaCeProgress


def parse_sdfg(
    dace_program: DaceProgram,
    config: DaceConfig,
    optimization: OptimizationConfig | None,
    is_topmost_code: bool,
    *args: Any,
    **kwargs: Any,
) -> SDFG:
    """Parse Dace program into an SDFG.

    Attributes:
        dace_program: the DaceProgram carrying reference to the original method/function
        config: the DaceConfig configuration
        optimization: the OptimizationConfig configuration
        is_topmost_code: internal - use to label the code properly
    """

    with DaCeProgress(
        config.get_orchestrate(), f"Parse code of {dace_program.name} to SDFG"
    ):
        print(f"Parse code of {dace_program.name} to SDFG {args} -- {kwargs}")
        sdfg = dace_program.to_sdfg(
            *args,
            save=False,
            simplify=False,
            validate=False,  # TODO: should we have a "debug flag" to turn this on?
            **dace_program.__sdfg_closure__(),
            **kwargs,
        )

    # Label the code (this is the topmost code)
    if sdfg is not None and optimization is not None and optimization.stree.enabled:
        set_label(
            sdfg,
            dace_program.f.__qualname__,
            is_top_sdfg=is_topmost_code,
            local_optimizations=optimization,
        )

    return sdfg
