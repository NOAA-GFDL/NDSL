from types import TracebackType

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn

import ndsl.dsl.dace.orchestration as orch
from ndsl import StencilFactory


def get_SDFG_and_purge(stencil_factory: StencilFactory) -> dace.CompiledSDFG:
    """Get the Precompiled SDFG from the dace config dict where they are cached post
    compilation and flush the cache in order for next build to re-use the function."""
    sdfg_repo = stencil_factory.config.dace_config.loaded_dace_executables

    if len(sdfg_repo.values()) != 1:
        raise RuntimeError("Failure to compile SDFG")
    sdfg = list(sdfg_repo.values())[0].compiled_sdfg

    sdfg_repo.clear()

    return sdfg


class StreePipeline:
    def __init__(self, *, passes: list[tn.ScheduleNodeVisitor] | None = None) -> None:
        self.passes = passes

    def __enter__(self) -> None:
        self.original_passes = orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION_PASSES
        orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION_PASSES = self.passes

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION_PASSES = self.original_passes
