from types import TracebackType

import dace

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


class StreeOptimization:
    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = False
