from pathlib import Path

import dace.sdfg.analysis.schedule_tree.treenodes as stree

from ndsl.dsl.dace.stree.optimizations import AxisIterator, CartesianAxisMerge
from ndsl.logging import ndsl_log_on_rank_0


class StreePipeline:
    def __init__(
        self,
        *,
        passes: list[stree.ScheduleNodeTransformer],
        cache_directory: Path | None = None,
    ) -> None:
        if cache_directory is None:
            cache_directory = Path()

        self.cache_directory = cache_directory
        self.passes = passes

    def __hash__(self) -> int:
        return hash(repr(self))

    def __repr__(self) -> str:
        return str([type(p) for p in self.passes])

    def run(
        self,
        stree: stree.ScheduleTreeRoot,
        verbose: bool = False,
    ) -> stree.ScheduleTreeRoot:
        for i, p in enumerate(self.passes):
            if verbose:
                path = self.cache_directory / f"pass{i}_{p}.txt"
                ndsl_log_on_rank_0.info(f"[Stree OPT] {p} (saving {path} after)")

            p.visit(stree)

            if verbose:
                with open(path, "w+") as f:
                    f.write(stree.as_string())

        return stree


class CPUPipeline(StreePipeline):
    def __init__(
        self,
        *,
        passes: list[stree.ScheduleNodeTransformer] | None = None,
        cache_directory: Path | None = None,
    ) -> None:
        super().__init__(
            passes=(
                passes if passes is not None else [CartesianAxisMerge(AxisIterator._K)]
            ),
            cache_directory=cache_directory,
        )


class GPUPipeline(StreePipeline):
    def __init__(
        self,
        passes: list[stree.ScheduleNodeTransformer] | None = None,
        cache_directory: Path | None = None,
    ) -> None:
        super().__init__(
            passes=passes if passes is not None else [],
            cache_directory=cache_directory,
        )
