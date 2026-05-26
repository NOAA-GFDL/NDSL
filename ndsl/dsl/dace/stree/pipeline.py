from pathlib import Path

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import Backend
from ndsl.dsl.dace.stree.optimizations import (
    CartesianMerge,
    CartesianRefineTransients,
    CleanUpScheduleTree,
)
from ndsl.dsl.dace.stree.optimizations.statistics import TreeOptimizationStatistics
from ndsl.logging import ndsl_log_on_rank_0


class StreePipeline:
    def __init__(
        self,
        *,
        passes: list[tn.ScheduleNodeVisitor],
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
        stree: tn.ScheduleTreeRoot,
        verbose: bool = False,
    ) -> tn.ScheduleTreeRoot:
        tree_stats = TreeOptimizationStatistics()
        tree_stats.original(stree)

        for i, p in enumerate(self.passes):
            if verbose:
                path = self.cache_directory / f"pass{i}_{p}.txt"
                ndsl_log_on_rank_0.info(f"[Stree OPT] {p} (saving {path} after)")

            p.visit(stree)

            if verbose:
                with open(path, "w+") as f:
                    f.write(stree.as_string())

        tree_stats.optimized(stree)

        if verbose:
            ndsl_log_on_rank_0.info(tree_stats.report())

        return stree


class CPUPipeline(StreePipeline):
    def __init__(
        self,
        backend: Backend,
        *,
        passes: list[tn.ScheduleNodeVisitor] | None = None,
        cache_directory: Path | None = None,
    ) -> None:
        if passes is None:
            passes = [
                CleanUpScheduleTree(),
                # TODO: Is it safe? Deactivate for now
                # InlineVertical2DWrite(),
                CartesianMerge(backend),
                CartesianRefineTransients(backend),
            ]
        super().__init__(
            passes=passes,
            cache_directory=cache_directory,
        )


class GPUPipeline(StreePipeline):
    def __init__(
        self,
        backend: Backend,
        *,
        passes: list[tn.ScheduleNodeVisitor] | None = None,
        cache_directory: Path | None = None,
    ) -> None:
        if passes is None:
            passes = [
                CleanUpScheduleTree(),
                # TODO: Is it safe? Deactivate for now
                # InlineVertical2DWrite(),
                CartesianMerge(backend),
                # 🐞 Transient refine can't be used
                #    because of bugs transients showing in code generation
                # CartesianRefineTransients(backend),
            ]
        super().__init__(
            passes=passes,
            cache_directory=cache_directory,
        )
        super().__init__(
            passes=passes if passes is not None else [],
            cache_directory=cache_directory,
        )
