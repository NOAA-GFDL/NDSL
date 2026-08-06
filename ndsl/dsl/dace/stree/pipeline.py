from collections.abc import Sequence
from pathlib import Path

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log_on_rank_0
from ndsl.dsl.dace.stree.statistics import TreeOptimizationStatistics


class StreePipeline:
    def __init__(
        self,
        *,
        passes: Sequence[
            "tn.ScheduleNodeVisitor | tn.ScheduleNodeTransformer | StreePipeline"
        ],
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
        *,
        nesting: int = 0,
        cache_directory: Path | None = None,
    ) -> tn.ScheduleTreeScope:
        # Re-entry for nested pipeline
        if cache_directory is None:
            cache_directory = self.cache_directory

        if nesting == 0:
            tree_stats = TreeOptimizationStatistics()
            tree_stats.original(stree)

        for i, p in enumerate(self.passes):
            path: Path | None = None
            if verbose:
                path = cache_directory / f"pass_n{nesting}_{i}_{p}.txt"
                ndsl_log_on_rank_0.info(f"[Stree OPT] {p} (saving {path} after)")

            if isinstance(p, tn.ScheduleNodeVisitor):
                p.visit(stree)
            elif isinstance(p, StreePipeline):
                p.run(
                    stree, verbose, nesting=nesting + 1, cache_directory=cache_directory
                )

            if verbose:
                assert path is not None
                with open(path, "w+") as f:
                    f.write(stree.as_string())

        if nesting == 0:
            tree_stats.optimized(stree)
            ndsl_log_on_rank_0.info(tree_stats.report())

        return stree
