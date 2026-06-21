from pathlib import Path

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import Backend, OptimizationConfig, ndsl_log_on_rank_0
from ndsl.dsl.dace.stree.optimizations import (
    CartesianMerge,
    CartesianRefineTransients,
    CleanUpScheduleTree,
    KernelizeMaps,
    TreeOptimizationStatistics,
)


class StreePipeline:
    def __init__(
        self,
        config: OptimizationConfig,
        *,
        passes: list[tn.ScheduleNodeVisitor],
        cache_directory: Path | None = None,
    ) -> None:
        if cache_directory is None:
            cache_directory = Path()

        self.cache_directory = cache_directory
        self.passes = passes
        self.config = config

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
            path: Path | None = None
            if verbose:
                path = self.cache_directory / f"pass{i}_{p}.txt"
                ndsl_log_on_rank_0.info(f"[Stree OPT] {p} (saving {path} after)")

            p.visit(stree)

            if verbose:
                assert path is not None
                with open(path, "w+") as f:
                    f.write(stree.as_string())

        tree_stats.optimized(stree)
        ndsl_log_on_rank_0.info(tree_stats.report())
        return stree


class CPUPipeline(StreePipeline):
    def __init__(
        self,
        config: OptimizationConfig,
        backend: Backend,
        *,
        passes: list[tn.ScheduleNodeVisitor] | None = None,
        cache_directory: Path | None = None,
    ) -> None:
        if passes is None:
            ppl_passes = [
                CleanUpScheduleTree(),
                # TODO: Is it safe? Deactivate for now
                # InlineVertical2DWrite(),
                CartesianMerge(backend, overcompute=config.stree.merger.overcompute),
                CartesianRefineTransients(backend),
            ]
        else:
            ppl_passes = passes
        super().__init__(
            config=config,
            passes=ppl_passes,
            cache_directory=cache_directory,
        )


class GPUPipeline(StreePipeline):
    def __init__(
        self,
        config: OptimizationConfig,
        backend: Backend,
        *,
        passes: list[tn.ScheduleNodeVisitor] | None = None,
        cache_directory: Path | None = None,
    ) -> None:
        if passes is None:
            ppl_passes = [CleanUpScheduleTree()]
            # TODO: Is it safe? Deactivate for now
            # ppl_passes.append(InlineVertical2DWrite())
            if config.stree.merger.enabled:
                ppl_passes.append(
                    CartesianMerge(backend, overcompute=config.stree.merger.overcompute)
                )
            if config.stree.kernalize:
                ppl_passes.append(KernelizeMaps(backend))
            # 🐞 Transient refine can't be used
            #    because of bugs transients showing in code generation
            # CartesianRefineTransients(backend),
        else:
            ppl_passes = passes
        super().__init__(
            config=config,
            passes=ppl_passes,
            cache_directory=cache_directory,
        )
