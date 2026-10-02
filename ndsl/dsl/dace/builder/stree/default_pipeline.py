from pathlib import Path

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import Backend, OptimizationConfig
from ndsl.dsl.dace.builder.stree.optimizations import (
    CartesianMergePipeline,
    CartesianRefineTransients,
    CleanUpScheduleTree,
    InlineVertical2DWrite,
    KernelizeMaps,
    LocalOptimizations,
)
from ndsl.dsl.dace.builder.stree.pipeline import StreePipeline
from ndsl.dsl.optimization_config import OptimizationOption


class CPUPipeline(StreePipeline):
    def __init__(
        self,
        config: OptimizationConfig,
        backend: Backend,
        *,
        passes: list[tn.ScheduleNodeVisitor] | None = None,
        cache_directory: Path | None = None,
    ) -> None:
        # TODO: checlk that `config` has been concretized

        if passes is None:
            ppl_passes = [CleanUpScheduleTree(), LocalOptimizations(backend)]
            if config.stree.inline_K_loops_size_one:
                ppl_passes.append(InlineVertical2DWrite())
            if config.stree.merger.enabled:
                ppl_passes.append(
                    CartesianMergePipeline(
                        backend,
                        hint=config.hint,
                        align_lhs_on_center_horizontal=config.stree.merger.align_lhs_on_center_horizontal,
                        align_lhs_on_center_vertical=config.stree.merger.align_lhs_on_center_vertical,
                        overcompute_horizontal=config.stree.merger.overcompute_horizontal,
                        overcompute_vertical=config.stree.merger.overcompute_vertical,
                        merge_order=config.stree.merger.order,
                    )
                )
            if config.stree.kernelize == OptimizationOption.APPLY:
                ppl_passes.append(KernelizeMaps(backend))
            if config.stree.refine_transients:
                ppl_passes.append(CartesianRefineTransients(backend))
        else:
            ppl_passes = passes
        super().__init__(
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
            ppl_passes = [CleanUpScheduleTree(), LocalOptimizations(backend)]
            if config.stree.inline_K_loops_size_one:
                ppl_passes.append(InlineVertical2DWrite())
            if config.stree.merger.enabled:
                ppl_passes.append(
                    CartesianMergePipeline(
                        backend,
                        hint=config.hint,
                        overcompute_horizontal=config.stree.merger.overcompute_horizontal,
                        overcompute_vertical=config.stree.merger.overcompute_vertical,
                    )
                )
            if config.stree.kernelize in [
                OptimizationOption.APPLY,
                OptimizationOption.AUTO,
            ]:
                ppl_passes.append(KernelizeMaps(backend))
            if config.stree.refine_transients:
                # TODO
                # 🐞 Transient refine can't be used
                #    because of bugs transients showing in code generation
                # ppl_passes.append(CartesianRefineTransients(backend))
                raise ValueError(
                    "Transient refinement is currently unavailable in the GPU pipeline."
                )
        else:
            ppl_passes = passes
        super().__init__(
            passes=ppl_passes,
            cache_directory=cache_directory,
        )
