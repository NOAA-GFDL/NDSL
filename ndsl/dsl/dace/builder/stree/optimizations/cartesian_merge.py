from ndsl.config import Backend
from ndsl.dsl.dace.builder.stree.common import AxisIterator
from ndsl.dsl.dace.builder.stree.common.memlet import (
    HORIZONTAL_AXIS_SYMBOLS,
    VERTICAL_AXIS_SYMBOLS,
    axis_from_backend,
)
from ndsl.dsl.dace.builder.stree.optimizations.axis_merge import CartesianAxisMerge
from ndsl.dsl.dace.builder.stree.optimizations.lhs_write_on_center import (
    LHSWriteOnCenter,
)
from ndsl.dsl.dace.builder.stree.optimizations.off_grid_conditionals import (
    ExtractOffGridConditionals,
    InlineOffGridConditionals,
    MergeConditionals,
    RevertSimplifyConditional,
    SimplifyConditional,
)
from ndsl.dsl.dace.builder.stree.optimizations.off_grid_tasklet import (
    ExtractOffGridTasklet,
    InlineOffGridTasklet,
)
from ndsl.dsl.dace.builder.stree.pipeline import StreePipeline
from ndsl.dsl.optimization_config import OptimizationHint


class CartesianMergePipeline(StreePipeline):
    """Merge Cartesian computation blocks.

    Args:
        backend: The loop order influences the merge order.
        align_lhs_on_center_horizontal: Attempt to align the LHS writes on center so we can push merging.
            Restrict application to horizontal axis.
            Experimental, defaults to False.
        align_lhs_on_center_vertical: Whether to merge vertical axis maps at the cost of an if statement.
            Restrict application to vertical axis.
            Experimental, defaults to True.
        overcompute_horizontal: Merge horizonal axis maps at the cost of an if statement.
            Defaults to True.
        overcompute_vertical: Merge vertical axis maps at the cost of an if statement.
            Defaults to True.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        hint: OptimizationHint,
        align_lhs_on_center_horizontal: bool = False,
        align_lhs_on_center_vertical: bool = False,
        overcompute_horizontal: bool = True,
        overcompute_vertical: bool = True,
        merge_order: str = "default",
    ) -> None:
        self._backend = backend
        self._merge_order = merge_order
        if self._merge_order not in (
            "default",
            "IJK",
            "IKJ",
            "JIK",
            "JKI",
            "KIJ",
            "KJI",
        ):
            raise ValueError(f"Unexpected merge order {self._merge_order}.")
        axis_merge_order = self._axis_merge_order()

        passes = []

        # Align LHS on center when possible to maximize mergeability
        for axis in axis_merge_order:
            if (axis in HORIZONTAL_AXIS_SYMBOLS and align_lhs_on_center_horizontal) or (
                axis in VERTICAL_AXIS_SYMBOLS and align_lhs_on_center_vertical
            ):
                passes.append(LHSWriteOnCenter(axis))

        # Get offgrid tasklet out of the way
        passes.append(ExtractOffGridTasklet())

        # Get conditional out of the way
        simplify_conditional = SimplifyConditional()
        passes.append(simplify_conditional)
        for axis in axis_merge_order:
            passes.append(InlineOffGridConditionals(axis))
        passes.append(RevertSimplifyConditional(simplify_conditional))

        # We are ready to merge
        for axis in axis_merge_order:
            # If we want to overcompute we first merge the maps that don't need overcomputation
            # to be merged so we gather the bigger contiguous maps first, before we introduce
            # the overcomputation guards.
            passes.append(
                CartesianAxisMerge(
                    axis,
                    overcompute=False,
                    hint=hint,
                )
            )
            if (axis in HORIZONTAL_AXIS_SYMBOLS and overcompute_horizontal) or (
                axis in VERTICAL_AXIS_SYMBOLS and overcompute_vertical
            ):
                passes.append(
                    CartesianAxisMerge(
                        axis,
                        overcompute=True,
                        hint=hint,
                    )
                )

        # Optimize cache-friendliness of offgrid conditional
        passes.append(ExtractOffGridConditionals())
        passes.append(MergeConditionals())

        # Optimize cache-friendliness of offgrid tasklet
        passes.append(InlineOffGridTasklet())

        super().__init__(passes=passes)

    def _axis_merge_order(self) -> tuple[AxisIterator, ...]:
        if self._merge_order == "default":
            return axis_from_backend(self._backend)

        return self._axis_from_merge_order()

    def _axis_from_merge_order(
        self,
    ) -> tuple[AxisIterator, ...]:
        assert len(self._merge_order) == 3
        return tuple(AxisIterator[f"_{axis}"] for axis in self._merge_order)
