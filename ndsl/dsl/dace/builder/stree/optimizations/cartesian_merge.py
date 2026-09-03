from ndsl.config import Backend, BackendLoopOrder
from ndsl.dsl.dace.builder.stree.common import AxisIterator
from ndsl.dsl.dace.builder.stree.optimizations.axis_merge import CartesianAxisMerge
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


class CartesianMergePipeline(StreePipeline):
    """Merge Cartesian computation blocks.

    Args:
        backend: The loop order influences the merge order.
        overcompute: Whether to merge at the cost of an if statement. Defaults to True.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        overcompute: bool = True,
        merge_order: str = "default",
    ) -> None:
        self._backend = backend
        self._overcompute = overcompute
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
            passes.append(
                CartesianAxisMerge(
                    axis,
                    overcompute=self._overcompute,
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
            return self._axis_from_backend()

        return self._axis_from_merge_order()

    def _axis_from_backend(
        self,
    ) -> tuple[AxisIterator, ...]:
        if self._backend.loop_order == BackendLoopOrder.IJK:
            return (AxisIterator._I, AxisIterator._J, AxisIterator._K)

        if self._backend.loop_order == BackendLoopOrder.IKJ:
            return (AxisIterator._I, AxisIterator._K, AxisIterator._J)

        if self._backend.loop_order == BackendLoopOrder.JIK:
            return (AxisIterator._J, AxisIterator._I, AxisIterator._K)

        if self._backend.loop_order == BackendLoopOrder.JKI:
            return (AxisIterator._J, AxisIterator._K, AxisIterator._I)

        if self._backend.loop_order == BackendLoopOrder.KIJ:
            return (AxisIterator._K, AxisIterator._I, AxisIterator._J)

        assert self._backend.loop_order == BackendLoopOrder.KJI
        return (AxisIterator._K, AxisIterator._J, AxisIterator._I)

    def _axis_from_merge_order(
        self,
    ) -> tuple[AxisIterator, ...]:
        assert len(self._merge_order) == 3
        return tuple(AxisIterator[f"_{axis}"] for axis in self._merge_order)
