from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl.config import Backend, BackendLoopOrder
from ndsl.dsl.dace.stree.optimizations.axis_merge import CartesianAxisMerge
from ndsl.dsl.dace.stree.optimizations.common import AxisIterator
from ndsl.dsl.dace.stree.optimizations.offgrid_conditionals import (
    ExtractOffgridConditionals,
    InlineOffgridConditionals,
    MergeConditionals,
)


class CartesianMerge(tn.ScheduleNodeTransformer):
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
        super().__init__()
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

    def __str__(self) -> str:
        return "CartesianMerge"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        axis_merge_order = self._axis_merge_order()
        for axis in axis_merge_order:
            InlineOffgridConditionals(axis).visit(node)
        MergeConditionals().visit(node)

        for axis in axis_merge_order:
            CartesianAxisMerge(axis, overcompute=self._overcompute).visit(node)

        ExtractOffgridConditionals().visit(node)
        MergeConditionals().visit(node)

    def _axis_merge_order(self) -> tuple[AxisIterator, AxisIterator, AxisIterator]:
        if self._merge_order == "default":
            return self._axis_merge_order_default()

        return self._axis_merge_order_custom()

    def _axis_merge_order_default(
        self,
    ) -> tuple[AxisIterator, AxisIterator, AxisIterator]:
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

    def _axis_merge_order_custom(
        self,
    ) -> tuple[AxisIterator, AxisIterator, AxisIterator]:
        if self._merge_order == "IJK":
            return (AxisIterator._I, AxisIterator._J, AxisIterator._K)

        if self._merge_order == "IKJ":
            return (AxisIterator._I, AxisIterator._K, AxisIterator._J)

        if self._merge_order == "JIK":
            return (AxisIterator._J, AxisIterator._I, AxisIterator._K)

        if self._merge_order == "JKI":
            return (AxisIterator._J, AxisIterator._K, AxisIterator._I)

        if self._merge_order == "KIJ":
            return (AxisIterator._K, AxisIterator._I, AxisIterator._J)

        assert self._merge_order == "KJI"
        return (AxisIterator._K, AxisIterator._J, AxisIterator._I)
