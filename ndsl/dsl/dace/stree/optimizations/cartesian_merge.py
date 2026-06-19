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

    def __init__(self, backend: Backend, *, overcompute: bool = True) -> None:
        super().__init__()
        self._backend = backend
        self._overcompute = overcompute

    def __str__(self) -> str:
        return "CartesianMerge"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        for axis in self._backend_order():
            InlineOffgridConditionals(axis).visit(node)
        MergeConditionals().visit(node)

        for axis in self._backend_order():
            CartesianAxisMerge(axis, overcompute=self._overcompute).visit(node)

        ExtractOffgridConditionals().visit(node)
        MergeConditionals().visit(node)

    def _backend_order(self) -> tuple[AxisIterator, AxisIterator, AxisIterator]:
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
