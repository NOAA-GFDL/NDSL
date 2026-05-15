from __future__ import annotations

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
    """Merge Cartesian computation blocks"""

    def __init__(self, backend: Backend, *, eager: bool = True) -> None:
        self._backend = backend
        self.eager = eager

    def __str__(self) -> str:
        return "CartesianMerge"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        InlineOffgridConditionals().visit(node)
        MergeConditionals().visit(node)

        if self._backend.loop_order == BackendLoopOrder.IJK:
            CartesianAxisMerge(AxisIterator._I).visit(node)
            CartesianAxisMerge(AxisIterator._J).visit(node)
            CartesianAxisMerge(AxisIterator._K).visit(node)
        elif self._backend.loop_order == BackendLoopOrder.IKJ:
            CartesianAxisMerge(AxisIterator._I).visit(node)
            CartesianAxisMerge(AxisIterator._K).visit(node)
            CartesianAxisMerge(AxisIterator._J).visit(node)
        elif self._backend.loop_order == BackendLoopOrder.JIK:
            CartesianAxisMerge(AxisIterator._J).visit(node)
            CartesianAxisMerge(AxisIterator._I).visit(node)
            CartesianAxisMerge(AxisIterator._K).visit(node)
        elif self._backend.loop_order == BackendLoopOrder.JKI:
            CartesianAxisMerge(AxisIterator._J).visit(node)
            CartesianAxisMerge(AxisIterator._K).visit(node)
            CartesianAxisMerge(AxisIterator._I).visit(node)
        elif self._backend.loop_order == BackendLoopOrder.KIJ:
            CartesianAxisMerge(AxisIterator._K).visit(node)
            CartesianAxisMerge(AxisIterator._I).visit(node)
            CartesianAxisMerge(AxisIterator._J).visit(node)
        elif self._backend.loop_order == BackendLoopOrder.KJI:
            CartesianAxisMerge(AxisIterator._K).visit(node)
            CartesianAxisMerge(AxisIterator._J).visit(node)
            CartesianAxisMerge(AxisIterator._I).visit(node)

        ExtractOffgridConditionals().visit(node)
        MergeConditionals().visit(node)
