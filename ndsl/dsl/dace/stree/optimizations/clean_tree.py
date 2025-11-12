from __future__ import annotations

import dace.sdfg.analysis.schedule_tree.treenodes as stree

from ndsl import ndsl_log


class CleanUpScheduleTree(stree.ScheduleNodeTransformer):
    """Clean up unused nodes, or nodes barrying further optimizations."""

    def __init__(self) -> None:
        self.cleaned_state_boundaries = 0

    def __str__(self) -> str:
        return "CleanUpScheduleTree"

    def _remove_from_my_childs(self, node: stree.ScheduleTreeScope):
        to_remove = [
            child
            for child in node.children
            if isinstance(child, stree.StateBoundaryNode)
        ]
        for to_remove_child in to_remove:
            self.cleaned_state_boundaries += 1
            node.children.remove(to_remove_child)

    def visit_WhileScope(self, node: stree.WhileScope):
        self._remove_from_my_childs(node)
        for child in node.children:
            self.visit(child)

        return node

    def visit_ForScope(self, node: stree.ForScope):
        self._remove_from_my_childs(node)
        for child in node.children:
            self.visit(child)

        return node

    def visit_MapScope(self, node: stree.MapScope):
        self._remove_from_my_childs(node)
        for child in node.children:
            self.visit(child)

        return node

    def visit_IfScope(self, node: stree.MapScope):
        self._remove_from_my_childs(node)
        for child in node.children:
            self.visit(child)

        return node

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        self._remove_from_my_childs(node)
        for child in node.children:
            self.visit(child)

        ndsl_log.debug(
            f"Clean up StateBoundary : {self.cleaned_state_boundaries} nodes"
        )
