from __future__ import annotations

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl.logging import ndsl_log


class CleanUpScheduleTree(tn.ScheduleNodeTransformer):
    """Remove `StateBoundary` nodes from children of ScheduleTreeScopes."""

    def __init__(self) -> None:
        self._removed_state_boundaries = 0

    def __str__(self) -> str:
        return "CleanUpScheduleTree"

    def _remove_state_boundaries_from_children(
        self, node: tn.ScheduleTreeScope
    ) -> None:
        to_remove = [
            child for child in node.children if isinstance(child, tn.StateBoundaryNode)
        ]
        for boundary in to_remove:
            self._removed_state_boundaries += 1
            node.children.remove(boundary)

    def visit_WhileScope(self, node: tn.WhileScope) -> tn.WhileScope:
        self._remove_state_boundaries_from_children(node)

        for child in node.children:
            self.visit(child)

        return node

    def visit_ForScope(self, node: tn.ForScope) -> tn.ForScope:
        self._remove_state_boundaries_from_children(node)

        for child in node.children:
            self.visit(child)

        return node

    def visit_MapScope(self, node: tn.MapScope) -> tn.MapScope:
        self._remove_state_boundaries_from_children(node)

        for child in node.children:
            self.visit(child)

        return node

    def visit_IfScope(self, node: tn.IfScope) -> tn.IfScope:
        self._remove_state_boundaries_from_children(node)
        for child in node.children:
            self.visit(child)

        return node

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        self._removed_state_boundaries = 0

        self._remove_state_boundaries_from_children(node)

        for child in node.children:
            self.visit(child)

        ndsl_log.debug(f"{self}: removed {self._removed_state_boundaries} nodes")
