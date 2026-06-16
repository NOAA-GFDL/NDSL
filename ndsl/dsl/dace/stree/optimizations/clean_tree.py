from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log


class CleanUpScheduleTree(tn.ScheduleNodeTransformer):
    """Remove NDSL runtime labels and `StateBoundary` nodes from children of ScheduleTreeScopes."""

    def __init__(self) -> None:
        super().__init__()
        self._removed_state_boundaries = 0
        self._removed_labels = 0

    def __str__(self) -> str:
        return "CleanUpScheduleTree"

    def visit_LibraryCall(self, node: tn.LibraryCall) -> tn.LibraryCall | None:
        if node.node.name == "NDSLRuntime_Label":
            self._removed_labels += 1
            return None
        return node

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

        self.generic_visit(node)

        return node

    def visit_ForScope(self, node: tn.ForScope) -> tn.ForScope:
        self._remove_state_boundaries_from_children(node)

        self.generic_visit(node)

        return node

    def visit_MapScope(self, node: tn.MapScope) -> tn.MapScope:
        self._remove_state_boundaries_from_children(node)

        self.generic_visit(node)

        return node

    def visit_IfScope(self, node: tn.IfScope) -> tn.IfScope:
        self._remove_state_boundaries_from_children(node)

        self.generic_visit(node)

        return node

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> tn.ScheduleTreeRoot:
        self._removed_state_boundaries = 0
        self._removed_labels = 0

        self._remove_state_boundaries_from_children(node)

        self.generic_visit(node)

        ndsl_log.debug(
            f"{self}: removed {self._removed_state_boundaries} boundaries and {self._removed_labels} labels."
        )
        return node
