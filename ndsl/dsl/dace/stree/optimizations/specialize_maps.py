import dace.subsets as sbs
from dace.sdfg.analysis.schedule_tree import treenodes as tn


class SpecializeCartesianMaps(tn.ScheduleNodeVisitor):
    def __init__(self, mappings: dict[str, int]) -> None:
        super().__init__()
        self._mappings = mappings

    def visit_MapScope(self, node: tn.MapScope) -> None:
        dims = []
        for p in node.node.map.params:
            assert isinstance(p, str)
            if p == "__i":
                dims.append((0, self._mappings["__I"], 1))
            if p == "__j":
                dims.append((0, self._mappings["__J"], 1))
            if p == "__k":
                dims.append((0, self._mappings["__K"], 1))
        node.node.map.range = sbs.Range(dims)

        self.visit(node.children)

    def __str__(self) -> str:
        return "SpecializeCartesianMaps"
