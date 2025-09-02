import dace.sdfg.analysis.schedule_tree.treenodes as dst
import dace.subsets as sbs


class SpecializeCartesianMaps(dst.ScheduleNodeVisitor):
    def __init__(self, mappings: dict[str, int]) -> None:
        super().__init__()
        self._mappings = mappings

    def visit_MapScope(self, node: dst.MapScope):
        dims = []
        for p in node.node.map.params:
            if p == "__i":
                dims.append((0, self._mappings["__I"], 1))
            if p == "__j":
                dims.append((0, self._mappings["__J"], 1))
            if p.startswith("__k"):
                dims.append((0, self._mappings["__K"], 1))
        node.node.map.range = sbs.Range(dims)

        self.visit(node.children)
