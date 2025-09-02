import dace.sdfg.analysis.schedule_tree.treenodes as dst


class RenameTaskletSymbols(dst.ScheduleNodeVisitor):
    def __init__(self, mappings: dict[str, str]) -> None:
        super().__init__()
        self._mappings = mappings

    def visit_TaskletNode(self, node: dst.TaskletNode):
        pass
