import itertools
import re

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.symbolic import symbol


class ReplaceAxisSymbol(tn.ScheduleNodeVisitor):
    def __init__(self, axis_replacements: dict[str | symbol, str | symbol]) -> None:
        self._axis_replacements = axis_replacements

    def visit_TaskletNode(self, node: tn.TaskletNode) -> None:
        for memlet in itertools.chain(
            node.in_memlets.values(), node.out_memlets.values()
        ):
            memlet.replace(self._axis_replacements)

        if node.node.label.startswith("masklet"):
            for old, new in self._axis_replacements.items():
                # use regex to match word boundaries (with `\b`)
                node.node.code.as_string = re.sub(
                    rf"\b{str(old)}\b", str(new), node.node.code.as_string
                )

    def visit_IfScope(self, node: tn.IfScope) -> None:
        for old, new in self._axis_replacements.items():
            # use regex to match word boundaries (with `\b`)
            node.condition.as_string = re.sub(
                rf"\b{str(old)}\b", str(new), node.condition.as_string
            )

        for child in node.children:
            self.visit(child)

    def __str__(self) -> str:
        return "ReplaceAxisSymbol"
