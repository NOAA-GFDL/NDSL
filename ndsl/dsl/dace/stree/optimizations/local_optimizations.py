from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import OptimizationConfig
from ndsl.dsl.dace.stree.optimizations.common import list_index


class LabeledSection(tn.ScheduleTreeScope):
    def __init__(
        self,
        *,
        children: list[tn.ScheduleTreeNode],
        parent: tn.ScheduleTreeScope,
        label: str,
        optimizations: OptimizationConfig,
    ) -> None:
        super().__init__(children=children, parent=parent)
        self.label = label
        self.optimizations = optimizations

    def as_string(self, indent: int = 0) -> str:
        result = indent * tn.INDENTATION + f"section '{self.label}':\n"
        return result + super().as_string(indent)


class _LabelSections(tn.ScheduleNodeVisitor):
    _enter_labels: list[tn.LibraryCall]

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "_LabelSections"

    def visit_LibraryCall(self, node: tn.LibraryCall) -> None:
        # Only look at "our" label nodes
        if node.node.name != "NDSLRuntime_Label":
            return

        if node.node.unique_name.startswith("Enter__"):
            # keep taps on where we start
            self._enter_labels.append(node)
            return

        if node.node.unique_name.startswith("Exit__"):
            # find the matching start point
            section_start = self._enter_labels.pop()

            # sanity checks
            # - ensure we have the right section
            name = section_start.node.unique_name.removeprefix("Enter__")
            exit_name = node.node.unique_name.removeprefix("Exit__")
            assert name == exit_name
            # - ensure we have the same parent (if not something is screwed up)
            parent = section_start.parent
            assert parent == node.parent

            # grab all the nodes in-between and put them in a `LabeledSection`
            start_index = list_index(parent.children, section_start)
            end_index = list_index(parent.children, node)
            new_node = LabeledSection(
                children=parent.children[start_index + 1 : end_index],
                parent=parent,
                label=name,
                optimizations=node.node._local_optimizations,
            )

            # overwrite the nodes (including the labels) with the new node
            parent.children[start_index : end_index + 1] = [new_node]

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        # reset the stack of enter labels
        self._enter_labels = []

        # then, visit all the children
        self.generic_visit(node)

        # make sure we have replaced everybody
        assert len(self._enter_labels) == 0


class LocalOptimizations(tn.ScheduleNodeVisitor):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "LocalOptimizations"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        # First, parse enter/exit labels into `LabeledSection`s.
        _LabelSections().visit(node)

        # Then, apply local optimizations on children of `LabeledSection`s.
        assert node
