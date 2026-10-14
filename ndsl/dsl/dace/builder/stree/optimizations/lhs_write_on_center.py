import itertools

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log
from ndsl.dsl.dace.builder.stree.common import (
    AxisIterator,
    is_axis_map,
)


class LHSWriteOnCenter(tn.ScheduleNodeVisitor):
    """Attempt to enforce a left hand side write on center, per axis, by moving the bounds
    of the map and local offsets on array access.

    This pass is written defensively and will only apply if all the offset are the same on all
    inputs and outputs of the tasklet under a cartesian map."""

    def __init__(self, axis: AxisIterator) -> None:
        super().__init__()
        self._axis = axis
        self._aligned_maps = 0

    def __str__(self) -> str:
        return f"LHSWriteOnCenter{self._axis.as_str().lower()}"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:

        for child in node.children:
            self.visit(child)

        ndsl_log.debug(
            f"🚀 Align {self._aligned_maps} {self._axis} maps to be centered on left hand side"
        )

    def _collect_offsets(self, node: tn.ScheduleTreeNode, offsets: set[str]) -> None:
        if isinstance(node, tn.TaskletNode):
            for memlet in itertools.chain(
                node.in_memlets.values(), node.out_memlets.values()
            ):
                for offset in memlet.subset.string_list():
                    if self._axis.as_str() == offset:
                        # Case of no offset
                        offsets.add(f"{self._axis.as_str()} + 0")
                    elif self._axis.as_str() in offset:
                        # extract the offset
                        offsets.add(offset)
        elif isinstance(node, tn.ScheduleTreeScope):
            for child in node.children:
                self._collect_offsets(child, offsets)

    def _process_offsets(
        self, node: tn.ScheduleTreeNode, replace: tuple[str, str]
    ) -> None:
        if isinstance(node, tn.TaskletNode):
            for memlet in itertools.chain(
                node.in_memlets.values(), node.out_memlets.values()
            ):
                replaced_memlet = memlet._label(None).replace(replace[0], replace[1])
                memlet._parse_memlet_from_str(replaced_memlet)
        elif isinstance(node, tn.ScheduleTreeScope):
            for child in node.children:
                self._process_offsets(child, replace)

    def visit_MapScope(self, node: tn.MapScope) -> None:
        # Dev NOTE: ⚠️ LHS offset in K is _allowed_ when using ForScope. ⚠️
        #   The system _should_ be resilient to it because it looks for homogeneity but shall remain
        #   a fundamental difference with IJ which is never allowed to be offseted on the LHS
        if not is_axis_map(node, self._axis):
            return

        offsets: set[str] = set()
        for child in node.children:
            self._collect_offsets(child, offsets)

        if len(offsets) != 1:
            return

        offset_and_axis = offsets.pop()
        offset_no_whitespace = offset_and_axis.replace(self._axis.as_str(), "").replace(
            " ", ""
        )
        if offset_no_whitespace == "+0":
            return

        # Our offset should look like "+X" with X the value
        assert offset_no_whitespace[0] in ["+", "-"]
        plus = offset_no_whitespace[0] == "-"
        value = int(offset_no_whitespace[1])

        # Offset the range of the Map
        range_ = [value]
        node.node.map.range.offset(range_, plus)

        # Move all K to 0
        for child in node.children:
            self._process_offsets(child, (offset_and_axis, self._axis.as_str()))

        self._aligned_maps += 1
