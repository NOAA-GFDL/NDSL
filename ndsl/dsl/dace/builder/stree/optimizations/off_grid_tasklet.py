from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log
from ndsl.dsl.dace.builder.stree.common import AxisIterator, list_index
from ndsl.dsl.dace.builder.stree.common.control_flow import is_off_grid_tasklet


class ExtractOffGridTasklet(tn.ScheduleNodeVisitor):
    """
    Gather all off-grid tasklet and store them at the top of the Schedule Tree
    like a C style declarative process.

    Dev note: this pass functions because we keep the tasklet _in order_ the entire
    time, keeping alive any dependancy on each other correct.
    """

    def __str__(self) -> str:
        return "ExtractOffGridTasklet"

    def __init__(self) -> None:
        super().__init__()
        self._off_grid_tasklets: list[tn.TaskletNode] = []
        self._on_grid_tasklets: list[tn.TaskletNode] = []

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:

        # Sort all tasklets between off/on-grid, without memlet
        # dataflow checks
        for child in node.children:
            self.visit(child)

        # Prune the off-grid tasklet where input memlets dataflow
        # is linked to an on-grid tasklet
        pruned_off_grid = []
        pruned_on_grid = self._on_grid_tasklets
        for off_grid_tasklet in self._off_grid_tasklets:
            is_on_grid_dep = False
            for on_grid_tasklet in pruned_on_grid:
                for in_memlet in off_grid_tasklet.in_memlets.values():
                    for out_memlet in on_grid_tasklet.out_memlets.values():
                        if in_memlet.data == out_memlet.data:
                            is_on_grid_dep = True
                            break
                if is_on_grid_dep:
                    break
            if is_on_grid_dep:
                pruned_on_grid.append(off_grid_tasklet)
            else:
                pruned_off_grid.append(off_grid_tasklet)

        # Un-parent
        for tasklet in pruned_off_grid:
            assert tasklet.parent is not None
            parent = tasklet.parent
            parent.children.pop(list_index(tasklet, parent.children))
            tasklet.parent = node

        # Re-insert in front
        node.children = [*pruned_off_grid, *node.children]

    def visit_TaskletNode(self, node: tn.TaskletNode) -> None:
        # Collect offgrid tasklet
        if is_off_grid_tasklet(node):
            self._off_grid_tasklets.append(node)
        else:
            self._on_grid_tasklets.append(node)


class InlineOffGridTasklet(tn.ScheduleNodeVisitor):
    """
    Move the off-grid tasklet as-close-as possible to their first usage but
    _outside_ of any grid tight loop, e.g. C++ optimization style.

    Dev note: movement need to be made in reverse order so we can keep any dependancy on
    each other tasklet correct e.g. in
    ```
    A = tasklet()
    B = tasklet(A)
    ...
    map _i
        C = tasklet(B, filed[_i])
    ```
    B needs to be move first against `map_i` then, A in before B
    """

    _axis: AxisIterator

    def __str__(self) -> str:
        return "InlineOffGridTasklet"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        ndsl_log.debug("InlineOffGridTasklet is not implemented yet.")
