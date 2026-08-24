from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log
from ndsl.dsl.dace.builder.stree.common import AxisIterator, list_index


class ExtractOffGridTasklet(tn.ScheduleNodeVisitor):
    """
    Gather all off-grid tasklet and store them at the top of the Schedule Tree
    like a C style declarative process.

    Dev note: this pass functions because we keep the tasklet _in order_ the entire
    time, keeping alive any dependency on each other correct.
    """

    def __str__(self) -> str:
        return "ExtractOffGridTasklet"

    def __init__(self) -> None:
        super().__init__()
        self._on_grid_data: set[str] = set()
        self._off_grid_tasklets: list[tn.TaskletNode] = []

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:

        # Sort all tasklets between off/on-grid, without memlet
        # dataflow checks
        for child in node.children:
            self.visit(child)

        # Un-parent
        for tasklet in self._off_grid_tasklets:
            assert tasklet.parent is not None
            parent = tasklet.parent
            parent.children.pop(list_index(tasklet, parent.children))
            tasklet.parent = node

        # Re-insert in front
        node.children = [*self._off_grid_tasklets, *node.children]

    def visit_TaskletNode(self, node: tn.TaskletNode) -> None:
        # Check the inputs are not on-grid
        for memlet in [*node.in_memlets.values(), *node.out_memlets.values()]:
            # Are we on-grid
            if memlet.free_symbols != set():
                # Collect the output for future check
                for out_memlet in node.out_memlets.values():
                    self._on_grid_data.add(out_memlet.data)
                return

            # Not on grid, but do we depend on a data that was calculated on
            # the grid before (transitivity)
            if memlet.data in self._on_grid_data:
                return

        # We are truly off-grid
        self._off_grid_tasklets.append(node)


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
