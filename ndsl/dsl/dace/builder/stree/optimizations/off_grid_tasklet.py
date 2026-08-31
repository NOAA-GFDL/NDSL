import copy

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log
from ndsl.dsl.dace.builder.stree.common import AxisIterator, list_index
from ndsl.dsl.dace.builder.stree.common.code_block import (
    make_unique_container_name,
    replace_variable_name,
)
from ndsl.dsl.dace.builder.stree.common.memlet import memlet_is_transient_scalar


class TransientScalarSSA(tn.ScheduleNodeVisitor):
    """Transform all transient scalar throught SSA, e.g.
    ```python
        A = tasklet()
        if A:
            f0 = tasklet(A, ...)
        A = tasklet()
        if A:
            f1 = tasklet(A, ...)
    ```
    becomes
    ```python
        A = tasklet()
        if A:
            f0 = tasklet(A, ...)
        A_0 = tasklet()
        if A_0:
            f1 = tasklet(A_0, ...)
    ```
    """

    def __init__(self) -> None:
        self._ssa_book: dict[str, str] = {}

    def _make_SSA(self, name: str, node: tn.ScheduleTreeNode) -> None:
        if name not in self._ssa_book:
            # The first assign - we keep the name
            self._ssa_book[name] = name
            return

        candidate = make_unique_container_name(name, node.get_root())
        self._ssa_book[name] = candidate
        node.get_root().containers[candidate] = copy.copy(
            node.get_root().containers[name]
        )

    def visit_TaskletNode(self, node: tn.TaskletNode) -> None:
        
        # Swap input names
        # ⚠️ this needs to be done _before_ we potentially make a 
        #    new name for a scalar so computation is done from the
        #    previous valid scalar name
        for in_memlet in node.in_memlets.values():
            if in_memlet.data in self._ssa_book:
                in_memlet.data = self._ssa_book[in_memlet.data]

        for out_memlet in node.out_memlets.values():
            if not memlet_is_transient_scalar(node.get_root(), out_memlet):
                continue
            name = out_memlet.data
            self._make_SSA(name, node)
            # Update the memlet data
            out_memlet.data = self._ssa_book[name]

    def visit_IfScope(self, node: tn.IfScope) -> None:
        for in_memlet in node.input_memlets():
            name = in_memlet.data
            if name not in self._ssa_book:
                continue
            # Update the condtional code and memlet data
            replace_variable_name(node.condition, name, self._ssa_book[name])
            in_memlet.data = self._ssa_book[name]


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
        self._off_grid_symbols: dict[str, str] = {}

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        # Make sure all transient scalar
        TransientScalarSSA().visit(node)

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

            # Are we on-grid, e.g. are we dependent on grid indexation
            #
            # Dev NOTE: because we can have _many_ grids due to DaCe parsing some pythong
            # into its own many-maps we go more stricter here and we restrict to
            # any symbols that are not resolved yet (and therefore could be indexer).
            # The fix is to create a pass that collects all the grids, and change this check
            # to look into the "indexer symbols" as well as the topology to see if we are indeed
            # under a grid calculation (e.g. under or after maps indexed)
            if memlet.free_symbols != set():
                # Collect the output for future check
                for out_memlet in node.out_memlets.values():
                    self._on_grid_data.add(out_memlet.data)
                return

            # Not on grid, but do we depend on a data that was calculated on
            # the grid before (transitivity)
            if memlet.data in self._on_grid_data:
                return

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
