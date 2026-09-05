import copy

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.utils import find_new_name

from ndsl import ndsl_log
from ndsl.dsl.dace.builder.stree.common import (
    AxisIterator,
    is_cartesian_axis,
    list_index,
)
from ndsl.dsl.dace.builder.stree.common.code_block import replace_variable_name
from ndsl.dsl.dace.builder.stree.common.memlet import memlet_is_transient_scalar

# ----- DEV NOTE -----
# The pass below have been coded defensively, e.g. they restrict their application to non-cartesian code
# by skipping the visitor as soon as we detect that we are entering cartesian code
# ----- DEV NOTE -----


class OffGridTransientScalarSSA(tn.ScheduleNodeVisitor):
    """Transform all off-grid transient scalar through SSA without Phi functions, e.g.
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
    but when we have
    ```
    A = tasklet()
    if B:
        A = tasklet(...)
    if C:
        A = tasklet(...)
    D = tasklet(A)
    ```
    we do keep `A` throughout (this is solvable by introducing a phi function for `D =tasklet(phi(A)))`
    which would capture the control flow above and replicate it.

    Because we do not address the Phi function problem this pass _will not_ yield a perfect SSA IR for
    transient scalars. Use `OffGridTransientScalarWriteCounter` afterward to map results.

    The pass _will not_ apply SSA within cartesian blocks, but will keep read renaming going.
    """

    def __init__(self) -> None:
        self._only_writes: set[str] = set()
        self._ssa_book: dict[str, str] = {}

    def _make_SSA(self, name: str, node: tn.ScheduleTreeNode) -> None:
        if name not in self._ssa_book:
            self._ssa_book[name] = name
            self._only_writes.add(name)
            return

        # If we didn't have any read before we don't SSA
        if name in self._only_writes:
            return

        candidate = find_new_name(name, node.get_root().containers)
        self._ssa_book[name] = candidate
        node.get_root().containers[candidate] = copy.copy(
            node.get_root().containers[name]
        )

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        for child in node.children:
            self.visit(child, in_cartesian=False)

    def visit_MapScope(self, node: tn.MapScope, in_cartesian: bool) -> None:
        for child in node.children:
            self.visit(child, in_cartesian=is_cartesian_axis(node))

    def visit_ForScope(self, node: tn.ForScope, in_cartesian: bool) -> None:
        for child in node.children:
            self.visit(child, in_cartesian=is_cartesian_axis(node))

    def visit_TaskletNode(self, node: tn.TaskletNode, in_cartesian: bool) -> None:

        # Swap input names
        # ⚠️ this needs to be done _before_ we potentially make a
        #    new name for a scalar so computation is done from the
        #    previous valid scalar name
        for in_memlet in node.in_memlets.values():
            if in_memlet.data in self._ssa_book:
                in_memlet.data = self._ssa_book[in_memlet.data]
                self._only_writes.discard(in_memlet.data)

        for out_memlet in node.out_memlets.values():
            if not memlet_is_transient_scalar(node.get_root(), out_memlet):
                continue

            # Restrict making SSA when we are not under a cartesian maps
            name = out_memlet.data
            if not in_cartesian:
                self._make_SSA(name, node)

            # Update the memlet data (if we have SSA executed on the scalar)
            if name in self._ssa_book:
                out_memlet.data = self._ssa_book[name]

    def visit_IfScope(self, node: tn.IfScope, in_cartesian: bool) -> None:
        for in_memlet in node.input_memlets():
            name = in_memlet.data
            if name not in self._ssa_book:
                continue

            # We read - remove from writes only
            self._only_writes.discard(in_memlet.data)

            # Update the conditional code and memlet data
            replace_variable_name(node.condition, name, self._ssa_book[name])
            in_memlet.data = self._ssa_book[name]

        for child in node.children:
            self.visit(child, in_cartesian=in_cartesian)

    def visit_WhileScope(self, node: tn.WhileScope, in_cartesian: bool) -> None:
        for in_memlet in node.input_memlets():
            name = in_memlet.data
            if name not in self._ssa_book:
                continue

            # We read - remove from writes only
            self._only_writes.discard(in_memlet.data)

            # Update the conditional codes and memlet data
            replace_variable_name(node.loop.init_statement, name, self._ssa_book[name])
            replace_variable_name(
                node.loop.update_statement, name, self._ssa_book[name]
            )
            replace_variable_name(node.loop.loop_condition, name, self._ssa_book[name])
            in_memlet.data = self._ssa_book[name]

        for child in node.children:
            self.visit(child, in_cartesian=in_cartesian)


class OffGridTransientScalarWriteCounter(tn.ScheduleNodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.scalar_writes: dict[str, int] = {}  # SSA

    def visit_TaskletNode(self, node: tn.TaskletNode) -> None:
        for out_memlet in node.out_memlets.values():
            if not memlet_is_transient_scalar(node.get_root(), out_memlet):
                continue

            self.scalar_writes[out_memlet.data] = (
                self.scalar_writes.get(out_memlet.data, 0) + 1
            )


class ExtractOffGridTasklet(tn.ScheduleNodeVisitor):
    """
    Gather all off-grid nodes and store them at the top of the Schedule Tree
    like a C style declarative process.

    Dev note: this pass functions because we keep the nodes _in order_ the entire
    time, keeping any dependency on each other correct.

    We move `tasklet` if all its writes have been SSA properly.

    The pass _will not_ apply within cartesian blocks.
    """

    def __str__(self) -> str:
        return "ExtractOffGridTasklet"

    def __init__(self) -> None:
        super().__init__()
        self._on_grid_data: set[str] = set()
        self._nodes_to_extract: list[tn.ScheduleTreeNode] = []
        self._transient_scalar_writes: dict[str, int] = {}

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        # Best effort at deploying SSA on transient scalar
        OffGridTransientScalarSSA().visit(node)

        # Count the result of the above pass
        count = OffGridTransientScalarWriteCounter()
        count.visit(node)
        self._transient_scalar_writes = count.scalar_writes

        # Sort all tasklets between off/on-grid, with memlet
        # dataflow checks and SSA check
        for child in node.children:
            self.visit(child)

        # Un-parent
        for tasklet in self._nodes_to_extract:
            assert tasklet.parent is not None
            parent = tasklet.parent
            parent.children.pop(list_index(tasklet, parent.children))
            tasklet.parent = node

        # Re-insert in front
        node.children = [*self._nodes_to_extract, *node.children]

    def visit_MapScope(self, node: tn.MapScope) -> None:
        if is_cartesian_axis(node):
            return

        for child in node.children:
            self.visit(child)

    def visit_ForScope(self, node: tn.ForScope) -> None:
        if is_cartesian_axis(node):
            return

        for child in node.children:
            self.visit(child)

    def visit_IfScope(self, node: tn.IfScope) -> None:
        # Parse all children first
        for child in node.children:
            self.visit(child)

        # Dev NOTE: We can detect the case where the IfScope is entirely
        # made of off-grid tasklets, e.g.
        #   if A:
        #        t_0(a, b)
        #        t_1(c, d)
        # If it is the case, then we need to move the `if` as whole.
        # Then if we have a mix of off and on-grid, we only move the tasklet that writes into SSA scalars.
        # Implementing the above failed into an infinite recursion when checking child status (e.g are you
        # offgrid). But the logic should work (will require modifying the Tasklet visitor so we
        # can sort between single writes offgrid and many writes offgrid)

        # Compute the nature of our conditional
        # all_off_grid = True
        # for child in node.children:
        #     if child not in self._off_grid_nodes:
        #         all_off_grid = False
        #         break

        # if all_off_grid:
        #     # We have our ideal case: move the entire `if`
        #     self._off_grid_nodes.append(node)
        # else:
        #     # We may have write dependency, remove all node that are many writes
        #     for child in node.children:
        #         if (
        #               child in self._single_transient_write_off_grid_nodes or
        #               child not in self._off_grid_nodes:
        #         ):
        #             continue
        #         self._off_grid_nodes.remove(node)

    def visit_TaskletNode(self, node: tn.TaskletNode) -> None:
        # Check the inputs are not on-grid and the tasklet output are all SSA
        for memlet in [*node.in_memlets.values(), *node.out_memlets.values()]:

            # Are we on-grid, e.g. are we dependent on grid indexation
            #
            # Dev NOTE: because we can have _many_ grids due to DaCe parsing some python
            # into its own many-maps we go more stricter here and we restrict to
            # any symbols that are not resolved yet (and therefore could be indexer).
            # The fix is to create a pass that collects all the grids, and change this check
            # to look into the "indexer symbols" as well as the topology to see if we are indeed
            # under a grid calculation (e.g. under or after maps indexed)
            if len(memlet.free_symbols) > 0:
                # Collect the output for future check
                for out_memlet in node.out_memlets.values():
                    self._on_grid_data.add(out_memlet.data)
                return

            # Not on grid, but do we depend on a data that was calculated on
            # the grid before (transitivity)
            if memlet.data in self._on_grid_data:
                return

            # We check that the output is SSA, e.g. written a single time throughout
            # if not then we will not extract the tasklet
            if self._transient_scalar_writes.get(memlet.data, 0) > 1:
                return

        # All check valid, this node can be extracted
        self._nodes_to_extract.append(node)


class InlineOffGridTasklet(tn.ScheduleNodeVisitor):
    """
    Move the off-grid tasklet as-close-as possible to their first usage but
    _outside_ of any grid tight loop, e.g. C++ optimization style.

    Dev note: movement need to be made in reverse order so we can keep any dependency on
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
