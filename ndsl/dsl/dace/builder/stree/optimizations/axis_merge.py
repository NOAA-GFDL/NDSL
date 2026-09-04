import copy

import dace
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log
from ndsl.dsl.dace.builder.stree.common import (
    AxisIterator,
    detect_cycle,
    get_next_node,
    is_axis_for,
    is_axis_map,
    is_last_node,
    list_index,
    no_data_dependencies_on_cartesian_axis,
    swap_node_position_in_tree,
)
from ndsl.dsl.dace.builder.stree.optimizations.replace_axis_symbol import (
    ReplaceAxisSymbol,
)
from ndsl.dsl.optimization_config import OptimizationHint


def _both_same_single_axis_maps(
    first: tn.MapScope, second: tn.MapScope, axis: AxisIterator
) -> bool:
    return (
        (
            len(first.node.map.params) == 1 and len(second.node.map.params) == 1
        )  # Single axis
        and is_axis_map(first, axis)  # Correct axis in first map
        and is_axis_map(second, axis)  # Correct axis in second map
    )


def _can_merge_axis_maps(
    first: tn.MapScope, second: tn.MapScope, axis: AxisIterator
) -> bool:
    # Dev NOTE: since the merger uses a re-entry system we might check mergeability
    #           many times within the same transformation execution.
    #           We could cache the map that have been tested as non-mergeable in the caller
    #           of this function instead of re-doing the expensive data dependencies analysis.
    #           The only thing to be careful is that we have merged, a map will have different dependencies
    #           but the logic would be that if map A and B can't be merged, any merged A or B will contain
    #           the dependencies that made it non-mergeable in the first place.
    return _both_same_single_axis_maps(
        first, second, axis
    ) and no_data_dependencies_on_cartesian_axis(first, second, axis)


class InsertOvercomputationGuard(tn.ScheduleNodeTransformer):
    """
    Recurse down the cartesian-block of maps before patching the inner maps with the if-guard.

    Recursing down allows to surface the maps for more subsequent merging.
    Recursing down also ensures that maximum parallelization is conserved.
    """

    def __init__(
        self,
        axis_as_string: str,
        *,
        merged_range: dace.subsets.Range,
        original_range: dace.subsets.Range,
        hint: OptimizationHint,
    ):
        self._axis_as_string = axis_as_string
        self._merged_range = merged_range
        self._original_range = original_range
        self._hint = hint

    def _execution_condition(self) -> CodeBlock:
        # NOTE range.ranges are inclusive, e.g.
        #      Range(0:4) -> ranges = (start=1, stop=3, step=1)
        range = self._original_range
        start = range.ranges[0][0]
        stop = range.ranges[0][1]
        step = range.ranges[0][2]
        return CodeBlock(
            f"{self._axis_as_string} >= {start} "
            f"and {self._axis_as_string} <= {stop} "
            f"and ({self._axis_as_string} - {start}) % {step} == 0"
        )

    def visit_MapScope(self, node: tn.MapScope) -> tn.MapScope:
        # To maximize parallelization we push the guard as deep into the cartesian
        # block as we can in order to surface all maps at the cost of higher FLOPs
        if self._hint == OptimizationHint.PARALLEL:
            all_children_are_maps = all(
                isinstance(child, tn.MapScope) for child in node.children
            )
            if all_children_are_maps:
                node.children = self.visit(node.children)
                return node

        if self._merged_range != self._original_range:
            if_scope = tn.IfScope(
                condition=self._execution_condition(),
                children=node.children,
                parent=node,
            )
            # Re-parent to IF
            for child in node.children:
                child.parent = if_scope
            node.children = [if_scope]

        return node


class CartesianAxisMerge(tn.ScheduleNodeTransformer):
    """Merge a cartesian axis if they are contiguous in code-flow.

    Can do:
        - merge a given axis with the next maps at the same recursion level
        - can overcompute to allow for more merging at the cost of an if

    It expects:
        - All Maps and ForLoop are on a single axis - but doesn't check for it.

    This pass has been designed to work in coordination with `OffGridConditional` and
    `OffGridTasklet` passes to allow to align maps for merge

    Args:
        axis: AxisIterator to be merged
        overcompute: merge at the cost of an if statement.
    """

    def __init__(
        self, axis: AxisIterator, *, overcompute: bool, hint: OptimizationHint
    ) -> None:
        self.axis = axis
        self.failed_due_to_data_dep = 0
        self.overcompute = overcompute
        self.hint = hint

    def __str__(self) -> str:
        suffix = "_overcompute" if self.overcompute else ""
        return f"CartesianAxisMerge_{self.axis.name}{suffix}"

    def _merge_node(
        self, node: tn.ScheduleTreeNode, nodes: list[tn.ScheduleTreeNode]
    ) -> int:
        """Direct code to the correct resolver for the node (e.g. visitor)

        Dev Note: Order matters!
            Default behavior for base class must be _after_ bespoke leaf class
            behavior (e.g. IfScope before ControlFlowScope)
        """

        if isinstance(node, tn.MapScope):
            return self._map_overcompute_merge(node, nodes)

        if isinstance(node, tn.ForScope):
            return self._for_merge(node)

        if isinstance(node, tn.TaskletNode):
            # We stop thinking - `OffGridTasklet` should have taken care of those
            return 0

        if isinstance(node, tn.ControlFlowScope):
            return self._default_control_flow(node)

        ndsl_log.debug(f"  (╯°□°)╯︵ ┻━┻: can't merge {type(node)}. Recursion ends.")
        return 0

    def _for_merge(self, the_for_scope: tn.ForScope) -> int:
        merged = 0

        if is_axis_for(the_for_scope, AxisIterator._K):
            # TODO: if the for scope is on a cartesian axis it can be
            # merged with other for scope going in the same direction
            pass
        else:
            # Non-cartesian for - can be pushed down if everything merged below
            if (
                len(the_for_scope.children) == 1
                and isinstance(the_for_scope.children[0], tn.MapScope)
                and is_axis_map(the_for_scope.children[0], self.axis)
            ):
                swap_node_position_in_tree(the_for_scope, the_for_scope.children[0])
                merged += 1

        return merged + self._default_control_flow(the_for_scope)

    def _default_control_flow(self, the_control_flow: tn.ControlFlowScope) -> int:
        if len(the_control_flow.children) != 0:
            return self._merge(the_control_flow)

        return 0

    def _map_overcompute_merge(
        self, the_map: tn.MapScope, nodes: list[tn.ScheduleTreeNode]
    ) -> int:
        # End of nodes OR
        # Not the right axis
        # --> recurse
        if is_last_node(the_map, nodes) or not is_axis_map(the_map, self.axis):
            merged = 0
            for child in the_map.children:
                merged += self._merge_node(child, the_map.children)
            return merged

        next_node = get_next_node(the_map, nodes)

        # Next node is not a MapScope - no merge
        if not isinstance(next_node, tn.MapScope):
            return 0

        # Attempt to merge consecutive maps
        if not _both_same_single_axis_maps(the_map, next_node, self.axis):
            return 0

        if not _can_merge_axis_maps(the_map, next_node, self.axis):
            self.failed_due_to_data_dep += 1
            return 0

        # Over compute to merge:
        # - force-merge by expanding the ranges
        first_map = the_map
        second_map = next_node
        first_range = first_map.node.map.range
        second_range = second_map.node.map.range
        merged_range = dace.subsets.Range(
            [
                (
                    f"min({first_range.ranges[0][0]}, {second_range.ranges[0][0]})",
                    f"max({first_range.ranges[0][1]}, {second_range.ranges[0][1]})",
                    1,  # NOTE: we can optimize this to gcd later
                )
            ]
        )

        # only overcompute if configured - otherwise no merge
        if not self.overcompute and (
            first_range != merged_range or second_range != merged_range
        ):
            return 0

        # Process range, with overcompute guard
        # for children to only run in their respective range
        axis_as_str = first_map.node.map.params[0]
        assert isinstance(axis_as_str, str)
        InsertOvercomputationGuard(
            axis_as_str,
            merged_range=merged_range,
            original_range=first_range,
            hint=self.hint,
        ).visit(first_map)
        InsertOvercomputationGuard(
            axis_as_str,
            merged_range=merged_range,
            original_range=second_range,
            hint=self.hint,
        ).visit(second_map)
        assert isinstance(first_map, tn.MapScope)
        assert isinstance(second_map, tn.MapScope)
        first_map.node.map.range = merged_range

        # Merge children
        merged_children: list[tn.ScheduleTreeNode] = [
            *first_map.children,
            *second_map.children,
        ]
        first_map.children = merged_children
        for child in merged_children:
            child.parent = first_map

        # K-maps use unique iterators (i.e. every k-map iterates over `k__[0-9]*`).
        # After merge, we need to replace the axis symbols of the second map's children
        # with the axis symbol of the first map.
        if second_map.node.map.params[0] != first_map.node.map.params[0]:
            replacements = {second_map.node.map.params[0]: first_map.node.map.params[0]}
            ReplaceAxisSymbol(replacements).visit(first_map)

        # delete now-merged second_map
        del nodes[list_index(second_map, nodes)]

        return 1

    def _merge(self, node: tn.ScheduleTreeScope) -> int:
        merged = 0
        tn.validate_children_and_parents_align(node)

        if __debug__:
            detect_cycle(node.children, set())

        i_candidate = 0
        while i_candidate < len(node.children):
            next_node = node.children[i_candidate]
            merged += self._merge_node(next_node, node.children)
            tn.validate_children_and_parents_align(node)
            i_candidate += 1

        if __debug__:
            detect_cycle(node.children, set())

        return merged

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        """Merge consecutive maps of the same cartesian axis.

        Dev NOTE: this pass has been implemented in coordination with the rest of the passes
        deployed in `CartesianMerge`. Some of the other passes aim to surface maps in a way to maximize
        the efficiency of this pass.

        The algorithm works as follows:
            - Try to merge the surfaced maps
            - When done, count the number of actual merges
            - If NO merges - restore the previous children
            (undo potential changes that didn't lead to map merge)
            - If merges - go again since we have modified the tree
            Then exit.

        ToDo:
            - ForLoop are not merge at the moment, only Maps.
            - Non-cartesian ForLoop could be merged down _if_ the maps below
            are unique (e.g. if everything has been merged) and the hint is PARALLEL.
            This is relevant for linear solvers and other iteration-dependent algorithmics
        """
        tn.validate_children_and_parents_align(node)
        overall_merged = 0
        passes_apply = 0
        i = 0
        while True:
            i += 1
            previous_children = copy.deepcopy(node.children)
            merged = self._merge(node)
            overall_merged += merged
            if __debug__:
                detect_cycle(node.children, set())

            # If we didn't merge, we revert the children
            # to the previous state
            if merged == 0:
                node.children = previous_children
                for child in node.children:
                    child.parent = node
                break
            passes_apply += 1

        tn.validate_has_no_other_node_types(node)
        tn.validate_children_and_parents_align(node)

        ndsl_log.debug(
            f"🚀 {self}: {overall_merged} maps merged in {passes_apply} passes "
            f"({self.failed_due_to_data_dep} merge blocked because of data dependency)"
        )
