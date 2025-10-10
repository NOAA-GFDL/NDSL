from __future__ import annotations

import copy
import re
from typing import Any, List

import dace
import dace.sdfg.analysis.schedule_tree.treenodes as dst
from dace.properties import CodeBlock

from ndsl import ndsl_log
from ndsl.dsl.dace.stree.optimizations.memlet_helpers import (
    AxisIterator,
    no_data_dependencies_on_cartesian_axis,
)
from ndsl.dsl.dace.stree.optimizations.tree_common_op import (
    detect_cycle,
    list_index,
    swap_node_position_in_tree,
)


def _is_axis_map(node: dst.MapScope, axis: AxisIterator) -> bool:
    """Returns true if node is a map over K."""
    map_parameter = node.node.params
    return len(map_parameter) == 1 and map_parameter[0].startswith(axis.as_str())


def _both_same_single_axis_maps(
    first: dst.MapScope,
    second: dst.MapScope,
    axis: AxisIterator,
) -> bool:
    return (
        (len(first.node.params) == 1 and len(second.node.params) == 1)  # Single axis
        and first.node.params[0] == second.node.params[0]  # Same axis
        and _is_axis_map(first, axis)  # Correct axis
    )


def _can_merge_axis_maps(
    first: dst.MapScope,
    second: dst.MapScope,
    axis: AxisIterator,
) -> bool:
    return _both_same_single_axis_maps(
        first, second, axis
    ) and no_data_dependencies_on_cartesian_axis(
        first,
        second,
        axis,
    )


class InsertOvercomputationGuard(dst.ScheduleNodeTransformer):
    def __init__(
        self,
        axis_as_string: str,
        *,
        merged_range: dace.subsets.Range,
        original_range: dace.subsets.Range,
    ):
        self._axis_as_string = axis_as_string
        self._merged_range = merged_range
        self._original_range = original_range

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

    def visit_MapScope(self, node: dst.MapScope) -> dst.MapScope:
        all_children_are_maps = all(
            [isinstance(child, dst.MapScope) for child in node.children]
        )
        if not all_children_are_maps:
            if self._merged_range != self._original_range:
                node.children = [
                    dst.IfScope(
                        condition=self._execution_condition(), children=node.children
                    )
                ]
            return node

        node.children = self.visit(node.children)
        return node


def _get_next_node(
    nodes: list[dst.ScheduleTreeNode],
    node: dst.ScheduleTreeNode,
) -> dst.ScheduleTreeNode:
    return nodes[list_index(nodes, node) + 1]


def _last_node(nodes: list[dst.ScheduleTreeNode], node: dst.ScheduleTreeNode) -> bool:
    return list_index(nodes, node) >= len(nodes) - 1


def _sanitize_axis(axis: AxisIterator, name_to_normalize: str) -> str:
    axis_clean = f"{axis.as_str()}"
    pattern = f"{axis.as_str()}_[0-9]*"

    return re.sub(pattern, axis_clean, name_to_normalize)


class NormalizeAxisSymbol(dst.ScheduleNodeVisitor):
    def __init__(self, axis: AxisIterator) -> None:
        self.axis = axis

    def visit_MapScope(
        self,
        map_scope: dst.MapScope,
        axis_rpl_dict: dict[str, str] = {},
        **kwargs: Any,
    ) -> None:
        for i_param, param in enumerate(map_scope.node.params):
            sanitized_param = _sanitize_axis(self.axis, param)
            axis_rpl_dict[param] = sanitized_param
            map_scope.node.params[i_param] = sanitized_param

        # visit children
        for child in map_scope.children:
            self.visit(child, axis_rpl_dict=axis_rpl_dict)

    def visit_TaskletNode(
        self,
        node: dst.TaskletNode,
        axis_rpl_dict: dict[str, str] = {},
        **kwargs: Any,
    ) -> None:
        for memlets in node.in_memlets.values():
            memlets.replace(axis_rpl_dict)
        for memlets in node.out_memlets.values():
            memlets.replace(axis_rpl_dict)


class CartesianAxisMerge(dst.ScheduleNodeTransformer):
    """Merge a cartesian axis if they are contiguous in code-flow.

    Can do:
        - merge a given axis with the next maps at the same recursion level
        - can overcompute (eager) do allow for more merging at a cost of an if

    Args:
        axis: AxisIterator to be merged
        eager: overcompute with a conditional guard
    """

    def __init__(
        self,
        axis: AxisIterator,
        *,
        eager: bool = True,
    ) -> None:
        self.axis = axis
        self.eager = eager

    def __str__(self) -> str:
        return f"CartesianAxisMerge({self.axis.name})"

    def _merge_node(
        self,
        node: dst.ScheduleTreeNode,
        nodes: list[dst.ScheduleTreeNode],
    ) -> int:
        """Direct code to the correct resolver for the node (e.g. visitor)

        Dev Note: Order matters!
            Default behavior for base class must be _after_ bespoke leaf class
            behavior (e.g. IfScope before ControlFlowScope)
        """

        if isinstance(node, dst.MapScope):
            return self._map_overcompute_merge(node, nodes)
        elif isinstance(node, dst.IfScope):
            return self._push_ifelse_down(node, nodes)
        elif isinstance(node, dst.TaskletNode):
            return self._push_tasklet_down(node, nodes)
        elif isinstance(node, dst.ControlFlowScope):
            return self._default_control_flow(node, nodes)
        else:
            ndsl_log.debug(
                f"  (╯°□°)╯︵ ┻━┻: can't merge {type(node)}. Recursion ends."
            )
        return 0

    def _default_control_flow(
        self,
        the_control_flow: dst.ControlFlowScope,
        nodes: list[dst.ScheduleTreeNode],
    ) -> int:
        if len(the_control_flow.children) != 0:
            return self._merge(the_control_flow)

        return 0

    def _push_tasklet_down(
        self,
        the_tasklet: dst.TaskletNode,
        nodes: list[dst.ScheduleTreeNode],
    ) -> int:
        """Push tasklet into a consecutive map."""
        in_memlets = the_tasklet.input_memlets()
        if len(in_memlets) != 0:
            if "__pystate" in [tasklet.data for tasklet in the_tasklet.input_memlets()]:
                return 0  # Tasklet is a callback

        next_index = list_index(nodes, the_tasklet)
        if next_index == len(nodes):
            return 0  # Last node - done

        next_node = nodes[next_index + 1]

        # Before checking the possibility of merging - attempt to surface
        # a map from the next nodes
        merged = self._merge_node(next_node, nodes)

        # Attempt to push the tasklet in the next map
        ndsl_log.debug("  Push tasklet down into next map")
        next_node = nodes[next_index + 1]
        if isinstance(next_node, dst.MapScope):
            next_node.children.insert(0, the_tasklet)
            the_tasklet.parent = next_node
            nodes.remove(the_tasklet)
            merged += self._merge_node(next_node, nodes)

        return merged

    def _push_ifelse_down(
        self,
        the_if: dst.IfScope,
        nodes: list[dst.ScheduleTreeNode],
    ) -> int:
        merged = 0

        # Recurse down if/else/elif
        if_index = list_index(nodes, the_if)
        if len(the_if.children) != 0:
            merged += self._merge_node(the_if.children[0], the_if.children)
        for else_index in range(if_index + 1, len(nodes)):
            else_node = nodes[else_index]
            if else_index < len(nodes) and (
                isinstance(else_node, dst.ElseScope)
                or isinstance(else_node, dst.ElifScope)
            ):
                merged += self._merge_node(else_node, else_node.children)
            else:
                break

        # Look at swapping if/else/elif first map w/ control flow

        # Gather all first maps - if they do not exists, get out
        all_maps = []
        if isinstance(the_if.children[0], dst.MapScope):
            all_maps.append(the_if.children[0])
        else:
            return merged
        for else_index in range(if_index + 1, len(nodes)):
            else_node = nodes[else_index]
            if else_index < len(nodes) and (
                isinstance(else_node, dst.ElseScope)
                or isinstance(else_node, dst.ElifScope)
            ):
                if isinstance(else_node.children[0], dst.MapScope):
                    all_maps.append(else_node.children[0])
                else:
                    return merged
            else:
                break

        # Check for mergeability
        if len(all_maps) > 1:
            the_map = all_maps[0]
            for _map in all_maps[1:]:
                if not _can_merge_axis_maps(the_map, _map, self.axis):
                    return merged

        # We are good to go - swap it all
        ndsl_log.debug(f"  Push IF {the_if.condition.as_string} down")
        inner_if_map = the_if.children[0]

        # Swap IF & maps
        if_index = list_index(nodes, the_if)
        swap_node_position_in_tree(the_if, inner_if_map)

        # Swap ELIF/ELSE & maps
        for else_index in range(if_index + 1, len(nodes)):
            if else_index < len(nodes) and (
                isinstance(nodes[else_index], dst.ElseScope)
                or isinstance(nodes[else_index], dst.ElifScope)
            ):
                swap_node_position_in_tree(
                    nodes[else_index], nodes[else_index].children[0]
                )
            else:
                break

        # Merge the Maps
        assert isinstance(nodes[if_index], dst.MapScope)
        merged += self._map_overcompute_merge(nodes[if_index], nodes)

        return merged

    def _map_overcompute_merge(
        self,
        the_map: dst.MapScope,
        nodes: list[dst.ScheduleTreeNode],
    ) -> int:
        if _last_node(nodes, the_map):
            return 0

        next_node = _get_next_node(nodes, the_map)

        # If we the next node is not a MapScope - recurse
        if not isinstance(next_node, dst.MapScope):
            merged = self._merge_node(next_node, nodes)
            new_next_node = _get_next_node(nodes, the_map)
            if new_next_node == next_node:
                return merged
            return merged + self._merge_node(the_map, nodes)

        # Attempt to merge consecutive maps
        if not _can_merge_axis_maps(the_map, next_node, self.axis):
            return 0

        # Only for maps in K:
        # - force-merge by expanding the ranges
        # - then, guard children to only run in their respective range
        first_range = the_map.node.map.range
        second_range = next_node.node.map.range
        merged_range = dace.subsets.Range(
            [
                (
                    f"min({first_range.ranges[0][0]}, {second_range.ranges[0][0]})",
                    f"max({first_range.ranges[0][1]}, {second_range.ranges[0][1]})",
                    1,  # NOTE: we can optimize this to gcd later
                )
            ]
        )

        ndsl_log.debug(
            f"  Merge K map: {first_range} ⋃ {second_range} -> {merged_range}"
        )

        # push IfScope down if children are just maps
        axis_as_str = the_map.node.params[0]
        first_map = InsertOvercomputationGuard(
            axis_as_str, merged_range=merged_range, original_range=first_range
        ).visit(the_map)
        second_map = InsertOvercomputationGuard(
            axis_as_str,
            merged_range=merged_range,
            original_range=second_range,
        ).visit(next_node)
        merged_children: List[dst.MapScope] = [
            *first_map.children,
            *second_map.children,
        ]
        first_map.children = merged_children

        # TODO also merge containers and symbols (if applicable)

        first_map.node.map.range = merged_range

        # delete now-merged second_map
        del nodes[list_index(nodes, next_node)]

        return 1

    def _merge(self, node: dst.ScheduleTreeRoot | dst.ScheduleTreeScope) -> int:
        merged = 0

        if __debug__:
            detect_cycle(node.children, set())

        i_candidate = 0
        while i_candidate < len(node.children):
            next_node = node.children[i_candidate]
            merged += self._merge_node(next_node, node.children)
            i_candidate += 1

        if __debug__:
            detect_cycle(node.children, set())

        return merged

    def visit_ScheduleTreeRoot(self, node: dst.ScheduleTreeRoot):
        """Merge as many maps as possible.

        The algorithm works as follows:
            - Start merging - move nodes to surface maps as much as possible
            - Try to merge the surfaced maps
            - When done, count the number of actual merges
            - If NO merges - restore the previous children
            (undo potential changes that didn't lead to map merge)
            Then exit.
        """

        # TODO: many interval generate many iterator name right now
        #    e.g. _k_0, _k_1...
        # This makes merging more difficult. We could write a pre-pass
        # that cleans this up BUT we have an issue with the THIS_K feature
        # in the tasklet...
        # NormalizeAxisSymbol(self.axis).visit(node)

        overall_merged = 0
        i = 0
        while True:
            i += 1
            ndsl_log.debug(f"🔥 Merge attempt #{i}")
            previous_children = copy.deepcopy(node.children)
            try:
                merged = self._merge(node)
                overall_merged += merged
                if __debug__:
                    detect_cycle(node.children, set())
            except RecursionError as re:
                raise re

            # If we didn't merge, we revert the children
            # to the previous state
            if merged == 0:
                ndsl_log.debug("🥹 No merges, revert!")
                node.children = previous_children
                break

        ndsl_log.debug(
            f"🚀 Cartesian Axis Merge ({self.axis.name}): {overall_merged} map merged"
        )
