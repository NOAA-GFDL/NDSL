import dace
import dace.sdfg.analysis.schedule_tree.treenodes as dace_stree
from dace.properties import CodeBlock

from typing import Any, List

from ndsl.dsl.dace.stree.optimizations.memlet_helpers import (
    no_data_dependencies_on_cartesian_axis,
    AxisIterator,
)


def _is_axis_map(node: dace_stree.MapScope, axis: AxisIterator) -> bool:
    """Returns true iff node is a map over K."""
    map_parameter = node.node.params
    return len(map_parameter) == 1 and map_parameter[0] == axis.as_str()


def _both_same_axis_maps(
    first: dace_stree.MapScope,
    second: dace_stree.MapScope,
    axis: AxisIterator,
) -> bool:
    return _is_axis_map(first, axis) and _is_axis_map(second, axis)


def _can_merge_axis_maps(
    first: dace_stree.MapScope,
    second: dace_stree.MapScope,
    axis: AxisIterator,
) -> bool:
    return _both_same_axis_maps(
        first, second, axis
    ) and no_data_dependencies_on_cartesian_axis(
        first,
        second,
        axis,
    )


def _try_to_push_scalar_tasklet_into_next_map(
    first_map: dace_stree.MapScope,
    second_tasklet: dace_stree.TaskletNode,
    axis: AxisIterator,
    nodes: List[dace_stree.ScheduleTreeNode],
) -> dace_stree.MapScope | None:
    """Attempt to push a inputless tasklet (scalar) inside the maps
    if the next node is a mergeable Map.

    Return: new MapScope to merge, if mergeable
    """
    if len(second_tasklet.input_memlets()) != 0:
        return None  # Tasklet is not inputless
    second_index = nodes.index(second_tasklet)
    if second_index == len(nodes):
        return None  # Last node - done
    second_map = nodes[second_index + 1]
    if not isinstance(second_map, dace_stree.MapScope):
        return None  # Next node is not a MapScope
    if not _can_merge_axis_maps(first_map, second_map, axis):
        return None  # Next map is not mergeable

    # Push the tasklet into the map, remove from node list
    first_map.children.insert(0, second_tasklet)
    second_tasklet.parent = first_map
    nodes.remove(second_tasklet)
    return second_map


def _try_push_ifelse_into_child_map(
    first_map: dace_stree.MapScope,
    second_ifscope: dace_stree.IfScope,
    axis: AxisIterator,
    nodes: List[dace_stree.ScheduleTreeNode],
) -> dace_stree.MapScope | None:
    """Attempt to push a if/else inside the maps if the first child
    is a mergeable Map.

    Return: new MapScope to merge, if mergeable
    """

    print("WIP if/else")
    return None

    if len(second_ifscope.children) == 0:
        return None
    child_map = second_ifscope.children[0]
    if isinstance(child_map, dace_stree.IfScope):
        # Find first map through N ifs
        while isinstance(child_map, dace_stree.IfScope):
            if len(child_map.children) == 0:
                child_map = None
            else:
                child_map = child_map.children[0]
        if child_map is None or not isinstance(child_map, dace_stree.MapScope):
            return None
    if not isinstance(child_map, dace_stree.MapScope):
        return None
    if not _can_merge_axis_maps(first_map, child_map, axis):
        return None

    # TODO: do the else
    # ifscope_index = nodes.index(second_ifscope)
    # if ifscope_index + 1 > len(nodes):

    # Push the If/Else into the child map

    # old_child_index = child_map.parent.children.index(child_map)
    # child_map.parent.children.insert(old_child_index, old_children)
    # child_map.parent.children.remove(child_map)

    return child_map


class InsertOvercomputationGuard(dace_stree.ScheduleNodeTransformer):
    def __init__(
        self,
        axis: AxisIterator,
        *,
        merged_range: dace.subsets.Range,
        original_range: dace.subsets.Range,
    ):
        self._axis = axis
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
            f"{self._axis.as_str()} >= {start} and {self._axis.as_str()} <= {stop} and ({self._axis.as_str()} - {start}) % {step} == 0"
        )

    def visit_MapScope(self, node: dace_stree.MapScope) -> dace_stree.MapScope:
        all_children_are_maps = all(
            [isinstance(child, dace_stree.MapScope) for child in node.children]
        )
        if not all_children_are_maps:
            if self._merged_range != self._original_range:
                node.children = [
                    dace_stree.IfScope(
                        condition=self._execution_condition(), children=node.children
                    )
                ]
            return node

        node.children = self.visit(node.children)
        return node


class CartesianAxisMerge(dace_stree.ScheduleNodeTransformer):
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

    def _merge(self, children: List[dace_stree.ScheduleTreeNode]) -> bool:
        # count number of maps in children
        map_scopes = [
            map_scope
            for map_scope in children
            if isinstance(map_scope, dace_stree.MapScope)
        ]

        if not map_scopes:
            # stop the recursion
            return children

        if len(map_scopes) == 1:
            map_scope = map_scopes[0]
            map_index = children.index(map_scope)

            # recurse deeper, see if we can merge more maps
            children[map_index].children = self._merge(map_scope.children)
            return children

        # We have at least two maps at this level. Attempt to merge consecutive maps
        i = 0
        while i < len(children):
            first_map = children[i]

            # skip all non-maps
            if not isinstance(first_map, dace_stree.MapScope):
                i += 1
                continue

            j = i + 1
            while j < len(children):
                second_map = children[j]

                # Non-maps node - we need to keep code-flow correct, so we go to the next
                # available candidate
                if not isinstance(second_map, dace_stree.MapScope):
                    # TODO: for if/else
                    if isinstance(second_map, dace_stree.TaskletNode):
                        second_map = _try_to_push_scalar_tasklet_into_next_map(
                            first_map, second_map, self.axis, children
                        )
                    elif isinstance(second_map, dace_stree.IfScope):
                        second_map = _try_push_ifelse_into_child_map(
                            first_map, second_map, self.axis, children
                        )
                    else:
                        second_map = None

                    # Check if we can merge or get out
                    can_merge = second_map is not None
                    if not can_merge:
                        i += 1
                        break
                else:
                    can_merge = _can_merge_axis_maps(first_map, second_map, self.axis)

                if can_merge:
                    print(
                        f"{self.axis.name} merge: {first_map.node.map.params} in {first_map.node.map.range}"
                    )
                    # Only for maps in K:
                    # force-merge by expanding the ranges
                    # then, guard children to only run in their respective range
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

                    # push IfScope down if children are just maps
                    first_map = InsertOvercomputationGuard(
                        self.axis, merged_range=merged_range, original_range=first_range
                    ).visit(first_map)
                    second_map = InsertOvercomputationGuard(
                        self.axis,
                        merged_range=merged_range,
                        original_range=second_range,
                    ).visit(second_map)
                    merged_children: List[dace_stree.MapNode] = [
                        *first_map.children,
                        *second_map.children,
                    ]
                    first_map.children = merged_children

                    # TODO also merge containers and symbols (if applicable)

                    # TODO Question: is this all it needs?
                    first_map.node.map.range = merged_range

                    # delete now-merged second_map
                    del children[j]

                    # recurse into children
                    first_map.children = self._merge(first_map.children)
                else:
                    # we couldn't merge, try the next consecutive pair
                    i += 1

                # break out of the inner while loop
                break

            # in case we merged everything ...
            if j >= len(children):
                i += 1

        return children

    def visit_ScheduleTreeRoot(self, node: dace_stree.ScheduleTreeRoot):
        node.children = self._merge(node.children)
