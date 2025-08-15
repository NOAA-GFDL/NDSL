from __future__ import annotations

from abc import ABC, abstractmethod
import dace
import dace.sdfg.analysis.schedule_tree.treenodes as dst
from dace.properties import CodeBlock

from typing import List

from ndsl.dsl.dace.stree.optimizations.memlet_helpers import (
    no_data_dependencies_on_cartesian_axis,
    AxisIterator,
)


def _is_axis_map(node: dst.MapScope, axis: AxisIterator) -> bool:
    """Returns true iff node is a map over K."""
    map_parameter = node.node.params
    return len(map_parameter) == 1 and map_parameter[0] == axis.as_str()


def _both_same_axis_maps(
    first: dst.MapScope,
    second: dst.MapScope,
    axis: AxisIterator,
) -> bool:
    return _is_axis_map(first, axis) and _is_axis_map(second, axis)


def _can_merge_axis_maps(
    first: dst.MapScope,
    second: dst.MapScope,
    axis: AxisIterator,
) -> bool:
    return _both_same_axis_maps(
        first, second, axis
    ) and no_data_dependencies_on_cartesian_axis(
        first,
        second,
        axis,
    )


class InsertOvercomputationGuard(dst.ScheduleNodeTransformer):
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


class MapRemover(dst.ScheduleNodeTransformer):
    def __init__(self, maps_to_remove: list[dst.MapScope]) -> None:
        self._maps_to_remove = maps_to_remove

    def visit_MapScope(self, node: dst.MapScope) -> dst.MapScope | None:
        if node.parent is None:
            return node

        my_idx = node.parent.children.index(node)
        for child in node.children:
            child.parent = node.parent
        node.parent.children = (
            node.parent.children[0:my_idx]
            + node.children
            + node.parent.children[my_idx + 1 :]
        )
        return None


class AxisMergeOp(ABC):
    def __init__(
        self,
        the_map: dst.MapScope,
        node: dst.ScheduleTreeNode,
        axis: AxisIterator,
        verbose: bool = False,
    ) -> None:
        self.the_map = the_map
        self.node_to_merge = node
        self.axis = axis
        self.verbose = verbose

    @abstractmethod
    def can_merge(
        self,
        nodes: List[dst.ScheduleTreeNode],
    ) -> bool:
        """Can we merge the node."""
        ...

    @abstractmethod
    def apply(
        self,
        nodes: List[dst.ScheduleTreeNode],
    ) -> bool:
        """Apply changes leading to merge, return if we can keep
        merging or if we should start fresh."""
        ...


class AxisMergeOp_TasletScalar(AxisMergeOp):
    def __init__(
        self,
        the_map: dst.MapScope,
        node: dst.ScheduleTreeNode,
        axis: AxisIterator,
    ) -> None:
        super().__init__(the_map, node, axis)

    def can_merge(
        self,
        nodes: List[dst.ScheduleTreeNode],
    ) -> bool:
        if len(self.node_to_merge.input_memlets()) != 0:
            return False  # Tasklet is not inputless
        second_index = nodes.index(self.node_to_merge)
        if second_index == len(nodes):
            return False  # Last node - done
        second_map = nodes[second_index + 1]
        if not isinstance(second_map, dst.MapScope):
            return False  # Next node is not a MapScope
        if not _can_merge_axis_maps(self.the_map, second_map, self.axis):
            return False  # Next map is not mergeable

        return True

    def apply(
        self,
        nodes: List[dst.ScheduleTreeNode],
    ) -> bool:
        # Push the tasklet into the map, remove from node list
        print(" Merge K map: Push inputless tasklet down into Map")
        self.the_map.children.insert(0, self.node_to_merge)
        self.node_to_merge.parent = self.the_map
        nodes.remove(self.node_to_merge)
        return True


class AxisMergeOp_IfElse(AxisMergeOp):
    def __init__(
        self,
        the_map: dst.MapScope,
        node: dst.ScheduleTreeNode,
        axis: AxisIterator,
    ) -> None:
        super().__init__(the_map, node, axis)

    def can_merge(
        self,
        nodes: List[dst.ScheduleTreeNode],
    ) -> bool:
        return False
        if len(self.node_to_merge.input_memlets()) != 0:
            return False  # Tasklet is not inputless
        second_index = nodes.index(self.node_to_merge)
        if second_index == len(nodes):
            return False  # Last node - done
        second_map = nodes[second_index + 1]
        if not isinstance(second_map, dst.MapScope):
            return False  # Next node is not a MapScope
        if not _can_merge_axis_maps(self.the_map, second_map, self.axis):
            return False  # Next map is not mergeable

        return True

    def apply(
        self,
        the_map: dst.MapScope,
        tasklet: dst.TaskletNode,
        axis: AxisIterator,
        nodes: List[dst.ScheduleTreeNode],
    ) -> bool:
        return False
        child_map.children = [second_ifscope]
        MapRemover([child_map]).visit(second_ifscope)


class AxisMergeOp_OvercomputeMerge(AxisMergeOp):
    def __init__(
        self,
        the_map: dst.MapScope,
        node: dst.MapScope,
        axis: AxisIterator,
    ) -> None:
        super().__init__(the_map, node, axis)

    def can_merge(
        self,
        nodes: List[dst.ScheduleTreeNode],
    ) -> bool:
        return _can_merge_axis_maps(self.the_map, self.node_to_merge, self.axis)

    def apply(
        self,
        nodes: List[dst.ScheduleTreeNode],
    ) -> bool:
        # Only for maps in K:
        # force-merge by expanding the ranges
        # then, guard children to only run in their respective range
        first_range = self.the_map.node.map.range
        second_range = self.node_to_merge.node.map.range
        merged_range = dace.subsets.Range(
            [
                (
                    f"min({first_range.ranges[0][0]}, {second_range.ranges[0][0]})",
                    f"max({first_range.ranges[0][1]}, {second_range.ranges[0][1]})",
                    1,  # NOTE: we can optimize this to gcd later
                )
            ]
        )

        print(f" Merge K map: {first_range} ⋃ {second_range} -> {merged_range}")

        # push IfScope down if children are just maps
        first_map = InsertOvercomputationGuard(
            self.axis, merged_range=merged_range, original_range=first_range
        ).visit(self.the_map)
        second_map = InsertOvercomputationGuard(
            self.axis,
            merged_range=merged_range,
            original_range=second_range,
        ).visit(self.node_to_merge)
        merged_children: List[dst.MapScope] = [
            *first_map.children,
            *second_map.children,
        ]
        first_map.children = merged_children

        # TODO also merge containers and symbols (if applicable)

        first_map.node.map.range = merged_range

        # delete now-merged second_map
        del nodes[nodes.index(self.node_to_merge)]

        return True


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
        self.merged = 0

    def __str__(self) -> str:
        return f"CartesianAxisMerge({self.axis.name})"

    def _get_operator(
        self,
        the_map: dst.MapScope,
        candidate: dst.ScheduleTreeNode,
        nodes: List[dst.ScheduleTreeNode],
    ) -> AxisMergeOp | None:
        if isinstance(candidate, dst.MapScope):
            return AxisMergeOp_OvercomputeMerge(the_map, candidate, self.axis)
        elif isinstance(candidate, dst.IfScope):
            return AxisMergeOp_IfElse(the_map, candidate, self.axis)
        elif isinstance(candidate, dst.TaskletNode):
            return AxisMergeOp_TasletScalar(the_map, candidate, self.axis)

        return None

    def _merge(self, node: dst.ScheduleTreeRoot) -> bool:
        # Step 1: Find a candidate
        map_scopes = [
            map_scope
            for map_scope in node.children
            if isinstance(map_scope, dst.MapScope)
        ]

        if not map_scopes:
            # stop the recursion
            return False

        # Recursion would be good - but it break the 3-step algorithmics
        # if len(map_scopes) == 1:
        #     map_scope = map_scopes[0]
        #     map_index = node.children.index(map_scope)

        #     # recurse deeper, see if we can merge more maps
        #     node.children[map_index].children = self._merge_2(map_scope.children)
        #     return children

        i_candidate = 0
        the_map = None
        while i_candidate < len(node.children) - 1:  # Skip last child
            first_map = node.children[i_candidate]

            # skip all non-maps
            if not isinstance(first_map, dst.MapScope):
                i_candidate += 1
                continue

            op = self._get_operator(
                first_map, node.children[i_candidate + 1], node.children
            )
            if op is not None and op.can_merge(node.children):
                the_map = first_map
                break

            i_candidate += 1

        if the_map is None:
            return False

        # Step 2: Mark required operation as far as we can
        i_to_merge = i_candidate + 1
        operations: list[AxisMergeOp] = []
        while i_to_merge < len(node.children):
            op = self._get_operator(
                the_map,
                node.children[i_to_merge],
                node.children,
            )
            if op is not None and op.can_merge(node.children):
                operations.append(op)
                i_to_merge += 1
                continue

            break

        assert len(operations) >= 1

        # Step 3: Commit changes marked above
        for op in operations:
            keep_merging = op.apply(node.children)
            if not keep_merging:
                return True

        # We have comitted changes, flag for merge again
        return True

    def visit_ScheduleTreeRoot(self, node: dst.ScheduleTreeRoot):
        # node.children = self._merge(node.children)
        commit = self._merge(node)
        while commit:
            commit = self._merge(node)

        print(f"Cartesian Axis Merge ({self.axis.name}): {self.merged} map merged")
