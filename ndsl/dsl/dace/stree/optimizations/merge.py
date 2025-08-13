import dace
import dace.sdfg.analysis.schedule_tree.treenodes as dace_stree
from dace.memlet import Memlet
from dace.properties import CodeBlock

from enum import Enum
from typing import List


def is_k_map(node: dace_stree.MapScope) -> bool:
    """Returns true iff node is a map over K."""
    map_parameter = node.node.params
    return len(map_parameter) == 1 and map_parameter[0] == "__k"


def both_k_maps(first: dace_stree.MapScope, second: dace_stree.MapScope) -> bool:
    return is_k_map(first) and is_k_map(second)


def no_data_dependencies(
    first: dace_stree.MapScope,
    second: dace_stree.MapScope,
    restrict_check_to_k=False,
) -> bool:
    write_collector = MemletCollector(collect_reads=False)
    write_collector.visit(first)
    read_collector = MemletCollector(collect_writes=False)
    read_collector.visit(second)
    for write in write_collector.out_memlets:
        # Make sure we don't have read after write conditions.
        # TODO: this can be optimized to allow non-overlapping intervals and such in the future
        if restrict_check_to_k:
            if write.subset.dims() < 3:
                # Case of 2D write - no K dependency
                continue

            previous_k_index = write.subset[2][0]
            for read in read_collector.in_memlets:
                if write.data == read.data:
                    if previous_k_index != read.subset[2][0]:
                        print(
                            "[K Merge] Found read after write conflict "
                            f"for {write.data} "
                            "w/ different offset to K ("
                            f"write at {read.subset[2][0]}, "
                            f"read at {read.subset[2][0]})"
                        )
                        return False

        else:
            if write.data in [read.data for read in read_collector.in_memlets]:
                print(
                    f"[All dims merge] Found potential read after write conflict for {write.data}"
                )
                return False
    return True


def has_dynamic_memlets(
    first: dace_stree.MapScope, second: dace_stree.MapScope
) -> bool:
    first_collector = MemletCollector()
    second_collector = MemletCollector()
    first_collector.visit(first)
    second_collector.visit(second)
    has_dynamic_memlets = any(
        [
            memlet.dynamic
            for memlet in [
                *first_collector.in_memlets,
                *first_collector.out_memlets,
                *second_collector.in_memlets,
                *second_collector.out_memlets,
            ]
        ]
    )
    return has_dynamic_memlets


class MergeStrategy(Enum):
    NoOp = 0
    Trivial = 1
    Force_K = 2


class MemletCollector(dace_stree.ScheduleNodeVisitor):
    """Gathers in_memlets and out_memlets of TaskNodes and LibraryCalls."""

    in_memlets: list[Memlet]
    out_memlets: list[Memlet]

    def __init__(self, *, collect_reads=True, collect_writes=True):
        self._collect_reads = collect_reads
        self._collect_writes = collect_writes

        self.in_memlets = []
        self.out_memlets = []

    def visit_TaskletNode(self, node: dace_stree.TaskletNode) -> None:
        if self._collect_reads:
            self.in_memlets.extend([memlet for memlet in node.in_memlets.values()])
        if self._collect_writes:
            self.out_memlets.extend([memlet for memlet in node.out_memlets.values()])

    def visit_LibraryCall(self, node: dace_stree.LibraryCall) -> None:
        if self._collect_reads:
            if isinstance(node.in_memlets, set):
                self.in_memlets.extend(node.in_memlets)
            else:
                assert isinstance(node.in_memlets, dict)
                self.in_memlets.extend([memlet for memlet in node.in_memlets.values()])

        if self._collect_writes:
            if isinstance(node.out_memlets, set):
                self.out_memlets.extend(node.out_memlets)
            else:
                assert isinstance(node.out_memlets, dict)
                self.out_memlets.extend(
                    [memlet for memlet in node.out_memlets.values()]
                )


class PushDownIfStatement(dace_stree.ScheduleNodeTransformer):
    def __init__(
        self, *, merged_range: dace.subsets.Range, original_range: dace.subsets.Range
    ):
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
            f"__k >= {start} and k <= {stop} and (__k - {start}) % {step} == 0"
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


class MapMerge(dace_stree.ScheduleNodeTransformer):
    def __init__(
        self,
        *,
        merge_strategy: MergeStrategy = MergeStrategy.Trivial,
        allow_dynamic_memlets: bool = False,
    ) -> None:
        self.merge_strategy = merge_strategy
        self.allow_dynamic_memlets = allow_dynamic_memlets

    def _merge_maps(self, children: List[dace_stree.ScheduleTreeNode]):
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
            children[map_index].children = self._merge_maps(map_scope.children)
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

                # skip all non-maps
                if not isinstance(second_map, dace_stree.MapScope):
                    j += 1
                    continue

                equal_map_params = first_map.node.params == second_map.node.params
                equal_map_ranges = first_map.node.map.range == second_map.node.map.range
                trivial_merge = (
                    self.merge_strategy != MergeStrategy.NoOp
                    and equal_map_params
                    and equal_map_ranges
                    and no_data_dependencies(first_map, second_map)
                    and (
                        self.allow_dynamic_memlets
                        or not has_dynamic_memlets(first_map, second_map)
                    )
                )
                forced_K_merge = (
                    self.merge_strategy == MergeStrategy.Force_K
                    and both_k_maps(first_map, second_map)
                    and no_data_dependencies(
                        first_map,
                        second_map,
                        restrict_check_to_k=True,
                    )
                )
                if trivial_merge:
                    # merge
                    print(
                        f"Trivial merge: {first_map.node.map.params} in {first_map.node.map.range}"
                    )
                    first_map.children.extend(second_map.children)
                    del children[j]

                    # TODO also merge containers and symbols (if applicable)

                    # recurse into children
                    first_map.children = self._merge_maps(first_map.children)
                elif forced_K_merge:
                    print(
                        f"K merge: {first_map.node.map.params} in {first_map.node.map.range}"
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
                    first_map = PushDownIfStatement(
                        merged_range=merged_range, original_range=first_range
                    ).visit(first_map)
                    second_map = PushDownIfStatement(
                        merged_range=merged_range, original_range=second_range
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
                    first_map.children = self._merge_maps(first_map.children)
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
        node.children = self._merge_maps(node.children)


class KMapLoopFlip(dace_stree.ScheduleNodeVisitor):
    def _move_k_map_loop(
        self, k_map: dace_stree.MapScope | dace_stree.ForScope
    ) -> None:
        """Move k-{map, loop} out, one level at a time, as far as possible."""
        while isinstance(k_map.parent, dace_stree.MapScope):
            parent = k_map.parent
            grand_parent = parent.parent

            parent.children = k_map.children
            parent.parent = k_map

            k_map.parent = grand_parent
            k_map.children = [parent]

            if not isinstance(grand_parent, dace_stree.MapScope):
                # Assumption: We are at the top (tree root) now
                #
                # insert k_map before parent, then delete parent
                # from the list of children in grand_parent
                parent_index = grand_parent.children.index(parent)
                grand_parent.children.insert(parent_index, k_map)
                grand_parent.children.remove(parent)
                return

            assert grand_parent is not None
            grand_parent.children = [k_map]

    def visit_MapScope(self, map_scope: dace_stree.MapScope) -> None:
        map_parameter = map_scope.node.params
        # detect k-map
        if len(map_parameter) == 1 and map_parameter[0] == "__k":
            # attempt to move it out as far as possible
            return self._move_k_map_loop(map_scope)

        # visit children
        for child in map_scope.children:
            self.visit(child)

    def visit_ForScope(self, node: dace_stree.ForScope) -> None:
        # detect k-loop
        if node.header.itervar == "__k":
            # attempt to move it out as far as possible
            return self._move_k_map_loop(node)

        # visit children
        for child in node.children:
            self.visit(child)
