from __future__ import annotations

import dace.data
import dace.sdfg.analysis.schedule_tree.treenodes as stree

from ndsl import ndsl_log
from ndsl.dsl.dace.stree.optimizations.memlet_helpers import AxisIterator


def _zero_index_of_tuple(tuple_: tuple[int, ...], index: int) -> tuple[int, ...]:
    new_list = list(tuple_)
    new_list[index] = 1
    return tuple(new_list)


def _reduce_axis_size_to_1(
    axis_iterator: AxisIterator,
    transient_map_access: set[stree.nodes.MapEntry],
    data: dace.data.Data,
    ijk_order: tuple[int, int, int],
) -> bool:
    access_in_map_count = 0
    for map_entry in transient_map_access:
        if axis_iterator.value[0] in map_entry.params[0]:
            access_in_map_count += 1

    # If this transient is used in exactly one single-Axis map
    # therefore this dimension can be removed
    if access_in_map_count != 1:
        return False

    data.shape = _zero_index_of_tuple(data.shape, axis_iterator.value[1])
    data.set_strides_from_layout(*ijk_order)
    return True


class CollectTransientAccessInCartesianMaps(stree.ScheduleNodeVisitor):
    """Collect all access of transient arrays per Maps."""

    def __init__(self) -> None:
        self.transient_map_access: dict[str, set[stree.nodes.MapEntry]] = {}
        self._cartesian_current_map_nesting: list[stree.nodes.MapEntry | None] = [
            None,
            None,
            None,
        ]

    def __str__(self) -> str:
        return "CartesianCollectMaps"

    def visit_MapScope(self, node: stree.MapScope) -> None:
        if len(node.node.params) > 1:
            ndsl_log.debug(
                "Can't apply CartesianRefineTransients, require unidimensional Maps"
            )

        if AxisIterator._I.value[0] in node.node.params[0]:
            self._cartesian_current_map_nesting[0] = node.node
        elif AxisIterator._J.value[0] in node.node.params[0]:
            self._cartesian_current_map_nesting[1] = node.node
        elif AxisIterator._K.value[0] in node.node.params[0]:
            self._cartesian_current_map_nesting[2] = node.node

        for child in node.children:
            return self.visit(child)

        if AxisIterator._I.value[0] in node.node.params[0]:
            self._cartesian_current_map_nesting[0] = None
        elif AxisIterator._J.value[0] in node.node.params[0]:
            self._cartesian_current_map_nesting[1] = None
        elif AxisIterator._K.value[0] in node.node.params[0]:
            self._cartesian_current_map_nesting[2] = None

    def visit_TaskletNode(self, node: stree.TaskletNode) -> None:
        for memlet in node.input_memlets():
            if self.containers[memlet.data].transient:
                for map_entry in self._cartesian_current_map_nesting:
                    if map_entry is not None:
                        self.transient_map_access[memlet.data].add(map_entry)
        for memlet in node.output_memlets():
            if self.containers[memlet.data].transient:
                for map_entry in self._cartesian_current_map_nesting:
                    if map_entry is not None:
                        self.transient_map_access[memlet.data].add(map_entry)

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        self.containers = node.containers
        for name, data in self.containers.items():
            if data.transient:
                self.transient_map_access[name] = set()

        for child in node.children:
            self.visit(child)


class RebuildMemletsFromContainers(stree.ScheduleNodeVisitor):
    """Rebuild memlets from containers to ensure they are scope to the right size."""

    def __str__(self) -> str:
        return "RefineTransientAxis"

    def visit_TaskletNode(self, node: stree.TaskletNode) -> None:
        for name, memlet in node.in_memlets.items():
            if self.containers[memlet.data].transient:
                node.in_memlets[name] = memlet.from_array(
                    memlet.data, self.containers[memlet.data]
                )

        for name, memlet in node.out_memlets.items():
            if self.containers[memlet.data].transient:
                node.out_memlets[name] = memlet.from_array(
                    memlet.data, self.containers[memlet.data]
                )

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        self.containers = node.containers
        for child in node.children:
            self.visit(child)


class CartesianRefineTransients(stree.ScheduleNodeTransformer):
    """ """

    def __init__(self, ijk_order: tuple[int, int, int]) -> None:
        self.ijk_order = ijk_order

    def __str__(self) -> str:
        return "CartesianRefineTransients"

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        collect_map = CollectTransientAccessInCartesianMaps()
        collect_map.visit(node)

        # Remove Axis
        refined_transient = 0
        for name, data in node.containers.items():
            if data.transient:
                refined = False
                for axis in AxisIterator:
                    refined |= _reduce_axis_size_to_1(
                        axis,
                        collect_map.transient_map_access[name],
                        data,
                        self.ijk_order,
                    )
                refined_transient += 1 if refined else 0

        RebuildMemletsFromContainers().visit(node)

        ndsl_log.debug(f"🚀 {refined_transient} Transient refined")
