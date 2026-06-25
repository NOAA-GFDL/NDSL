from enum import Enum

from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.symbolic import symbol

from ndsl import ndsl_log


class AxisIterator(Enum):
    _I = ("__i", 0)
    _J = ("__j", 1)
    _K = ("__k", 2)

    def as_str(self) -> str:
        return self.value[0]

    def as_cartesian_index(self) -> int:
        return self.value[1]

    def is_equal(self, other: str) -> bool:
        if self == AxisIterator._K:
            return other.startswith(self.as_str())

        return other == self.as_str()


def normalize_cartesian_indexation(
    index: symbol, axis: AxisIterator, map_scope: tn.MapScope
) -> symbol:
    """Return a normalized indexation symbol for cartesian indexation."""
    if len(map_scope.node.map.params) != 1:
        raise ValueError(
            f"Expected a map with only one parameter, got {map_scope.node.map.params}."
        )

    axis_name = axis.as_str()
    if not map_scope.node.map.params[0].startswith(axis_name):
        raise ValueError(
            f"Mismatch of axis iterator {axis} and MapScope parameter {map_scope.node.map.params}."
        )

    # potentially rename
    rename_maps = {}
    for sym in index.free_symbols:
        if sym.name != axis_name and sym.name.startswith(axis_name):
            rename_maps[sym] = symbol(axis_name)
    renamed = index.subs(rename_maps)

    # handle potential map start
    map_start = map_scope.node.map.range.min_element()[0]
    if map_start != 0:
        return renamed + map_start

    return renamed


def no_data_dependencies_on_cartesian_axis(
    first: tn.MapScope,
    second: tn.MapScope,
    axis: AxisIterator,
) -> bool:
    """Check for read after write and write after write with different offsets."""

    write_collector = MemletCollector(collect_reads=False)
    write_collector.visit(first)
    other_writes = MemletCollector(collect_reads=False)
    other_writes.visit(second)
    read_collector = MemletCollector(collect_writes=False)
    read_collector.visit(second)

    for write in write_collector.out_memlets:
        # TODO: this can be optimized to allow non-overlapping intervals and such in the future

        axis_index = axis.as_cartesian_index()

        if write.subset.dims() <= axis_index:
            # Dimension does not exist
            continue

        previous_axis_index = normalize_cartesian_indexation(
            write.subset[axis_index][0], axis, first
        )

        # Write-after-write with an offset case
        for other_write in other_writes.out_memlets:
            if write.data == other_write.data:
                current_axis_index = normalize_cartesian_indexation(
                    other_write.subset[axis_index][0], axis, second
                )
                if previous_axis_index != current_axis_index:
                    ndsl_log.debug(
                        f"[{axis.name} Merge] Found write after write conflict "
                        f"for {write.data} "
                        f"w/ different offset to {axis.name} ("
                        f"first write at {previous_axis_index}, "
                        f"second write at {current_axis_index})"
                    )
                    return False

        # Read-after-write with an offset case
        for read in read_collector.in_memlets:
            if write.data == read.data:
                current_axis_index = normalize_cartesian_indexation(
                    read.subset[axis_index][0], axis, second
                )
                if previous_axis_index != current_axis_index:
                    ndsl_log.debug(
                        f"[{axis.name} Merge] Found read after write conflict "
                        f"for {write.data} "
                        f"w/ different offset to {axis.name} ("
                        f"write at {previous_axis_index}, "
                        f"read at {current_axis_index})"
                    )
                    return False

    return True


class MemletCollector(tn.ScheduleNodeVisitor):
    """Gathers in_memlets and out_memlets of TaskNodes and LibraryCalls."""

    in_memlets: list[Memlet]
    out_memlets: list[Memlet]

    def __init__(
        self, *, collect_reads: bool = True, collect_writes: bool = True
    ) -> None:
        self._collect_reads = collect_reads
        self._collect_writes = collect_writes

        self.in_memlets = []
        self.out_memlets = []

    def visit_TaskletNode(self, node: tn.TaskletNode) -> None:
        if self._collect_reads:
            self.in_memlets.extend([memlet for memlet in node.in_memlets.values()])
        if self._collect_writes:
            self.out_memlets.extend([memlet for memlet in node.out_memlets.values()])

    def visit_LibraryCall(self, node: tn.LibraryCall) -> None:
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


def has_dynamic_memlets(first: tn.MapScope, second: tn.MapScope) -> bool:
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
