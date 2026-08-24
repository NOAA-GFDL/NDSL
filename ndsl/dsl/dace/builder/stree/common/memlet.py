from collections import defaultdict
from enum import Enum

from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn


class AxisIterator(Enum):
    _I = ("__i", 0)
    _J = ("__j", 1)
    _K = ("__k", 2)

    def as_str(self) -> str:
        return self.value[0]

    def as_cartesian_index(self) -> int:
        return self.value[1]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AxisIterator):
            return self.value == other.value
        if isinstance(other, str):
            return self.as_str() == other

        raise ValueError("Equality with AxisIterator or string is undefined")

    # Restore hashing that got sniped by __eq__ (see object.__hash__ in py doc)
    __hash__ = Enum.__hash__


CARTESIAN_AXIS_SYMBOLS = [AxisIterator._I, AxisIterator._J, AxisIterator._K]


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

    axis_index = axis.as_cartesian_index()

    for write in write_collector.out_memlets:
        # TODO: this can be optimized to allow non-overlapping intervals and such in the future

        if write.subset.dims() <= axis_index:
            # Dimension does not exist
            continue

        previous_axis_index = write.subset[axis_index][0]

        # Write-after-write with an offset case
        for other_write in other_writes.out_memlets:
            if (
                write.data == other_write.data
                and previous_axis_index != other_write.subset[axis_index][0]
            ):
                return False

        # Read-after-write with an offset case
        for read in read_collector.in_memlets:
            if (
                write.data == read.data
                and previous_axis_index != read.subset[axis_index][0]
            ):
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
        memlet.dynamic
        for memlet in [
            *first_collector.in_memlets,
            *first_collector.out_memlets,
            *second_collector.in_memlets,
            *second_collector.out_memlets,
        ]
    )
    return has_dynamic_memlets


class WriteDependencyCollector(tn.ScheduleNodeVisitor):
    """Collect write dependency for all data as list of memlet
    used as inputs in tasklet."""

    def __init__(self) -> None:
        super().__init__()
        self.dataflow: dict[str, list[Memlet]] = defaultdict(list)

    def visit_TaskletNode(self, node: tn.TaskletNode) -> None:
        # Go through each tasklet
        # For every output memlet.data, gather an _ordered_ list of the inputs.data
        # Build a dict[memlet.data, list[memlet]]

        for out_memlet in node.out_memlets.values():
            for in_memlet in node.in_memlets.values():
                self.dataflow[out_memlet.data].append(in_memlet)
