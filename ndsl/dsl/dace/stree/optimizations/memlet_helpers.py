import dace.sdfg.analysis.schedule_tree.treenodes as dst
from dace.memlet import Memlet

from enum import Enum


class AxisIterator(Enum):
    _I = ("__i", 0)
    _J = ("__j", 1)
    _K = ("__k", 2)

    def as_str(self) -> str:
        return self.value[0]

    def as_cartesian_index(self) -> int:
        return self.value[1]


def no_data_dependencies_on_cartesian_axis(
    first: dst.MapScope,
    second: dst.MapScope,
    axis: AxisIterator,
) -> bool:
    """Check for read after write. Allow when indexation on the axis
    is not offseted."""

    write_collector = MemletCollector(collect_reads=False)
    write_collector.visit(first)
    read_collector = MemletCollector(collect_writes=False)
    read_collector.visit(second)
    for write in write_collector.out_memlets:
        # TODO: this can be optimized to allow non-overlapping intervals and such in the future

        if write.subset.dims() <= axis.as_cartesian_index():
            # Dimension does not exist
            continue

        previous_axis_index = write.subset[axis.as_cartesian_index()][0]
        for read in read_collector.in_memlets:
            if write.data == read.data:
                if previous_axis_index != read.subset[axis.as_cartesian_index()][0]:
                    print(
                        f"[{axis.name} Merge] Found read after write conflict "
                        f"for {write.data} "
                        f"w/ different offset to {axis.name} ("
                        f"write at {read.subset[axis.as_cartesian_index()][0]}, "
                        f"read at {read.subset[axis.as_cartesian_index()][0]})"
                    )
                    return False
    return True


def no_data_dependencies(
    first: dst.MapScope,
    second: dst.MapScope,
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


class MemletCollector(dst.ScheduleNodeVisitor):
    """Gathers in_memlets and out_memlets of TaskNodes and LibraryCalls."""

    in_memlets: list[Memlet]
    out_memlets: list[Memlet]

    def __init__(self, *, collect_reads=True, collect_writes=True):
        self._collect_reads = collect_reads
        self._collect_writes = collect_writes

        self.in_memlets = []
        self.out_memlets = []

    def visit_TaskletNode(self, node: dst.TaskletNode) -> None:
        if self._collect_reads:
            self.in_memlets.extend([memlet for memlet in node.in_memlets.values()])
        if self._collect_writes:
            self.out_memlets.extend([memlet for memlet in node.out_memlets.values()])

    def visit_LibraryCall(self, node: dst.LibraryCall) -> None:
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


def has_dynamic_memlets(first: dst.MapScope, second: dst.MapScope) -> bool:
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
