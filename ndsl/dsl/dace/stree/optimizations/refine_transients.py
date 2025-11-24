from __future__ import annotations

import warnings

import dace.data
import dace.sdfg.analysis.schedule_tree.treenodes as stree

from ndsl import ndsl_log
from ndsl.dsl.dace.stree.optimizations.memlet_helpers import AxisIterator


def _change_index_of_tuple(
    old_tuple: tuple[int, ...], index: int, value: int = 1
) -> tuple[int, ...]:
    """Return a copy of the given tuple with `old_tuple[index]` being replaced by `value`.

    Args:
        old_tuple: to be copied
        index: at which index to replace a value
        value: to replace `old_tuple[index]`
    """
    new_list = list(old_tuple)
    new_list[index] = value
    return tuple(new_list)


def _reduce_cartesian_axis_size_to_1(
    axis: AxisIterator,
    transient_map_reads: dace.subsets.Range | None,
    transient_map_writes: dace.subsets.Range | None,
    transient_data: dace.data.Data,
    ijk_order: tuple[int, int, int],
) -> bool:
    """Reduce dimension size of transient to 1 if all access (reads and writes)
    are atomic"""

    # Dev Note: Better dataflow analysis would look at exactly
    #           what's goin on here!

    # Assume 3D cartesian!
    if len(transient_data.shape) < 3:
        warnings.warn(
            f"Potential non-3D array: {transient_data}, skipping.",
            UserWarning,
            stacklevel=2,
        )
        return False

    read_write_range: dace.subsets.Range = dace.subsets.union(
        transient_map_reads, transient_map_writes
    )

    if read_write_range is None:
        return False

    if read_write_range.size()[axis.as_cartesian_index()] != 1:
        return False

    # This transient read and write access is done on exactly one element
    # therefore this dimension can be removed. BUT we are not truly
    # removing it, we are reducing it to 1 to not have to deal
    # with different slicing.
    transient_data.shape = _change_index_of_tuple(
        transient_data.shape,
        axis.as_cartesian_index(),
        value=1,
    )

    if len(transient_data.shape) == 3:
        layout = [*ijk_order]
    else:
        data_dim_count = len(transient_data.shape) - 3
        layout = [dim + data_dim_count for dim in ijk_order] + [
            i - 1 for i in range(data_dim_count, 0, -1)
        ]

    transient_data.set_strides_from_layout(*layout)
    transient_data.lifetime = dace.dtypes.AllocationLifetime.State
    return True


def _axis_from_loop_iterator(node: stree.MapScope | stree.ForScope) -> AxisIterator:
    if isinstance(node, stree.MapScope):
        if AxisIterator._I.value[0] in node.node.params[0]:
            return AxisIterator._I
        if AxisIterator._J.value[0] in node.node.params[0]:
            return AxisIterator._J
        elif AxisIterator._K.value[0] in node.node.params[0]:
            return AxisIterator._K
    elif isinstance(node, stree.ForScope):
        if AxisIterator._I.value[0] in node.header.itervar:
            return AxisIterator._I
        elif AxisIterator._J.value[0] in node.header.itervar:
            return AxisIterator._J
        elif AxisIterator._K.value[0] in node.header.itervar:
            return AxisIterator._K
    raise NotImplementedError(f"Uncaught cartesian loop type {node}")


class CollectTransientRangeAccess(stree.ScheduleNodeVisitor):
    """Unionize all transient arrays access into a single Range."""

    def __init__(self) -> None:
        # Map access is a `list` instead of a `set` because we want to double count
        # access that are in/out as two access on the axis.
        self.transients_range_writes: dict[str, dace.subsets.Range | None] = {}
        self.transients_range_reads: dict[str, dace.subsets.Range | None] = {}

    def __str__(self) -> str:
        return "CartesianCollectMaps"

    def _record_access(
        self,
        memlets: stree.MemletSet,
        recording_set: dict[str, dace.subsets.Range | None],
    ) -> None:
        for memlet in memlets:
            data = self.containers[memlet.data]
            if data.transient and isinstance(data, dace.data.Array):
                if not isinstance(memlet.subset, dace.subsets.Range):
                    raise NotImplementedError(
                        "Memlet refining only works with Range subsets"
                    )
                recording_set[memlet.data] = dace.subsets.union(
                    recording_set[memlet.data], memlet.subset
                )

    def visit_TaskletNode(self, node: stree.TaskletNode) -> None:
        self._record_access(node.input_memlets(), self.transients_range_writes)
        self._record_access(node.output_memlets(), self.transients_range_reads)

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        self.containers = node.containers
        for name, data in self.containers.items():
            if data.transient and isinstance(data, dace.data.Array):
                self.transients_range_writes[name] = None
                self.transients_range_reads[name] = None

        for child in node.children:
            self.visit(child)


class RebuildMemletsFromContainers(stree.ScheduleNodeVisitor):
    """Rebuild memlets from containers to ensure they are scope to the right size."""

    def __init__(self, refined_arrays: set[str]) -> None:
        self._refined_arrays = refined_arrays

    def __str__(self) -> str:
        return "RefineTransientAxis"

    def visit_TaskletNode(self, node: stree.TaskletNode) -> None:
        for memlet in [*node.output_memlets(), *node.input_memlets()]:
            if memlet.data not in self._refined_arrays:
                continue
            array = self.containers[memlet.data]
            if array.transient:
                if not isinstance(memlet.subset, dace.subsets.Range):
                    raise NotImplementedError(
                        "Memlet refining only works with Range subsets"
                    )

                # Reduce "refined" dimension to a single element, effectively
                # eliminating it.
                for index, _ in enumerate(memlet.subset.ranges):
                    if array.shape[index] == 1:
                        memlet.subset.ranges[index] = (0, 0, 1)

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        self.containers = node.containers
        for child in node.children:
            self.visit(child)


class CartesianRefineTransients(stree.ScheduleNodeTransformer):
    """Refine (reduce dimensionality) of transients based on their true use in
    the cartesian dimensions.


    It can do:
        - Looking at usage of a transient in a cartesian axis (e.g. loop over a
        cartesian axis) it will reduce that axis to 1 if all access are atomic
        (exactly _one_ element of the array is ever worked on)

    It should but cannot do/will bug if:
        - If the transient is _written_ before being _read_ this won't catch it (not its job), but we could
        - With better dataflow analysis, we can reduce the dimensions to the correct lowest
        size needed on the axis (e.g. transient[K] and transient[K+1], requires a 2-element
        buffer), instead of the defensive _no refine_ strategy used now. We have _most_ of the
        info in the `Range`
        - Current action when detecting a valid candidate is to reduce the size of the dimension
        to 1, rather than removing it. This will effectively, if generic compilers do their job, reduce
        the cache access significantly. This also has been implemented to _not_ deal with offset/slicing
        downstream impact of removing an axis. Nevertheless the xis should be removed if it's not
        used.
        - It only knows how to deal with 3D cartesian and 3D cartesian + data dimensions. Anything else will
        fail `_reduce_cartesian_axes_size_to_1` calculation

    More tests:
        - Test for dataflow with offset
        - Test for I/J refine but not in K
        - Test for J refine but not in I or K
        - Test with dataflow: if/else, while, etc.
        - Test with ForScope (FORWARD/BACKWARD) instead of Map

    Coding traps:
        - We reduce the "refined" dimensions to 1 which is functionally eliminating it. This is solid. In the
        case of the one we can't eliminate we don't do anything. We could find the "smallest buffer size" needed
        and reduce the local dimension to it. BUT if we do this, we have to take into account the offset into
        memory (e.g. halo) for the `RebuildMemletsFromContainers`!
    """

    def __init__(self, backend: str) -> None:
        warnings.warn(
            "CartesianRefineTransients is a WIP. It's usage is *severely* limited "
            "and will most likely lead to bad numerics. Check the docs, check utest.",
            UserWarning,
            stacklevel=2,
        )

        if backend in ["dace:cpu_kfirst"]:
            self.ijk_order = (2, 1, 0)
        elif backend in ["dace:cpu", "dace:gpu"]:
            self.ijk_order = (1, 0, 2)
        else:
            raise NotImplementedError(
                "[Schedule Tree Opt] CartesianRefineTransient not implemented for "
                f"backend {backend}"
            )

        self.refined_array: set[str] = set()

    def __str__(self) -> str:
        return "CartesianRefineTransients"

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        collect_map = CollectTransientRangeAccess()
        collect_map.visit(node)

        # Remove Axis
        refined_transient = 0
        for name, data in node.containers.items():
            if not (data.transient and isinstance(data, dace.data.Array)):
                continue
            refined = False
            for axis in AxisIterator:
                refined |= _reduce_cartesian_axis_size_to_1(
                    axis,
                    collect_map.transients_range_reads[name],
                    collect_map.transients_range_writes[name],
                    data,
                    self.ijk_order,
                )

            refined_transient += 1 if refined else 0
            self.refined_array.add(name)

        RebuildMemletsFromContainers(self.refined_array).visit(node)

        ndsl_log.debug(f"🚀 {refined_transient} Transient refined")
