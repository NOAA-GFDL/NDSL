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


class CollectTransientRangeAccess(stree.ScheduleNodeVisitor):
    """Unionize all transient arrays access into a single Range."""

    def __init__(self) -> None:
        # Map access is a `list` instead of a `set` because we want to double count
        # access that are in/out as two access on the axis.
        self.transients_cartesian_maps: dict[
            str,
            tuple[
                set[dace.nodes.MapEntry],
                set[dace.nodes.MapEntry],
                set[dace.nodes.MapEntry],
            ],
        ] = {}
        self.transients_range_writes: dict[str, dace.subsets.Range | None] = {}
        self.transients_range_reads: dict[str, dace.subsets.Range | None] = {}

    def __str__(self) -> str:
        return "CartesianCollectMaps"

    def _find_first_map_or_loop(
        self,
        node: stree.TaskletNode,
        axis: AxisIterator,
    ) -> dace.nodes.MapEntry | None:
        parent = node.parent
        while parent is not None:
            if isinstance(parent, stree.MapScope):
                for p in parent.node.params:
                    if p.startswith(axis.as_str()):
                        return parent.node

            parent = parent.parent
        return None

    def _record_access(
        self,
        node: stree.TaskletNode,
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

                # Union the range
                recording_set[memlet.data] = dace.subsets.union(
                    recording_set[memlet.data], memlet.subset
                )

                # Record the map
                map_entry = self._find_first_map_or_loop(node, AxisIterator._I)
                if map_entry:
                    self.transients_cartesian_maps[memlet.data][
                        AxisIterator._I.as_cartesian_index()
                    ].add(map_entry)
                map_entry = self._find_first_map_or_loop(node, AxisIterator._J)
                if map_entry:
                    self.transients_cartesian_maps[memlet.data][
                        AxisIterator._J.as_cartesian_index()
                    ].add(map_entry)
                map_entry = self._find_first_map_or_loop(node, AxisIterator._K)
                if map_entry:
                    self.transients_cartesian_maps[memlet.data][
                        AxisIterator._K.as_cartesian_index()
                    ].add(map_entry)

    def visit_TaskletNode(self, node: stree.TaskletNode) -> None:
        self._record_access(node, node.input_memlets(), self.transients_range_writes)
        self._record_access(node, node.output_memlets(), self.transients_range_reads)

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        self.containers = node.containers
        for name, data in self.containers.items():
            if data.transient and isinstance(data, dace.data.Array):
                self.transients_cartesian_maps[name] = (set(), set(), set())
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


    It expects:
        - All Maps and ForLoop are on a single axis - but doesn't check for it.

    It can do:
        - Looking at usage of a transient in a cartesian axis (e.g. loop over a
        cartesian axis) it will reduce that axis to 1 if all access are atomic
        (exactly _one_ element of the array is ever worked on in a single loop)
        - It will refuse to merge if the transient is used in multiple loops of for
        a given axis - irrigardless of it's access pattern (e.g. even if it could be
        refine because it's always written first.)

    It should but cannot do/will bug if:
        - With better dataflow analysis, we can reduce the dimensions to the correct lowest
        size needed on the axis (e.g. transient[K] and transient[K+1], requires a 2-element
        buffer), instead of the defensive _no refine_ strategy used now. We have _most_ of the
        info in the `Range`
        - Current action when detecting a valid candidate is to reduce the size of the dimension
        to 1, rather than removing it. This will effectively, if generic compilers do their job, reduce
        the cache access significantly. This also has been implemented to _not_ deal with offset/slicing
        downstream impact of removing an axis. Nevertheless the axis should be removed if it's not
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
        elif backend in ["dace:gpu", "dace:cpu_KJI"]:
            self.ijk_order = (0, 1, 2)
        elif backend in ["dace:cpu"]:
            self.ijk_order = (1, 2, 0)
        else:
            raise NotImplementedError(
                f"[Schedule Tree Opt] CartesianRefineTransient not implemented for backend {backend}"
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
                # We do not refine multi-map transients
                if (
                    len(
                        collect_map.transients_cartesian_maps[name][
                            axis.as_cartesian_index()
                        ]
                    )
                    > 1
                ):
                    continue
                # Refine axis down to 1
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
