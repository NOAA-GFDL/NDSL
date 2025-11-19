from __future__ import annotations

import warnings
from types import TracebackType

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


def _reduce_cartesian_axes_size_to_1(
    transient_map_access: set[stree.nodes.MapEntry],
    transient_data: dace.data.Data,
    ijk_order: tuple[int, int, int],
) -> bool:
    """Reduce dimension size of transient to 1 if their are accessed only
    in a single Map for the cartesian dimensions"""
    refined = False
    for axis in AxisIterator:
        access_in_map_count = 0
        for map_entry in transient_map_access:
            if axis.as_str() in map_entry.params[0]:
                access_in_map_count += 1

        if access_in_map_count != 1:
            continue

        # This transient is used in exactly one single-Axis map
        # therefore this dimension can be removed. BUT we are not truly
        # removing it, we are reducing it to 1 to not have to deal
        # with different slicing.
        transient_data.shape = _change_index_of_tuple(
            transient_data.shape,
            axis.as_cartesian_index(),
            value=1,
        )

        # Assume 3D cartesian!
        if len(transient_data.shape) < 3:
            warnings.warn(
                f"Potential non-3D array: {transient_data}, skipping.",
                UserWarning,
                stacklevel=2,
            )
            return refined
        elif len(transient_data.shape) == 3:
            layout = [*ijk_order]
        else:
            data_dim_count = len(transient_data.shape) - 3
            layout = [dim + data_dim_count for dim in ijk_order] + [
                i - 1 for i in range(data_dim_count, 0, -1)
            ]

        transient_data.set_strides_from_layout(*layout)
        transient_data.lifetime = dace.dtypes.AllocationLifetime.State
        refined = True

    return refined


class _CartesianMapNesting:
    def __init__(
        self,
        cartesian_current_map_nesting: list[stree.nodes.MapEntry | None],
        node: stree.MapScope,
    ) -> None:
        self._cartesian_current_map_nesting = cartesian_current_map_nesting
        self._node = node

    def __enter__(self) -> None:
        if AxisIterator._I.value[0] in self._node.node.params[0]:
            self._cartesian_current_map_nesting[0] = self._node.node
        elif AxisIterator._J.value[0] in self._node.node.params[0]:
            self._cartesian_current_map_nesting[1] = self._node.node
        elif AxisIterator._K.value[0] in self._node.node.params[0]:
            self._cartesian_current_map_nesting[2] = self._node.node

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if AxisIterator._I.value[0] in self._node.node.params[0]:
            self._cartesian_current_map_nesting[0] = None
        elif AxisIterator._J.value[0] in self._node.node.params[0]:
            self._cartesian_current_map_nesting[1] = None
        elif AxisIterator._K.value[0] in self._node.node.params[0]:
            self._cartesian_current_map_nesting[2] = None


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
            return

        with _CartesianMapNesting(self._cartesian_current_map_nesting, node):
            for child in node.children:
                self.visit(child)

    def visit_TaskletNode(self, node: stree.TaskletNode) -> None:
        for memlet in [*node.input_memlets(), *node.output_memlets()]:
            data = self.containers[memlet.data]
            if data.transient and isinstance(data, dace.data.Array):
                for map_entry in self._cartesian_current_map_nesting:
                    if map_entry is not None:
                        self.transient_map_access[memlet.data].add(map_entry)

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        self.containers = node.containers
        for name, data in self.containers.items():
            if data.transient and isinstance(data, dace.data.Array):
                self.transient_map_access[name] = set()

        for child in node.children:
            self.visit(child)


class RebuildMemletsFromContainers(stree.ScheduleNodeVisitor):
    """Rebuild memlets from containers to ensure they are scope to the right size."""

    def __str__(self) -> str:
        return "RefineTransientAxis"

    def visit_TaskletNode(self, node: stree.TaskletNode) -> None:
        for memlet in [*node.out_memlets.values(), *node.out_memlets.values()]:
            array = self.containers[memlet.data]
            if array.transient and array:
                replace_cartesian_access = {}
                if len(array.shape) >= 1 and array.shape[0] == 1:
                    replace_cartesian_access[AxisIterator._I.as_str()] = 0
                if len(array.shape) >= 2 and array.shape[1] == 1:
                    replace_cartesian_access[AxisIterator._J.as_str()] = 0
                if len(array.shape) >= 3 and array.shape[2] == 1:
                    # Workaround because the iterator can be `__k_0` instead of `__k`
                    axis = None
                    for axis_symbol in memlet.free_symbols:
                        if axis_symbol.startswith(AxisIterator._K.as_str()):
                            axis = axis_symbol
                            break
                    if axis:
                        replace_cartesian_access[axis] = 0
                memlet.replace(replace_cartesian_access)

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        self.containers = node.containers
        for child in node.children:
            self.visit(child)


class CartesianRefineTransients(stree.ScheduleNodeTransformer):
    """Refine (reduce dimensionality) of transients based on their true use in
    the cartesian dimensions.


    It can do:
        - Looking at usage of a transient in a cartesian axis (e.g. loop over a
        cartesian axis) it will reduce that axis to 1 if it exists in _only one_.

    It should but cannot do/will bug if:
        - Dataflow analysis on the axis to prevent reducing an axis to one where
        the transient is used with offset, leading to faulty numerics
        - Using the dataflow above, we can reduce the dimensions to the correct lowest
        size needed on the axis (e.g. transient[K] and transient[K+1], requires a 2-element
        buffer)
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

    def __str__(self) -> str:
        return "CartesianRefineTransients"

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        collect_map = CollectTransientAccessInCartesianMaps()
        collect_map.visit(node)

        # Remove Axis
        refined_transient = 0
        for name, data in node.containers.items():
            if not (data.transient and isinstance(data, dace.data.Array)):
                continue
            refined = _reduce_cartesian_axes_size_to_1(
                collect_map.transient_map_access[name],
                data,
                self.ijk_order,
            )
            refined_transient += 1 if refined else 0

        RebuildMemletsFromContainers().visit(node)

        ndsl_log.debug(f"🚀 {refined_transient} Transient refined")
