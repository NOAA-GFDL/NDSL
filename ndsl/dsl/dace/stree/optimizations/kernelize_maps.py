from copy import deepcopy

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import Backend
from ndsl.config import BackendLoopOrder, BackendTargetDevice
from ndsl.dsl.dace.stree.optimizations.common import (
    AxisIterator,
    is_axis_map,
    is_cartesian_axis,
)


class _KernelizeMap(tn.ScheduleNodeTransformer):
    def __init__(self, axis: AxisIterator) -> None:
        super().__init__()
        self._axis = axis

    def __str__(self) -> str:
        return f"KernelizeMap_{self._axis}"

    def _count_cartesian_children(self, node: tn.ScheduleTreeScope) -> int:
        cartesian_children = 0
        for child in node.children:
            if isinstance(child, (tn.MapScope, tn.ForScope)) and is_cartesian_axis(
                child
            ):
                cartesian_children += 1
        return cartesian_children

    def visit_MapScope(self, node: tn.MapScope) -> tn.MapScope | list[tn.MapScope]:
        # if this is a map on a cartesian axis
        # and the children contain two or more cartesian axes
        if is_axis_map(node, self._axis) and self._count_cartesian_children(node) > 1:
            kernelized_maps: list[tn.MapScope] = []
            current_children: list[tn.ScheduleTreeNode] = []

            for child in node.children:
                current_children.append(child)
                if isinstance(child, (tn.MapScope, tn.ForScope)) and is_cartesian_axis(
                    child
                ):
                    kernelized_maps.append(
                        tn.MapScope(
                            node=deepcopy(node.node),
                            children=[child for child in current_children],
                            parent=node.parent,
                            state=node.state,
                        )
                    )
                    current_children = []
            return kernelized_maps

        return self.generic_visit(node)


class KernelizeMaps(tn.ScheduleNodeVisitor):
    def __init__(self, backend: Backend) -> None:
        super().__init__()
        self._backend = backend

        if self._backend.device != BackendTargetDevice.GPU:
            raise ValueError(
                "The transformation `KernelizeMaps` is only intended to run on GPUs."
            )

    def __str__(self) -> str:
        return "KernelizeMaps"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        for axis in self._axis_order():
            _KernelizeMap(axis).visit(node)

    def _axis_order(self) -> list[AxisIterator]:
        if self._backend.loop_order == BackendLoopOrder.IJK:
            return [AxisIterator._J, AxisIterator._I]
        if self._backend.loop_order == BackendLoopOrder.KJI:
            return [AxisIterator._J, AxisIterator._K]

        raise NotImplementedError(
            f"KernelizeMaps is not configured for loop order {self._backend.loop_order}."
        )
