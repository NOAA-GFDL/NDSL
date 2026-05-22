import dataclasses

import dace
import dace.sdfg.analysis.schedule_tree.treenodes as stree

from ndsl.dsl.dace.stree.optimizations.common import (
    AxisIterator,
    is_axis_for,
    is_axis_map,
)


class CountCartesianLoops(stree.ScheduleNodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._maps = [0, 0, 0]
        self._fors = [0, 0, 0]

    def visit_MapScope(self, node: stree.MapScope) -> None:
        for axis in AxisIterator:
            if is_axis_map(node, axis):
                self._maps[axis.as_cartesian_index()] += 1

        self.visit(node.children)

    def visit_ForScope(self, node: stree.ForScope) -> None:
        for axis in AxisIterator:
            if is_axis_for(node, axis):
                self._fors[axis.as_cartesian_index()] += 1

        self.visit(node.children)


class CountTransient(stree.ScheduleNodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._counts = [0, 0, 0, 0, 0]

    def visit_ScheduleTreeRoot(self, node: stree.ScheduleTreeRoot) -> None:
        for data in node.containers.values():
            non_atomic_dims_count = sum(1 for x in data.shape if x != 1)
            if isinstance(data, dace.data.Array) and data.transient:
                if non_atomic_dims_count == 0:
                    self._counts[0] += 1
                elif non_atomic_dims_count == 1:
                    self._counts[1] += 1
                elif non_atomic_dims_count == 2:
                    self._counts[2] += 1
                elif non_atomic_dims_count == 3:
                    self._counts[3] += 1
                else:
                    self._counts[4] += 1


class TreeOptimizationStatistics:
    """Capture basic statistics on the schedule tree optimization actions"""

    @dataclasses.dataclass
    class Record:
        """Private record of a state of a tree"""

        cartesian_maps: list[int] = dataclasses.field(default_factory=lambda: [0, 0, 0])
        cartesian_fors: list[int] = dataclasses.field(default_factory=lambda: [0, 0, 0])
        transients: list[int] = dataclasses.field(
            default_factory=lambda: [0, 0, 0, 0, 0]
        )

    def __init__(self) -> None:
        self._original_record = TreeOptimizationStatistics.Record()
        self._optimized_record = TreeOptimizationStatistics.Record()

    def _record(
        self,
        record: Record,
        tree_root: stree.ScheduleTreeRoot,
    ) -> None:
        """Record the state of a tree"""
        c = CountCartesianLoops()
        c.visit(tree_root)
        record.cartesian_fors = c._fors
        record.cartesian_maps = c._maps

        c = CountTransient()
        c.visit(tree_root)
        record.transients = c._counts

    def original(self, tree_root: stree.ScheduleTreeRoot) -> None:
        """Record the original state of the tree, before optimization"""
        self._record(self._original_record, tree_root)

    def optimized(self, tree_root: stree.ScheduleTreeRoot) -> None:
        """Record the state of the tree after optimization"""
        self._record(self._optimized_record, tree_root)

    def report(self) -> str:
        """Craft a concize string reporting on the statistics"""
        msg = "Tree optimization:\n"
        msg += f"  Cartesian maps [I, J, K]: {self._original_record.cartesian_maps} -> {self._optimized_record.cartesian_maps}\n"
        msg += f"  Cartesian fors [I, J, K]: {self._original_record.cartesian_fors} -> {self._optimized_record.cartesian_fors}\n"
        msg += f"  Transients [Scalarized Array, 1D, 2D, 3D, 4D+]: {self._original_record.transients} -> {self._optimized_record.transients}\n"
        return msg
