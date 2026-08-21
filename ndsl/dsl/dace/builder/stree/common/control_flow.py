from ndsl.dsl.dace.builder.stree.common.memlet import CARTESIAN_AXIS_SYMBOLS
import dace.sdfg.analysis.schedule_tree.treenodes as tn

from ndsl.dsl.dace.builder.stree.common import AxisIterator


def is_axis_map(node: tn.MapScope, axis: AxisIterator) -> bool:
    """Returns true if node is a Map over the given axis."""
    if len(node.node.map.params) != 1:
        return False

    param = node.node.map.params[0]
    assert isinstance(param, str)
    return axis == param


def is_axis_for(node: tn.ForScope, axis: AxisIterator) -> bool:
    """Returns true if node is a For over the given axis."""
    return axis == node.loop.loop_variable


def is_cartesian_axis(node: tn.MapScope | tn.ForScope) -> bool:
    """Returns true if the given node is a map over any cartesian axis."""
    for axis in AxisIterator:
        if (isinstance(node, tn.MapScope) and is_axis_map(node, axis)) or (
            isinstance(node, tn.ForScope) and is_axis_for(node, axis)
        ):
            return True

    return False


def is_off_grid_conditional(node: tn.IfScope) -> bool:
    """Conditional is off-grid if the code block does not refer to the cartesian symbols"""
    return not any(
        symbol in CARTESIAN_AXIS_SYMBOLS
        for symbol in node.condition.get_free_symbols()
    )

def is_off_grid_tasklet(node: tn.TaskletNode) -> bool:
    """Tasklet processing arrays not indexed by cartesian symbols"""

    for memlet in (
        *node.in_memlets.values(),
        *node.out_memlets.values(),
    ):
        if any(symbol in CARTESIAN_AXIS_SYMBOLS for symbol in memlet.free_symbols):
            return False

    return True