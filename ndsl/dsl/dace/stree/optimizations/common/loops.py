import dace.sdfg.analysis.schedule_tree.treenodes as tn

from ndsl.dsl.dace.stree.optimizations.common import AxisIterator


def is_axis_map(node: tn.MapScope, axis: AxisIterator) -> bool:
    """Returns true if node is a Map over the given axis."""
    map_parameter = node.node.map.params
    if len(map_parameter) != 1:
        return False

    return axis.is_equal(map_parameter[0])


def is_axis_for(node: tn.ForScope, axis: AxisIterator) -> bool:
    """Returns true if node is a For over the given axis."""
    return axis.is_equal(node.loop.loop_variable)
