import dace.sdfg.analysis.schedule_tree.treenodes as tn

from ndsl.dsl.dace.stree.optimizations.common import AxisIterator


def is_axis_map(node: tn.MapScope, axis: AxisIterator) -> bool:
    """Returns true if node is a Map over the given axis."""
    map_parameter = node.node.map.params
    if len(map_parameter) != 1:
        return False

    if axis == AxisIterator._K:
        return map_parameter[0].startswith(axis.as_str())

    return map_parameter[0] == axis.as_str()


def is_axis_for(node: tn.ForScope, axis: AxisIterator) -> bool:
    """Returns true if node is a For over the given axis."""
    if axis == AxisIterator._K:
        return node.loop.loop_variable.startswith(axis.as_str())

    return node.loop.loop_variable == axis.as_str()
