import dace.sdfg.analysis.schedule_tree.treenodes as tn

from ndsl.dsl.dace.stree.common import AxisIterator


def is_axis_map(node: tn.MapScope, axis: AxisIterator) -> bool:
    """Returns true if node is a Map over the given axis."""
    if len(node.node.map.params) != 1:
        return False

    param = node.node.map.params[0]
    assert isinstance(param, str)
    return axis.is_equal(param)


def is_axis_for(node: tn.ForScope, axis: AxisIterator) -> bool:
    """Returns true if node is a For over the given axis."""
    return axis.is_equal(node.loop.loop_variable)


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
    for symbol in node.condition.get_free_symbols():
        if (
            AxisIterator._I.as_str() in symbol
            or AxisIterator._J.as_str() in symbol
            or AxisIterator._K.as_str() in symbol
        ):
            return False
    return True
