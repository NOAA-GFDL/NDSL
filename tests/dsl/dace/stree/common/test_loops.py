from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion

from ndsl.dsl.dace.stree.optimizations.common import (
    AxisIterator,
    is_axis_for,
    is_axis_map,
)


def test_is_axis_map_multiple_params() -> None:
    node = tn.MapScope(
        node=nodes.MapEntry(
            nodes.Map("map_ij", ["__i", "__j"], [(0, 3, 1), (0, 4, 1)])
        ),
        children=[],
    )
    assert not is_axis_map(node, AxisIterator._I)
    assert not is_axis_map(node, AxisIterator._J)


def test_is_axis_map_I() -> None:
    node = tn.MapScope(
        node=nodes.MapEntry(nodes.Map("map_i", ["__i"], [(0, 3, 1)])), children=[]
    )
    assert is_axis_map(node, AxisIterator._I)


def test_is_axis_map_not_I() -> None:
    node = tn.MapScope(
        node=nodes.MapEntry(nodes.Map("map_other_i", ["__i0"], [(0, 3, 1)])),
        children=[],
    )
    assert not is_axis_map(node, AxisIterator._I)


def test_is_axis_map_K() -> None:
    node = tn.MapScope(
        node=nodes.MapEntry(nodes.Map("map_k", ["__k_1234"], [(0, 3, 1)])), children=[]
    )
    assert is_axis_map(node, AxisIterator._K)


def test_is_axis_map_wrong_iterator() -> None:
    node = tn.MapScope(
        node=nodes.MapEntry(nodes.Map("map_i", ["__i"], [(0, 3, 1)])), children=[]
    )
    assert not is_axis_map(node, AxisIterator._J)


def test_is_axis_for_k() -> None:
    node = tn.ForScope(loop=LoopRegion("for_k", loop_var="__k_1234"), children=[])
    assert is_axis_for(node, AxisIterator._K)


def test_is_axis_for_wrong_iterator() -> None:
    node = tn.ForScope(loop=LoopRegion("for_k", loop_var="__k_1234"), children=[])
    assert not is_axis_for(node, AxisIterator._I)


def test_is_axis_for_i() -> None:
    node = tn.ForScope(loop=LoopRegion("for_i", loop_var="__i"), children=[])
    assert is_axis_for(node, AxisIterator._I)


def test_is_axis_for_not_i() -> None:
    node = tn.ForScope(loop=LoopRegion("for_i", loop_var="__i0"), children=[])
    assert not is_axis_for(node, AxisIterator._I)
