import pytest
from dace import nodes, subsets
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.symbolic import symbol

from ndsl.dsl.dace.stree.optimizations.common import AxisIterator
from ndsl.dsl.dace.stree.optimizations.common.memlet import (
    normalize_cartesian_indexation,
)


@pytest.fixture
def k_map() -> tn.MapScope:
    return tn.MapScope(
        node=nodes.MapEntry(
            nodes.Map("map", ["__k_123456789"], subsets.Range.from_string("0:5"))
        ),
        children=[],
    )


def test_normalize_cartesian_index(k_map: tn.MapScope) -> None:
    # Case of __k_id(node) - original case
    original_symbol = symbol("__k_12345678789")
    norm_symbol = normalize_cartesian_indexation(
        original_symbol, AxisIterator._K, k_map
    )

    assert norm_symbol == symbol("__k")

    # Case of offset
    original_symbol = 1 + symbol("__k_12345678789")
    norm_symbol = normalize_cartesian_indexation(
        original_symbol, AxisIterator._K, k_map
    )

    assert norm_symbol == symbol("__k") + 1

    # Case of no-op (with offset)
    original_symbol = 1 + symbol("__k")
    norm_symbol = normalize_cartesian_indexation(
        original_symbol, AxisIterator._K, k_map
    )

    assert norm_symbol == symbol("__k") + 1

    # Case of index named with _k - so not a cartesian axis
    original_symbol = 1 + symbol("_kindex")
    norm_symbol = normalize_cartesian_indexation(
        original_symbol, AxisIterator._K, k_map
    )

    assert norm_symbol == symbol("_kindex") + 1


def test_normalize_cartesian_index_map_two_params() -> None:
    ij_map = tn.MapScope(
        node=nodes.MapEntry(
            nodes.Map("map", ["__i", "__j"], subsets.Range([(0, 3, 1), (0, 4, 2)]))
        ),
        children=[],
    )
    with pytest.raises(ValueError, match="Expected a map with only one parameter"):
        normalize_cartesian_indexation(symbol("__i"), AxisIterator._I, ij_map)


def test_normalize_cartesian_index_map_wrong_index(k_map) -> None:
    with pytest.raises(ValueError, match="Mismatch of axis iterator"):
        normalize_cartesian_indexation(symbol("__i"), AxisIterator._I, k_map)


def test_normalize_cartesian_index_map_start(k_map) -> None:
    map_m1 = tn.MapScope(
        node=nodes.MapEntry(
            nodes.Map("map", ["__i"], subsets.Range.from_string("-1:3"))
        ),
        children=[],
    )

    original_symbol = symbol("__i")
    normalized = normalize_cartesian_indexation(
        original_symbol, AxisIterator._I, map_m1
    )
    assert normalized == original_symbol - 1

    original_symbol = 1 + symbol("__i")
    normalized = normalize_cartesian_indexation(
        original_symbol, AxisIterator._I, map_m1
    )
    assert normalized == symbol("__i")

    original_symbol = symbol("__i") + 5
    normalized = normalize_cartesian_indexation(
        original_symbol, AxisIterator._I, map_m1
    )
    assert normalized == symbol("__i") + 4

    original_symbol = 1 + symbol("__i_1234")
    normalized = normalize_cartesian_indexation(
        original_symbol, AxisIterator._I, map_m1
    )
    assert normalized == symbol("__i")
