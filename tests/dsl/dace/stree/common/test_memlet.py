from dace.symbolic import symbol

from ndsl.dsl.dace.stree.optimizations.common import AxisIterator
from ndsl.dsl.dace.stree.optimizations.common.memlet import (
    normalize_cartesian_indexation,
)


def test_normalize_cartesian_index():
    # Case of __k_id(node) - original case
    original_symbol = symbol("__k_12345678789")
    norm_symbol = normalize_cartesian_indexation(original_symbol, AxisIterator._K)

    assert norm_symbol == symbol("__k")

    # Case of offset
    original_symbol = 1 + symbol("__k_12345678789")
    norm_symbol = normalize_cartesian_indexation(original_symbol, AxisIterator._K)

    assert norm_symbol == symbol("__k") + 1

    # Case of no-op (with offset)
    original_symbol = 1 + symbol("__k")
    norm_symbol = normalize_cartesian_indexation(original_symbol, AxisIterator._K)

    assert norm_symbol == symbol("__k") + 1

    # Case of index named with _k - so not a cartesian axis
    original_symbol = 1 + symbol("_kindex")
    norm_symbol = normalize_cartesian_indexation(original_symbol, AxisIterator._K)

    assert norm_symbol == symbol("_kindex") + 1
