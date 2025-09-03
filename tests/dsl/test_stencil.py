import pytest
from gt4py.storage import empty, ones

from ndsl import CompilationConfig, GridIndexing, StencilConfig, StencilFactory
from ndsl.dsl.gt4py import PARALLEL, Field, computation, interval
from tests.dsl import utils


def test_timing_collector() -> None:
    grid_indexing = GridIndexing(
        domain=(5, 5, 5),
        n_halo=2,
        south_edge=True,
        north_edge=True,
        west_edge=True,
        east_edge=True,
    )
    stencil_config = StencilConfig(
        compilation_config=CompilationConfig(backend="numpy", rebuild=True)
    )

    stencil_factory = StencilFactory(stencil_config, grid_indexing)

    def func(inp: Field[float], out: Field[float]):
        with computation(PARALLEL), interval(...):
            out = inp

    test = stencil_factory.from_origin_domain(
        func, (0, 0, 0), domain=grid_indexing.domain
    )

    build_report = stencil_factory.build_report(key="parse_time")
    assert "func" in build_report

    inp = utils.make_storage(ones, grid_indexing, stencil_config, dtype=float)
    out = utils.make_storage(empty, grid_indexing, stencil_config, dtype=float)

    test(inp, out)
    exec_report = stencil_factory.exec_report()
    assert "func" in exec_report


@pytest.mark.parametrize("klevel,expected_origin_k", [(None, 0), (1, 1), (30, 30)])
def test_grid_indexing_get_2d_compute_origin_domain(
    klevel: int | None,
    expected_origin_k: int,
):
    indexing = GridIndexing(
        domain=(12, 12, 79),
        n_halo=3,
        south_edge=True,
        north_edge=True,
        west_edge=True,
        east_edge=True,
    )

    if klevel is None:
        origin, domain = indexing.get_2d_compute_origin_domain()
    else:
        origin, domain = indexing.get_2d_compute_origin_domain(klevel)

    assert origin[2] == expected_origin_k
    assert domain[2] == 1
