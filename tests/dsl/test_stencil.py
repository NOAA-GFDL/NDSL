from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gt4py.storage import empty, ones

from ndsl import (
    CompilationConfig,
    FrozenStencil,
    GridIndexing,
    StencilConfig,
    StencilFactory,
)
from ndsl.dsl.gt4py import FORWARD, PARALLEL, Field, computation, interval
from ndsl.dsl.typing import (
    FloatField,
    FloatFieldIJ,
    FloatFieldIJ32,
    FloatFieldIJ64,
    IntFieldIJ,
    IntFieldIJ32,
    IntFieldIJ64,
)
from ndsl.quantity import Quantity
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


def copy_stencil(q_in: FloatField, q_out: FloatField):  # type: ignore
    with computation(PARALLEL), interval(...):
        q_out[0, 0, 0] = q_in


@pytest.mark.parametrize(
    "extent,dimensions,domain,call_count",
    [
        ((20, 20, 30), ["x", "y", "z"], (20, 20, 20), 0),
        ((20, 20), ["x", "y"], (20, 20, 30), 0),
        ((20, 20), ["x_interface", "y"], (20, 20, 30), 0),
        ((20, 20), ["x", "y_interface"], (20, 20, 30), 0),
        ((20,), ["z"], (20, 20, 10), 0),
        ((20,), ["z_interface"], (20, 20, 10), 0),
        ((15, 20, 30), ["x", "y", "z"], (20, 20, 30), 1),
        ((20, 15, 30), ["x", "y", "z"], (20, 20, 30), 1),
        ((20, 20, 15), ["x", "y", "z"], (20, 20, 30), 1),
    ],
)
def test_domain_size_comparison(
    extent: tuple[int],
    dimensions: list[str],
    domain: tuple[int],
    call_count: int,
):
    quantity = Quantity(np.zeros(extent), dimensions, "n/a", extent=extent)
    stencil = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=domain,
        stencil_config=MagicMock(spec=StencilConfig()),
    )
    # with expectation:
    warning_mock = MagicMock()
    with patch("ndsl.ndsl_log.warning", warning_mock):
        stencil._validate_quantity_sizes(quantity)

    assert warning_mock.call_count == call_count


def two_dim_temporaries_stencil(q_out: FloatField) -> None:
    with computation(FORWARD), interval(0, 1):
        tmp_2d_fij: FloatFieldIJ = 1.0
        tmp_2d_fij32: FloatFieldIJ32 = 2.0
        tmp_3d_fij64: FloatFieldIJ64 = 3.0
        tmp_3d_iij: IntFieldIJ = 4
        tmp_3d_iij32: IntFieldIJ32 = 5
        tmp_3d_iij64: IntFieldIJ64 = 6

    with computation(PARALLEL), interval(...):
        q_out = (
            tmp_2d_fij
            + tmp_2d_fij32
            + tmp_3d_fij64
            + tmp_3d_iij
            + tmp_3d_iij32
            + tmp_3d_iij64
        )


@pytest.mark.parametrize(
    "extent,dimensions,domain,call_count",
    [
        ((2, 2, 5), ["x", "y", "z"], (2, 2, 5), 0),
    ],
)
def test_stencil_2D_temporaries(
    extent: tuple[int],
    dimensions: list[str],
    domain: tuple[int],
    call_count: int,
) -> None:
    quantity = Quantity(np.zeros(extent), dimensions, "n/a", extent=extent)
    stencil = FrozenStencil(
        two_dim_temporaries_stencil,
        origin=(0, 0, 0),
        domain=domain,
        stencil_config=MagicMock(spec=StencilConfig()),
    )
    stencil(quantity)
    assert (quantity.data[1, 1, :] == 21.0).all()
