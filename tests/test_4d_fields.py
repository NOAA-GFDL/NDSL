from ndsl import StencilFactory
from ndsl.boilerplate import (
    get_factories_single_tile,
    get_factories_single_tile_orchestrated,
)
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval, max
from ndsl.dsl.typing import Float, FloatField, set_4d_field_size


TRACER_DIM = "n_tracers"
FloatFieldTracer = set_4d_field_size(9, Float)
ntracers = 9
ntke = 8
fill_value = 42.0


def sample_4d_stencil(q_in: FloatFieldTracer, q_out: FloatField):
    from __externals__ import ntke

    with computation(PARALLEL), interval(...):
        q_out = max(q_in[0, 0, 0][ntke], 1.0e-9)


class SampleCalculation:
    def __init__(self, stencil_factory: StencilFactory, *, ntke: int):
        self._test_calc = stencil_factory.from_dims_halo(
            func=sample_4d_stencil,
            externals={
                "ntke": ntke,
            },
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, q_in: FloatFieldTracer, q_out: FloatField):
        self._test_calc(q_in, q_out)


def test_non_orchestrated_call() -> None:
    stencil_factory, quantity_factory = get_factories_single_tile(24, 24, 91, 3)
    quantity_factory.set_extra_dim_lengths(
        **{
            TRACER_DIM: ntracers,
        }
    )

    q_out = quantity_factory.zeros(
        [X_DIM, Y_DIM, Z_DIM],
        units="unknown",
    )
    q_in = quantity_factory.zeros(
        [X_DIM, Y_DIM, Z_DIM, TRACER_DIM],
        units="unknown",
    )
    q_in.field[:, :, :, ntke] = fill_value

    calc = SampleCalculation(stencil_factory, ntke=ntke)
    calc(q_in, q_out)
    assert (q_out.field[:, :, :] == fill_value).all()


def test_orchestrated_call() -> None:
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        24, 24, 91, 3
    )
    quantity_factory.set_extra_dim_lengths(
        **{
            TRACER_DIM: ntracers,
        }
    )

    q_out = quantity_factory.zeros(
        [X_DIM, Y_DIM, Z_DIM],
        units="unknown",
    )
    q_in = quantity_factory.zeros(
        [X_DIM, Y_DIM, Z_DIM, TRACER_DIM],
        units="unknown",
    )
    q_in.field[:, :, :, ntke] = fill_value

    calc = SampleCalculation(stencil_factory, ntke=ntke)
    calc(q_in, q_out)
    assert (q_out.field[:, :, :] == fill_value).all()
