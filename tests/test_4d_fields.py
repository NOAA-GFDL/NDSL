# from pace.dsl.dace.orchestration import orchestrate
from ndsl import StencilFactory
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import Float, FloatField, set_4d_field_size


FloatFieldTracer = set_4d_field_size(9, Float)
TRACER_DIM = "n_tracers"


def sample_4d_stencil(
    q_in: FloatFieldTracer,
    q_out: FloatField,
):
    from __externals__ import ntke

    with computation(PARALLEL), interval(...):
        q_out = max(q_in[0, 0, 0][ntke], 1.0e-9)


class SampleCalculation:
    def __init__(
        self,
        stencil_factory: StencilFactory,
    ):
        self._test_calc = stencil_factory.from_dims_halo(
            func=sample_4d_stencil,
            externals={
                "ntke": 8,
            },
            compute_dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM],
        )

    def __call__(
        self,
        q_in: FloatFieldTracer,
        q_out: FloatField,
    ):
        self._test_calc(q_in, q_out)


def test_4d_stencil_call():
    ntracers = 9
    stencil_factory, quantity_factory = get_factories_single_tile(24, 24, 91, 3)
    quantity_factory.set_extra_dim_lengths(
        **{
            TRACER_DIM: ntracers,
        }
    )
    q_out = quantity_factory.zeros(
        [X_DIM, Y_DIM, Z_INTERFACE_DIM],
        units="unknown",
        dtype=Float,
    )
    q_in = quantity_factory.zeros(
        [X_DIM, Y_DIM, Z_DIM, TRACER_DIM],
        units="unknown",
        dtype=Float,
    )
    calc = SampleCalculation(stencil_factory)
    calc(q_in, q_out)
