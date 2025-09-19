"""This module includes integration tests for the `TracerBundle` class,
testing whole workflows."""

from typing import TypeAlias

from ndsl import StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField
from ndsl.quantity import TracerBundle, TracerBundleTypeRegistry


# workflow

# 1. register tracer bundle type from (name, size, dtype)
# 2. initialize tracer bundle from (type_name, quantity_factory, [mapping, unit])
#    - size can be derived from registered type (via type_name)
#    - dtype can be derived from registered type (via type_name)

_TRACER_BUNDLE_TYPENAME = "TracerBundleTypeWorkflowTests"
_TracerBundleType: TypeAlias = TracerBundleTypeRegistry.register(
    _TRACER_BUNDLE_TYPENAME, 5
)


def copy_into_tracer(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(...):
        out_field = in_field


def loop_over_tracers(out_field: _TracerBundleType, n_tracers: int):
    with computation(PARALLEL), interval(...):
        n = 0
        while n < n_tracers:
            out_field[0, 0, 0][n] = 42


class Code:
    def __init__(self, stencil_factory: StencilFactory):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["tracers"],
        )
        self._copy_into_tracer_stencil = stencil_factory.from_dims_halo(
            func=copy_into_tracer,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self._loop_over_tracers_stencil = stencil_factory.from_dims_halo(
            func=loop_over_tracers,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, in_field: FloatField, tracers: _TracerBundleType):
        # single tracer representing all memory
        tracers.ice.data[:] = 20

        # single tracer sliced into the compute domain
        tracers.ice.field[:] = 10

        self._copy_into_tracer_stencil(in_field, tracers.vapor)
        self._loop_over_tracers_stencil(tracers, len(tracers))

        for index in range(len(tracers)):
            self._copy_into_tracer_stencil(in_field, tracers[index])


def test_stencil_workflow() -> None:
    backend = "dace:cpu"
    domain = (2, 3, 4)
    halo_points = 1

    stencil_factory, quantity_factory = get_factories_single_tile(
        domain[0], domain[1], domain[2], halo_points, backend=backend
    )

    field = quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a")
    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=quantity_factory,
        mapping={"ice": 3, "vapor": 0},
    )

    code = Code(stencil_factory)
    code(field, tracers)
