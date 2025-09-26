"""This module includes integration tests for the `TracerBundle` class,
testing whole workflows."""

import pytest

from ndsl import Quantity, StencilFactory, orchestrate
from ndsl.boilerplate import (
    get_factories_single_tile,
    get_factories_single_tile_orchestrated,
)
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
_TracerBundleType = TracerBundleTypeRegistry.register(_TRACER_BUNDLE_TYPENAME, 5)


def copy_into_tracer(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(...):
        out_field = in_field


def loop_over_tracers(
    out_field: _TracerBundleType, n_tracers: int, fill_value: int = 42
):
    with computation(PARALLEL), interval(...):
        n = 0
        while n < n_tracers:
            out_field[0, 0, 0][n] = fill_value
            n = n + 1


class IceTracerSetup:
    def __init__(self, stencil_factory: StencilFactory):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["tracers"],
        )

    # def __call__(self, quantity: Quantity):
    #    # single tracer representing all memory
    #    quantity.data[:] = 20
    #    quantity.field[:] = 10

    def __call__(self, tracers: TracerBundle):
        # # single tracer representing all memory
        # tracers.ice.data[:] = 20
        # tracers.ice.field[:] = 10

        tracers.fill_tracer_by_name("ice", value=20)
        # single tracer sliced into the compute domain
        tracers.fill_tracer_by_name("ice", value=10, compute_domain_only=True)


@pytest.mark.parametrize("backend", ("numpy", "dace:cpu"))
def test_stencil_ice_tracer_setup(backend) -> None:
    domain = (2, 3, 4)
    halo_points = 1

    stencil_factory, quantity_factory = get_factories_single_tile(
        domain[0], domain[1], domain[2], halo_points, backend=backend
    )

    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=quantity_factory,
        mapping={"ice": 3, "vapor": 0},
    )

    setup = IceTracerSetup(stencil_factory)
    setup(tracers)

    assert (tracers.ice.data[0] == 20).all()  # check a part of the halo
    assert (tracers.ice.field[:] == 10).all()  # check the compute domain


def test_orchestrated_ice_tracer_setup() -> None:
    domain = (2, 3, 4)
    halo_points = 1

    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0],
        domain[1],
        domain[2],
        halo_points,
    )

    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=quantity_factory,
        mapping={"ice": 3, "vapor": 0},
    )

    setup = IceTracerSetup(stencil_factory)
    setup(tracers)

    assert (tracers.ice.data[0] == 20).all()  # check a part of the halo
    assert (tracers.ice.field[:] == 10).all()  # check the compute domain


class CopyIntoVaporTracer:
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

    def __call__(self, tracers: TracerBundle, vapor_field: Quantity):
        tracers.vapor.data = 0  # __g_tracers_vapor_data

        self._copy_into_tracer_stencil(vapor_field, tracers.vapor)  # __g_tracers_vapor


@pytest.mark.parametrize("backend", ("numpy", "dace:cpu"))
def test_stencil_copy_into_vapor_tracer(backend) -> None:
    domain = (2, 3, 4)
    halo_points = 1

    stencil_factory, quantity_factory = get_factories_single_tile(
        domain[0], domain[1], domain[2], halo_points, backend=backend
    )

    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=quantity_factory,
        mapping={"ice": 3, "vapor": 0},
    )
    vapor_setup = CopyIntoVaporTracer(stencil_factory)

    field = quantity_factory.ones(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a")
    vapor_setup(tracers, field)
    assert (tracers.vapor.field[:] == 1).all()


class ResetTracers:
    def __init__(self, stencil_factory: StencilFactory):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["tracers"],
        )
        self._loop_over_tracers_stencil = stencil_factory.from_dims_halo(
            func=loop_over_tracers,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, tracers: TracerBundle, fill_value: int) -> None:
        self._loop_over_tracers_stencil(tracers, len(tracers), fill_value)


@pytest.mark.parametrize(
    "backend",
    (
        "dace:cpu",
        pytest.param(
            "numpy",
            marks=pytest.mark.xfail(reason="numpy backend cannot handle access of [n]"),
        ),
    ),
)
def test_stencil_reset_tracers(backend) -> None:
    domain = (2, 3, 4)
    halo_points = 1

    stencil_factory, quantity_factory = get_factories_single_tile(
        domain[0], domain[1], domain[2], halo_points, backend=backend
    )

    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=quantity_factory,
        mapping={"ice": 3, "vapor": 0},
    )
    reset_tracers = ResetTracers(stencil_factory)

    fill_value = 42
    reset_tracers(tracers, fill_value)
    for tracer in tracers:
        assert (tracer.field[:] == fill_value).all()


class CopyIntoAllTracers:
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

    def __call__(self, tracers: TracerBundle, field: Quantity):
        for tracer in tracers:
            self._copy_into_tracer_stencil(field, tracer)


@pytest.mark.parametrize("backend", ("numpy", "dace:cpu"))
def test_stencil_copy_into_all_tracer(backend) -> None:
    domain = (2, 3, 4)
    halo_points = 1

    stencil_factory, quantity_factory = get_factories_single_tile(
        domain[0], domain[1], domain[2], halo_points, backend=backend
    )

    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=quantity_factory,
        mapping={"ice": 3, "vapor": 0},
    )
    tracer_setup = CopyIntoAllTracers(stencil_factory)

    field = quantity_factory.ones(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a")
    tracer_setup(tracers, field)

    for tracer in tracers:
        assert (tracer.field[:] == 1).all()
