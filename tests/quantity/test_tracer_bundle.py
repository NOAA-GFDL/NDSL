"""This module includes unit tests for the `TracerBundle` class."""

import pytest

from ndsl.boilerplate import get_factories_single_tile
from ndsl.quantity.tracer_bundle import Tracer, TracerBundle
from ndsl.quantity.tracer_bundle_type import TracerBundleTypeRegistry


_TRACER_BUNDLE_TYPENAME = "TracerBundleTypeUnitTests"
TracerBundleTypeRegistry.register(_TRACER_BUNDLE_TYPENAME, 5)


def test_query_size_of_bundle_with_len() -> None:
    _, quantity_factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME, quantity_factory=quantity_factory
    )

    assert len(tracers) == 5


def test_access_tracer_by_name() -> None:
    _, factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=factory,
        mapping={"ice": 3, "vapor": 1},
    )

    assert isinstance(tracers.ice, Tracer)
    assert isinstance(tracers.vapor, Tracer)
    assert tracers.snow is None


def test_access_tracer_by_index() -> None:
    _, factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=factory,
        mapping={"ice": 3, "vapor": 1},
    )

    assert isinstance(tracers[0], Tracer)

    with pytest.raises(IndexError, match=".*select tracers in range.*"):
        tracers[len(tracers)]

    with pytest.raises(IndexError, match=".*select tracers in range.*"):
        tracers[-1]


def test_same_tracer_by_name_and_index() -> None:
    _, factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=factory,
        mapping={"ice": 3, "vapor": 1},
    )

    ice_tracer = tracers.ice
    other_ice_tracer = tracers[3]

    assert ice_tracer is other_ice_tracer


def test_units_are_propagated_to_tracers() -> None:
    _, factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    unit = "u"
    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=factory,
        unit=unit,
        mapping={"ice": 3, "vapor": 1},
    )

    ice_tracer = tracers.ice
    other_ice_tracer = tracers[3]

    assert ice_tracer.units == unit
    assert other_ice_tracer.units == unit


def test_loop_over_all_tracers_index() -> None:
    _, factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=factory,
        mapping={"ice": 3, "vapor": 1},
    )

    for index in range(len(tracers)):
        assert isinstance(tracers[index], Tracer)


def test_loop_over_all_tracers() -> None:
    _, factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    tracers = TracerBundle(
        type_name=_TRACER_BUNDLE_TYPENAME,
        quantity_factory=factory,
        mapping={"ice": 3, "vapor": 1},
    )

    for tracer in tracers:
        assert isinstance(tracer, Tracer)
