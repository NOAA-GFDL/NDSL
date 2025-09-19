import pytest

from ndsl.boilerplate import get_factories_single_tile
from ndsl.quantity import TracerBundle
from ndsl.quantity.tracer_bundle import Tracer


def test_query_size_of_bundle_with_len() -> None:
    _, quantity_factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    size = 5
    tracers = TracerBundle(quantity_factory=quantity_factory, size=size)

    assert len(tracers) == size


def test_access_tracer_by_name() -> None:
    _, factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    tracers = TracerBundle(
        quantity_factory=factory, size=5, mapping={"ice": 3, "vapor": 1}
    )

    assert isinstance(tracers.ice, Tracer)
    assert isinstance(tracers.vapor, Tracer)
    assert tracers.snow is None


def test_access_tracer_by_index() -> None:
    _, factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    tracers = TracerBundle(
        quantity_factory=factory, size=5, mapping={"ice": 3, "vapor": 1}
    )

    assert isinstance(tracers[0], Tracer)

    with pytest.raises(ValueError, match=".*select tracers in range.*"):
        tracers[5]

    with pytest.raises(ValueError, match=".*select tracers in range.*"):
        tracers[-1]


def test_same_tracer_by_name_and_index() -> None:
    _, factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    tracers = TracerBundle(
        quantity_factory=factory, size=5, mapping={"ice": 3, "vapor": 1}
    )

    ice_tracer = tracers.ice
    other_ice_tracer = tracers[3]

    assert ice_tracer is other_ice_tracer


def test_loop_over_all_tracers() -> None:
    _, factory = get_factories_single_tile(nx=2, ny=3, nz=4, nhalo=1)
    tracers = TracerBundle(
        quantity_factory=factory, size=5, mapping={"ice": 3, "vapor": 1}
    )

    for index in range(len(tracers)):
        assert isinstance(tracers[index], Tracer)
