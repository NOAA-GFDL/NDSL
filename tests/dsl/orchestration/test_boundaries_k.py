import numpy as np
import pytest

from ndsl import Backend, NDSLRuntime, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, computation, interval
from ndsl.dsl.typing import FloatField
from tests.dsl.dace.stree.optimizations import Factories


def accumulate_down(in_field: FloatField, out_field: FloatField) -> None:  # type: ignore
    with computation(BACKWARD):
        # handle top layer separately
        with interval(-1, None):
            out_field = in_field

        # accumulate "downwards"
        with interval(0, -1):
            out_field = out_field[0, 0, 1] + in_field


def accumulate_down_from_interface_field(interface_field: FloatField, out_field: FloatField) -> None:  # type: ignore
    with computation(BACKWARD):
        # handle top layer separately
        with interval(-1, None):
            out_field = interface_field + interface_field[0, 0, 1]

        # accumulate "downwards"
        with interval(0, -1):
            out_field = out_field[0, 0, 1] + interface_field


def accumulate_on_interface(interface_field: FloatField, out_field: FloatField) -> None:  # type: ignore
    with computation(BACKWARD):
        # handle top layer separately
        with interval(-2, -1):
            out_field = interface_field + interface_field[0, 0, 1]

        # accumulate "downwards"
        with interval(0, -2):
            out_field = out_field[0, 0, 1] + interface_field


def accumulate_up(in_field: FloatField, out_field: FloatField) -> None:  # type: ignore
    with computation(FORWARD):
        # handle bottom layer separately
        with interval(0, 1):
            out_field = in_field

        # accumulate "upwards"
        with interval(1, None):
            out_field = out_field[0, 0, -1] + in_field


def accumulate_up_interface(in_field: FloatField, interface_field: FloatField) -> None:  # type: ignore
    with computation(FORWARD):
        # handle bottom layer separately
        with interval(0, 1):
            interface_field = in_field

        # accumulate "upwards"
        with interval(1, None):
            interface_field = interface_field[0, 0, -1] + in_field[0, 0, -1]


class OrchestratedCode(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory) -> None:
        super().__init__(stencil_factory)

        methods_to_orchestrate = [
            "accumulate_down",
            "accumulate_down_from_interface_field",
            "accumulate_on_interface",
            "accumulate_up",
            "accumulate_up_interface",
        ]

        for method in methods_to_orchestrate:
            orchestrate(
                obj=self,
                method_to_orchestrate=method,
                config=stencil_factory.config.dace_config,
            )

        self._accumulate_down = stencil_factory.from_dims_halo(
            func=accumulate_down, compute_dims=(I_DIM, J_DIM, K_DIM)
        )

        self._accumulate_down_from_interface_field = stencil_factory.from_dims_halo(
            func=accumulate_down_from_interface_field,
            compute_dims=(I_DIM, J_DIM, K_DIM),
        )

        self._accumulate_on_interface = stencil_factory.from_dims_halo(
            func=accumulate_on_interface, compute_dims=(I_DIM, J_DIM, K_INTERFACE_DIM)
        )

        self._accumulate_up = stencil_factory.from_dims_halo(
            func=accumulate_up, compute_dims=(I_DIM, J_DIM, K_DIM)
        )

        self._accumulate_up_interface = stencil_factory.from_dims_halo(
            func=accumulate_up_interface, compute_dims=(I_DIM, J_DIM, K_INTERFACE_DIM)
        )

    def accumulate_down(self, in_field: FloatField, out_field: FloatField) -> None:  # type: ignore
        self._accumulate_down(in_field, out_field)

    def accumulate_down_from_interface_field(self, interface_field: FloatField, out_field: FloatField) -> None:  # type: ignore
        self._accumulate_down_from_interface_field(interface_field, out_field)

    def accumulate_on_interface(self, interface_field: FloatField, out_field: FloatField) -> None:  # type: ignore
        self._accumulate_on_interface(interface_field, out_field)

    def accumulate_up(self, in_field: FloatField, out_field: FloatField) -> None:  # type: ignore
        self._accumulate_up(in_field, out_field)

    def accumulate_up_interface(self, in_field: FloatField, interface_field: FloatField) -> None:  # type: ignore
        self._accumulate_up_interface(in_field, interface_field)


class TestBoundariesK:
    @pytest.fixture(
        params=[
            "orch:dace:cpu:IJK",
            "orch:dace:cpu:KJI",
            "st:dace:cpu:IJK",
            "st:dace:cpu:KJI",
        ]
    )
    def factories(self, request: pytest.FixtureRequest) -> Factories:
        domain = (3, 4, 5)
        return get_factories_single_tile(
            nx=domain[0],
            ny=domain[1],
            nz=domain[2],
            nhalo=0,
            backend=Backend(request.param),
        )

    def test_accumulate_down(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        in_field = quantity_factory.ones((I_DIM, J_DIM, K_DIM), units="")
        out_field = quantity_factory.zeros((I_DIM, J_DIM, K_DIM), units="")

        code.accumulate_down(in_field, out_field)
        assert np.array_equal(out_field.field[0, 0, :], [5, 4, 3, 2, 1])

    def test_accumulate_interface_field(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        interface_field = quantity_factory.ones(
            (I_DIM, J_DIM, K_INTERFACE_DIM), units=""
        )
        out_field = quantity_factory.zeros((I_DIM, J_DIM, K_DIM), units="")

        code.accumulate_down_from_interface_field(interface_field, out_field)
        assert np.array_equal(out_field.field[0, 0, :], [6, 5, 4, 3, 2])

    def test_accumulate_interface_domain(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        interface_field = quantity_factory.ones(
            (I_DIM, J_DIM, K_INTERFACE_DIM), units=""
        )
        out_field = quantity_factory.zeros((I_DIM, J_DIM, K_DIM), units="")

        code.accumulate_on_interface(interface_field, out_field)
        assert np.array_equal(out_field.field[0, 0, :], [6, 5, 4, 3, 2])

    def test_accumulate_up(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        in_field = quantity_factory.ones((I_DIM, J_DIM, K_DIM), units="")
        out_field = quantity_factory.zeros((I_DIM, J_DIM, K_DIM), units="")

        code.accumulate_up(in_field, out_field)
        assert np.array_equal(out_field.field[0, 0, :], [1, 2, 3, 4, 5])

    def test_accumulate_up_interface(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        in_field = quantity_factory.ones((I_DIM, J_DIM, K_DIM), units="")
        interface_field = quantity_factory.zeros(
            (I_DIM, J_DIM, K_INTERFACE_DIM), units=""
        )

        code.accumulate_up_interface(in_field, interface_field)
        assert np.array_equal(interface_field.field[0, 0, :], [1, 2, 3, 4, 5, 6])
