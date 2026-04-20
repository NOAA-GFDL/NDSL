import re

import pytest

from ndsl import Backend, NDSLRuntime, Quantity, QuantityFactory, StencilFactory
from ndsl.boilerplate import (
    get_factories_single_tile,
    get_factories_single_tile_orchestrated,
)
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField
from ndsl.quantity.data_dimensions_field import DataDimensionsField


Tracers = DataDimensionsField.declare()
TracersAndPlumes = DataDimensionsField.declare()

_DOMAIN = (2, 2, 5)


def _the_stencil_5D(in_field: TracersAndPlumes, out_field: FloatField, add: FloatField):
    with computation(PARALLEL), interval(...):
        out_field = in_field.A[1, 1] + add


def _the_stencil_4D(in_tracers: Tracers, out_field: FloatField, add: FloatField):
    with computation(PARALLEL), interval(...):
        from __externals__ import C

        out_field = in_tracers[0, 0, 0][C] + add


def _the_stencil_3D(in_field: FloatField, out_field: FloatField, add: FloatField):
    with computation(PARALLEL), interval(...):
        out_field = in_field + add


SETUP_DDIMS_ONCE = False


def setup_data_dimensions(quantity_factory: QuantityFactory):
    quantity_factory.add_data_dimensions({"tracers": 8, "plumes": 3})

    # Make sure this is called once
    global SETUP_DDIMS_ONCE
    if SETUP_DDIMS_ONCE:
        return
    SETUP_DDIMS_ONCE = True

    mappings = {"A": 0, "C": 2, "D": 3, "G": 6, "H": 7}
    DataDimensionsField.register(Tracers, quantity_factory, ["tracers"], mappings)
    DataDimensionsField.register(
        TracersAndPlumes, quantity_factory, ["tracers", "plumes"], mappings
    )


class Code(NDSLRuntime):
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        super().__init__(stencil_factory)
        self._the_stencil_4D = stencil_factory.from_dims_halo(
            func=_the_stencil_4D,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals=Tracers.mapping,
        )
        self._the_stencil_3D = stencil_factory.from_dims_halo(
            func=_the_stencil_3D,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._the_stencil_5D = stencil_factory.from_dims_halo(
            func=_the_stencil_5D,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._my_local = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        self._my_local.field[:] = 2.0

    def __call__(
        self, in_tracers: Quantity, in_tracers_and_plumes, out_field: Quantity
    ):
        # Literal access, multi-axis access and external indexation
        self._the_stencil_4D(in_tracers, out_field, self._my_local)
        self._the_stencil_5D(in_tracers_and_plumes, out_field, self._my_local)

        # Blind loop on size
        for i_tracer in range(Tracers.size(0)):
            self._the_stencil_3D(
                in_tracers.data[:, :, :, i_tracer], out_field, self._my_local
            )

        # Direct variable access
        my_index = 5
        self._the_stencil_3D(
            in_tracers.data[:, :, :, my_index], out_field, self._my_local
        )

        # Name based access
        self._the_stencil_3D(
            in_tracers.data[:, :, :, Tracers.index("H")], out_field, self._my_local
        )

    def bad_call(
        self, in_tracers: Quantity, in_tracers_and_plumes, out_field: Quantity
    ):

        another_index = Tracers.index("H")  # BAD in orchestration
        self._the_stencil_3D(
            in_tracers.data[:, :, :, another_index], out_field, self._my_local
        )


def test_data_dimensions_registration_errors():
    _, quantity_factory = get_factories_single_tile(
        _DOMAIN[0], _DOMAIN[1], _DOMAIN[2], 0, backend=Backend("st:python:cpu:IJK")
    )
    with pytest.raises(
        KeyError,
        match=re.escape(
            'Data dimension axis "tracers" is not present in QuantityFactory. Use QuantityFactory.add_data_dimensions prior to registering field.'
        ),
    ):
        DataDimensionsField.register(
            TracersAndPlumes, quantity_factory, ["tracers"], {}
        )

    with pytest.raises(
        KeyError,
        match=re.escape(
            "Data dimension field Tracers is not registered. Call DataDimensionsField.register(Tracers)."
        ),
    ):
        Tracers.index("H")


def test_data_dimensions_fields_with_stencil_backend():
    stcil_fctry, qty_factry = get_factories_single_tile(
        _DOMAIN[0], _DOMAIN[1], _DOMAIN[2], 0, backend=Backend("st:python:cpu:IJK")
    )

    setup_data_dimensions(qty_factry)

    tracers_quantity = qty_factry.ones(
        dims=[I_DIM, J_DIM, K_DIM, "tracers"], units="inputs"
    )
    tracers_and_plume_quantity = qty_factry.full(
        dims=[I_DIM, J_DIM, K_DIM, "tracers", "plumes"], units="inputs", value=2
    )

    out_arr = qty_factry.zeros([I_DIM, J_DIM, K_DIM], units="outputs")

    code = Code(stcil_fctry, qty_factry)
    code(tracers_quantity, tracers_and_plume_quantity, out_arr)


def test_data_dimensions_fields_with_orchestrated_backend():
    stcil_fctry, qty_factry = get_factories_single_tile_orchestrated(
        _DOMAIN[0], _DOMAIN[1], _DOMAIN[2], 0, backend=Backend("orch:dace:cpu:IJK")
    )

    setup_data_dimensions(qty_factry)

    tracers_quantity = qty_factry.ones(
        dims=[I_DIM, J_DIM, K_DIM, "tracers"], units="inputs"
    )
    tracers_and_plume_quantity = qty_factry.full(
        dims=[I_DIM, J_DIM, K_DIM, "tracers", "plumes"], units="inputs", value=2
    )

    out_arr = qty_factry.zeros([I_DIM, J_DIM, K_DIM], units="outputs")

    code = Code(stcil_fctry, qty_factry)
    code(tracers_quantity, tracers_and_plume_quantity, out_arr)

    code.bad_call(tracers_quantity, tracers_and_plume_quantity, out_arr)


def test_data_dimensions_fields_functions():
    stcil_fctry, qty_factry = get_factories_single_tile(
        _DOMAIN[0], _DOMAIN[1], _DOMAIN[2], 0, backend=Backend("orch:dace:cpu:IJK")
    )

    setup_data_dimensions(qty_factry)

    assert Tracers.index("H") == 7
    assert TracersAndPlumes.size(0) == Tracers.size(0)
    assert TracersAndPlumes.size(1) == 3
