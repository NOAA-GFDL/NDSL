import re
from typing import TypeAlias

import pytest
from dace.frontend.python.common import DaceSyntaxError

from ndsl import (
    Backend,
    DataDimensionsField,
    NDSLRuntime,
    Quantity,
    QuantityFactory,
    StencilFactory,
    orchestrate,
)
from ndsl.boilerplate import (
    get_factories_single_tile,
    get_factories_single_tile_orchestrated,
)
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import Float, FloatField, Int


Tracers = DataDimensionsField.declare()
TracersAndPlumes = DataDimensionsField.declare()


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
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ) -> None:
        super().__init__(stencil_factory)
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="bad_call",
        )
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
    ) -> None:
        # Literal access, multi-axis access and external indexation
        self._the_stencil_4D(in_tracers, out_field, self._my_local)
        self._the_stencil_5D(in_tracers_and_plumes, out_field, self._my_local)

        # Blind loop on size
        for i_tracer in range(Tracers.size(0)):
            self._the_stencil_3D(
                in_tracers[:, :, :, i_tracer], out_field, self._my_local
            )

        # Direct variable access
        my_index = 5
        self._the_stencil_3D(in_tracers[:, :, :, my_index], out_field, self._my_local)

        # Name based access
        self._the_stencil_3D(
            in_tracers[:, :, :, Tracers.index("H")], out_field, self._my_local
        )

    def bad_call(
        self, in_tracers: Quantity, in_tracers_and_plumes, out_field: Quantity
    ) -> None:

        another_index = Tracers.index("H")  # BAD in orchestration
        self._the_stencil_3D(
            in_tracers[:, :, :, another_index], out_field, self._my_local
        )


Domain: TypeAlias = tuple[int, int, int]


@pytest.fixture
def domain() -> tuple[int, int, int]:
    return (2, 2, 5)


def test_data_dimensions_registration_errors(domain: Domain) -> None:
    _, quantity_factory = get_factories_single_tile(
        domain[0], domain[1], domain[2], 0, backend=Backend("st:python:cpu:IJK")
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


def test_data_dimensions_fields_with_stencil_backend(domain: Domain) -> None:
    stencil_factory, quantity_factory = get_factories_single_tile(
        domain[0], domain[1], domain[2], 0, backend=Backend("st:python:cpu:IJK")
    )

    setup_data_dimensions(quantity_factory)

    tracers_quantity = quantity_factory.ones(
        dims=[I_DIM, J_DIM, K_DIM, "tracers"], units="inputs"
    )
    tracers_and_plume_quantity = quantity_factory.full(
        dims=[I_DIM, J_DIM, K_DIM, "tracers", "plumes"], units="inputs", value=2
    )

    out_arr = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], units="outputs")

    code = Code(stencil_factory, quantity_factory)
    code(tracers_quantity, tracers_and_plume_quantity, out_arr)


def test_data_dimensions_fields_with_orchestrated_backend(domain: Domain) -> None:
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend=Backend("orch:dace:cpu:IJK")
    )

    setup_data_dimensions(quantity_factory)

    tracers_quantity = quantity_factory.ones(
        dims=[I_DIM, J_DIM, K_DIM, "tracers"], units="inputs"
    )
    tracers_and_plume_quantity = quantity_factory.full(
        dims=[I_DIM, J_DIM, K_DIM, "tracers", "plumes"], units="inputs", value=2
    )

    out_arr = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], units="outputs")

    code = Code(stencil_factory, quantity_factory)
    code(tracers_quantity, tracers_and_plume_quantity, out_arr)

    with pytest.raises(
        (
            DaceSyntaxError,
            TypeError,
        )
    ):
        code.bad_call(tracers_quantity, tracers_and_plume_quantity, out_arr)


def test_data_dimensions_fields_functions(domain: Domain) -> None:
    _, quantity_factory = get_factories_single_tile(
        domain[0], domain[1], domain[2], 0, backend=Backend("orch:dace:cpu:IJK")
    )

    setup_data_dimensions(quantity_factory)

    assert Tracers.index("H") == 7
    assert TracersAndPlumes.size(0) == Tracers.size(0)
    assert TracersAndPlumes.size(1) == 3


@pytest.mark.xfail(
    raises=RuntimeError,
    reason="Data dimension field declaration has to be on one line (for now), see [missing issue].",
)
def test_data_dim_multi_line_declare(domain: Domain) -> None:
    _, quantity_factory = get_factories_single_tile(
        nx=domain[0],
        ny=domain[1],
        nz=domain[2],
        nhalo=0,
        backend=Backend("st:dace:cpu:IJK"),
    )
    quantity_factory.update_data_dimensions({"data_dimension": 3})

    # The following currently fails because DataDimensionField declaration
    # only works if declared on one line, i.e.
    #   FloatField_with_data_dimension = DataDimensionField.declare()
    # (with a separate declaration) would just work.
    FloatField_with_data_dimension = DataDimensionsField.declare_and_register(
        quantity_factory, ["data_dimension"], dtype=Float
    )


def test_data_dim_ndsl_type(domain: Domain) -> None:
    _, quantity_factory = get_factories_single_tile(
        nx=domain[0],
        ny=domain[1],
        nz=domain[2],
        nhalo=0,
        backend=Backend("st:dace:cpu:IJK"),
    )

    with pytest.raises(TypeError, match="Wrong size type for data dimension"):
        quantity_factory.update_data_dimensions({"data_dimension": Int(3)})
