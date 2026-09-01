import pytest
from dace import nodes, unroll

from ndsl import (
    Backend,
    NDSLRuntime,
    OptimizationConfig,
    StencilFactory,
    orchestrate,
    stencils,
)
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import Float, FloatField
from tests.dsl.dace.stree import get_SDFG_and_purge
from tests.dsl.dace.stree.optimizations import Factories


def mult_stencil(inout_field: FloatField, scalar: Float):
    with computation(PARALLEL), interval(...):
        inout_field = inout_field * scalar


class OrchestratedCode(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory) -> None:
        config = OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=True,
                merger=OptimizationConfig.Tree.Merger(enabled=True),
            )
        )
        super().__init__(stencil_factory, config)

        methods_to_orchestrate = [
            "happy_case",
            "dace_auto_grid",
            "reuse_of_scalars",
            "reuse_of_scalars_in_inputs",
            "block_by_conditional",
        ]

        for method in methods_to_orchestrate:
            orchestrate(
                obj=self,
                config=stencil_factory.config.dace_config,
                method_to_orchestrate=method,
                optimization_config=config,
            )

        self._copy_stencil = stencil_factory.from_dims_halo(
            stencils.copy, [I_DIM, J_DIM, K_DIM]
        )
        self._mult_stencil = stencil_factory.from_dims_halo(
            mult_stencil, [I_DIM, J_DIM, K_DIM]
        )
        self._fillc_value = [True, False]

    def happy_case(
        self, scalar: float, in_field: FloatField, out_field: FloatField
    ) -> None:
        self._copy_stencil(in_field, out_field)
        local_scalar = scalar * 2
        self._mult_stencil(in_field, local_scalar)

    def dace_auto_grid(
        self, scalar: float, in_field: FloatField, out_field: FloatField
    ) -> None:
        in_field[:] = 43.0
        self._mult_stencil(in_field, scalar)

    def reuse_of_scalars(
        self, scalar: float, in_field: FloatField, out_field: FloatField
    ) -> None:
        for n in unroll(range(2)):
            fillc = self._fillc_value[n]
            if fillc:
                self._mult_stencil(in_field, scalar)

    def reuse_of_scalars_in_inputs(
        self, scalar: float, in_field: FloatField, out_field: FloatField
    ):
        tmp_scalar = scalar * 2.0
        self._mult_stencil(in_field, tmp_scalar)
        tmp_scalar = scalar * 2.0
        self._mult_stencil(in_field, tmp_scalar)

    def block_by_conditional(
        self, scalar: float, in_field: FloatField, out_field: FloatField
    ):
        self._mult_stencil(in_field, scalar)
        if scalar > 2:
            tmp_scalar = scalar * 2.0
            self._mult_stencil(in_field, tmp_scalar)


class TestStreeExtractOffgridTasklets:
    @pytest.fixture(params=["orch:dace:cpu:IJK", "orch:dace:cpu:KJI"])
    def factories(self, request: pytest.FixtureRequest) -> Factories:
        domain = (3, 3, 4)
        return get_factories_single_tile(
            domain[0], domain[1], domain[2], 0, backend=Backend(request.param)
        )

    def test_happy_case(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")
        in_flag = True

        code.happy_case(in_flag, in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        assert len(all_maps) == 1  # all merged and collapsed

    def test_dace_auto_grid(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")
        in_flag = True

        code.dace_auto_grid(in_flag, in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        assert len(all_maps) == 2  # not merged

    def test_reuse_of_scalars(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")
        scalar = 2.0

        code.reuse_of_scalars(scalar, in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        assert len(all_maps) == 1  # merged

        assert (in_quantity.field[:] == 2.0).all()

    def test_reuse_of_sclars_in_inputs(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")
        scalar = 2.0

        code.reuse_of_scalars_in_inputs(scalar, in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        assert len(all_maps) == 1  # merged

        assert (in_quantity.field[:] == 16.0).all()
