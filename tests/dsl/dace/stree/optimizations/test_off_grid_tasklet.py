import pytest
from dace import nodes

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
from ndsl.dsl.typing import FloatField
from ndsl.stencils import multiply_to_self
from tests.dsl.dace.stree import get_SDFG_and_purge
from tests.dsl.dace.stree.optimizations import Factories


class OrchestratedCode(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory) -> None:
        config = OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=True,
                merger=OptimizationConfig.Tree.Merger(enabled=True),
            )
        )
        super().__init__(stencil_factory, config)

        methods_to_orchestrate = ["happy_case", "dace_auto_grid"]

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
            multiply_to_self, [I_DIM, J_DIM, K_DIM]
        )

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


class TestStreeExtractOffgridConditionals:
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
