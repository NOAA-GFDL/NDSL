import pytest
from dace import nodes

from ndsl import Backend, NDSLRuntime, StencilFactory, orchestrate, stencils
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import FloatField
from tests.dsl.dace.stree import StreeOptimization, get_SDFG_and_purge
from tests.dsl.dace.stree.optimizations import Factories


class OrchestratedCode(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory) -> None:
        super().__init__(stencil_factory)

        methods_to_orchestrate = [
            "happy_case",
            "happy_case_2",
            "blocked_by_else",
            "blocked_by_other_nodes",
        ]

        for method in methods_to_orchestrate:
            orchestrate(
                obj=self,
                config=stencil_factory.config.dace_config,
                method_to_orchestrate=method,
            )

        self._copy_stencil = stencil_factory.from_dims_halo(
            func=stencils.copy, compute_dims=[I_DIM, J_DIM, K_DIM]
        )

    def happy_case(self, in_field: FloatField, out_field: FloatField) -> None:
        if in_field[0, 0, 0] > 0:
            self._copy_stencil(in_field, out_field)
        self._copy_stencil(in_field, out_field)

    def happy_case_2(self, in_field: FloatField, out_field: FloatField) -> None:
        if not in_field[0, 0, 0] > 0:
            self._copy_stencil(in_field, out_field)
        self._copy_stencil(in_field, out_field)

    def blocked_by_else(self, in_field: FloatField, out_field: FloatField) -> None:
        self._copy_stencil(in_field, out_field)

        if in_field[0, 0, 0] > 0:
            self._copy_stencil(in_field, out_field)
        else:
            self._copy_stencil(out_field, in_field)

    def blocked_by_other_nodes(
        self, in_field: FloatField, out_field: FloatField
    ) -> None:
        if in_field[0, 0, 0] > 0:
            in_field[:] = 42.0
            self._copy_stencil(in_field, out_field)
        self._copy_stencil(in_field, out_field)


class TestStreeInlineOffgridConditionals:
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

        with StreeOptimization():
            code.happy_case(in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        assert len(all_maps) == 1  # all merged and collapsed

    def test_happy_case_2(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            code.happy_case_2(in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        assert len(all_maps) == 1  # all merged and collapsed

    def test_blocked_by_else(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            code.blocked_by_else(in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        assert len(all_maps) == 3  # 3 * IJK/KJI

    def test_blocked_by_other_nodes(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            code.blocked_by_other_nodes(in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]

        # ⚠️ Dev note:
        # This should be just `assert len(all_maps) == 2`, but currently, the K-loops
        # can't merge because the K-iterators are different. To be fixed (and simplified
        # here) with a subsequent commit.
        assert len(all_maps) == 3
