from typing import Optional

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

        methods_to_orchestrate = [
            "happy_case",
            "happy_case_2",
            "simple_if_else",
            "blocked_by_other_nodes",
            "simple_if_elseif_else",
            "optional_field",
        ]

        for method in methods_to_orchestrate:
            orchestrate(
                obj=self,
                config=stencil_factory.config.dace_config,
                method_to_orchestrate=method,
                optimization_config=config,
            )

        self._copy_stencil = stencil_factory.from_dims_halo(
            func=stencils.copy, compute_dims=[I_DIM, J_DIM, K_DIM]
        )

    def happy_case(
        self, flag: bool, in_field: FloatField, out_field: FloatField
    ) -> None:
        if flag:
            self._copy_stencil(in_field, out_field)
        self._copy_stencil(in_field, out_field)

    def happy_case_2(
        self, flag: bool, in_field: FloatField, out_field: FloatField
    ) -> None:
        if not flag:
            self._copy_stencil(in_field, out_field)
        self._copy_stencil(in_field, out_field)

    def simple_if_else(
        self, flag: bool, in_field: FloatField, out_field: FloatField
    ) -> None:
        self._copy_stencil(in_field, out_field)

        if flag:
            self._copy_stencil(in_field, out_field)
        else:
            self._copy_stencil(out_field, in_field)

    def simple_if_elseif_else(
        self, flag: bool, flag_b: bool, in_field: FloatField, out_field: FloatField
    ) -> None:
        self._copy_stencil(in_field, out_field)

        if flag:
            self._copy_stencil(in_field, out_field)
        elif flag_b:
            self._copy_stencil(out_field, in_field)
        else:
            self._copy_stencil(out_field, in_field)

    def blocked_by_other_nodes(
        self, flag: bool, in_field: FloatField, out_field: FloatField
    ) -> None:
        if flag:
            in_field[:] = 42.0
            self._copy_stencil(in_field, out_field)
        self._copy_stencil(in_field, out_field)

    def optional_field(
        self,
        in_field: FloatField,
        out_field: FloatField,
        opt_field: Optional[FloatField] = None,
    ) -> None:
        self._copy_stencil(in_field, out_field)
        if opt_field is None:
            self._copy_stencil(out_field, in_field)
        else:
            self._copy_stencil(out_field, opt_field)


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
        in_flag = True

        code.happy_case(in_flag, in_quantity, out_quantity)

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
        in_flag = True

        code.happy_case_2(in_flag, in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        assert len(all_maps) == 1  # all merged and collapsed

    def test_if_else(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")
        in_flag = True

        code.simple_if_else(in_flag, in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        assert len(all_maps) == 1

    def test_if_elif_else(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")
        in_flag = True

        code.simple_if_elseif_else(in_flag, True, in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]

        # Dev note:
        #  ElseIf are parsed as Else then If. As long as it's the case
        #  the merging will work. BUT the proper `ElseIf` is not implemented
        assert len(all_maps) == 1

    def test_blocked_by_other_nodes(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")
        in_flag = True

        code.blocked_by_other_nodes(in_flag, in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]

        # ⚠️ Dev note:
        # The first node in_field[:] = 42.0 unroll as a 3-axis loop. This _would_
        # be mergeable but DaCe is not aware of the cartesianess and therefore names them
        # __i0, __i1, __i2 - which trips our merger (for now)
        assert len(all_maps) == 3

    def test_optional_field(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories

        code = OrchestratedCode(stencil_factory)
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        code.optional_field(in_quantity, out_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]

        assert len(all_maps) == 1
