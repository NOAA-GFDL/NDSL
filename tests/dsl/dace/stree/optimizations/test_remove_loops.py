from typing import TypeAlias

import pytest
from dace import nodes
from dace.sdfg.state import LoopRegion

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import Backend
from ndsl.constants import I_DIM, J_DIM, K_DIM, Float
from ndsl.dsl.gt4py import FORWARD, computation, interval
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from tests.dsl.dace.stree import StreeOptimization, get_SDFG_and_purge


def stencil_simple_2D_write(in_field: FloatField, out_fieldIJ: FloatFieldIJ) -> None:
    with computation(FORWARD), interval(0, 1):
        out_fieldIJ = in_field


def stencil_2D_write_at_K(in_field: FloatField, out_fieldIJ: FloatFieldIJ) -> None:
    with computation(FORWARD), interval(-1, None):
        out_fieldIJ = in_field


def stencil_forward_at_K(in_field: FloatField, out_field: FloatField) -> None:
    with computation(FORWARD), interval(...):
        out_field = in_field


class OrchestratedCode:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ) -> None:
        orchestratable_methods = ["write_at_0", "write_at_top", "do_not_inline"]
        for method in orchestratable_methods:
            orchestrate(
                obj=self,
                config=stencil_factory.config.dace_config,
                method_to_orchestrate=method,
            )

        self.stencil_simple_2D_write = stencil_factory.from_dims_halo(
            func=stencil_simple_2D_write,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self.stencil_2D_write_at_K = stencil_factory.from_dims_halo(
            func=stencil_2D_write_at_K,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self.stencil_do_not_inline = stencil_factory.from_dims_halo(
            func=stencil_forward_at_K,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

    def write_at_0(
        self,
        in_field: FloatField,
        out_field: FloatFieldIJ,
    ) -> None:
        self.stencil_simple_2D_write(in_field, out_field)

    def write_at_top(
        self,
        in_field: FloatField,
        out_field: FloatFieldIJ,
    ) -> None:
        self.stencil_2D_write_at_K(in_field, out_field)

    def do_not_inline(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil_do_not_inline(in_field, out_field)


Factories: TypeAlias = tuple[StencilFactory, QuantityFactory]


class TestStree2DWriteInline:
    @pytest.fixture(params=["orch:dace:cpu:IJK", "orch:dace:cpu:KJI"])
    def factories(self, request) -> Factories:

        domain = (3, 3, 4)
        return get_factories_single_tile(
            domain[0], domain[1], domain[2], 0, backend=Backend(request.param)
        )

    @pytest.fixture
    def code(self, factories: Factories) -> OrchestratedCode:
        return OrchestratedCode(*factories)

    def test_common_2D_write(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM], "")
        in_qty.field[:, :, 0] = Float(32.0)

        with StreeOptimization():
            code.write_at_0(in_qty, out_qty)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        all_loop_region = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, LoopRegion)
        ]

        assert len(all_maps) == 2
        assert len(all_loop_region) == 0
        assert (out_qty.field[:] == Float(32.0)).all()

    def test_2D_write_K_top(self, code: OrchestratedCode, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM], "")
        in_qty.field[:, :, -1] = Float(32.0)

        with StreeOptimization():
            code.write_at_top(in_qty, out_qty)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        all_loop_region = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, LoopRegion)
        ]

        assert len(all_maps) == 2
        assert len(all_loop_region) == 0
        assert (out_qty.field[:] == Float(32.0)).all()

    def test_do_not_inline(self, code: OrchestratedCode, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            code.do_not_inline(in_qty, out_qty)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        all_loop_region = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, LoopRegion)
        ]

        assert len(all_maps) == 2
        assert len(all_loop_region) == 1
        assert (out_qty.field[:] == Float(1)).all()
