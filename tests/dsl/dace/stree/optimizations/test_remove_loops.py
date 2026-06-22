import pytest
from dace import nodes
from dace.sdfg.state import LoopRegion

from ndsl import OptimizationConfig, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import Backend, BackendLoopOrder
from ndsl.constants import I_DIM, J_DIM, K_DIM, Float
from ndsl.dsl.gt4py import FORWARD, computation, interval
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from ndsl.stencils import copy
from tests.dsl.dace.stree import StreePipeline, get_SDFG_and_purge
from tests.dsl.dace.stree.optimizations import Factories


def stencil_simple_2D_write(in_field: FloatField, out_fieldIJ: FloatFieldIJ) -> None:
    with computation(FORWARD), interval(0, 1):
        out_fieldIJ = in_field


def stencil_multiple_2D_write(
    in_field: FloatField, out_fieldIJ: FloatFieldIJ, out_fieldIJ_2: FloatFieldIJ
) -> None:
    with computation(FORWARD), interval(0, 1):
        out_fieldIJ = in_field
        out_fieldIJ_2 = in_field + 1.0


def stencil_2D_write_at_K(in_field: FloatField, out_fieldIJ: FloatFieldIJ) -> None:
    with computation(FORWARD), interval(-1, None):
        out_fieldIJ = in_field


def stencil_forward_at_K(in_field: FloatField, out_field: FloatField) -> None:
    with computation(FORWARD), interval(...):
        out_field = in_field


class OrchestratedCode:
    def __init__(self, stencil_factory: StencilFactory) -> None:
        config = OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=True,
                inline_K_loops_size_one=True,
                merger=OptimizationConfig.Tree.Merger(enabled=True),
            )
        )
        methods_to_orchestrate = [
            "write_at_0",
            "write_at_top",
            "do_not_inline",
            "combined_stencils",
            "multiple_statements",
        ]
        for method in methods_to_orchestrate:
            orchestrate(
                obj=self,
                config=stencil_factory.config.dace_config,
                method_to_orchestrate=method,
                optimization_config=config,
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
        self.stencil_copy = stencil_factory.from_dims_halo(
            func=copy,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self.stencil_multiple_2D_write = stencil_factory.from_dims_halo(
            func=stencil_multiple_2D_write,
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

    def combined_stencils(
        self, field: FloatField, field2: FloatField, fieldIJ: FloatFieldIJ
    ) -> None:
        self.stencil_copy(field, field2)
        self.stencil_simple_2D_write(field2, fieldIJ)

    def multiple_statements(
        self, in_field: FloatField, out_field: FloatFieldIJ, out_field2: FloatFieldIJ
    ) -> None:
        self.stencil_copy(in_field, in_field)
        self.stencil_multiple_2D_write(in_field, out_field, out_field2)


class TestStree2DWriteInline:
    @pytest.fixture(params=["orch:dace:cpu:IJK", "orch:dace:cpu:KJI"])
    def factories(self, request: pytest.FixtureRequest) -> Factories:

        domain = (3, 3, 4)
        return get_factories_single_tile(
            domain[0], domain[1], domain[2], 0, backend=Backend(request.param)
        )

    def test_common_2D_write(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM], "")
        in_qty.field[:, :, 0] = Float(32.0)

        with StreePipeline():
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

        assert len(all_maps) == 1  # IJ/JI collapsed
        assert len(all_loop_region) == 0
        assert (out_qty.field[:] == Float(32.0)).all()

    def test_2D_write_K_top(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM], "")
        in_qty.field[:, :, -1] = Float(32.0)

        with StreePipeline():
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

        assert len(all_maps) == 1  # IJ/JI collapsed
        assert len(all_loop_region) == 0
        assert (out_qty.field[:] == Float(32.0)).all()

    def test_do_not_inline(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreePipeline():
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

        assert len(all_maps) == 1  # IJ/JI collapsed
        assert len(all_loop_region) == 1
        assert (out_qty.field[:] == Float(1)).all()

    def test_combined_stencils(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        field = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        field_2 = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")
        field_IJ = quantity_factory.zeros([I_DIM, J_DIM], "")

        with StreePipeline():
            code.combined_stencils(field, field_2, field_IJ)

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

        assert (
            len(all_maps) == 2  # IJ + K
            if stencil_factory.backend.loop_order == BackendLoopOrder.IJK
            else 2  # KJI + JI
        )
        assert len(all_loop_region) == 0
        assert (field_IJ.field[:] == Float(1)).all()

    def test_multiple_statements(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        field = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        field_IJ = quantity_factory.zeros([I_DIM, J_DIM], "")
        field_IJ_2 = quantity_factory.zeros([I_DIM, J_DIM], "")

        field.field[:, :, 0] = Float(42.0)
        with StreePipeline():
            code.multiple_statements(field, field_IJ, field_IJ_2)

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

        assert (
            len(all_maps) == 2  # IJ + K
            if stencil_factory.backend.loop_order == BackendLoopOrder.IJK
            else 2  # KJI + JI
        )
        assert len(all_loop_region) == 0
        assert (field_IJ.field[:] == Float(42.0)).all()
        assert (field_IJ_2.field[:] == Float(43.0)).all()
