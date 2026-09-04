import dace
import pytest
from dace import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion

from ndsl import OptimizationConfig, QuantityFactory, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.config import Backend
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, K, computation, interval
from ndsl.dsl.optimization_config import OptimizationHint, OptimizationOption
from ndsl.dsl.typing import FloatField
from tests.dsl.dace.stree import get_SDFG_and_purge
from tests.dsl.dace.stree.optimizations import Factories


def stencil(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        out_field = in_field + 1


def stencil_with_self_assign(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        out_field = out_field + in_field + 2


def stencil_with_forward_K(in_field: FloatField, out_field: FloatField) -> None:
    with computation(FORWARD), interval(...):
        out_field = in_field + 3


def stencil_with_different_intervals(
    in_field: FloatField,
    out_field: FloatField,
) -> None:
    with computation(PARALLEL), interval(1, None):
        out_field = in_field + 5


def stencil_with_buffer_read_offset_in_Km1(
    in_field: FloatField, out_field: FloatField, buffer: FloatField
) -> None:
    with computation(PARALLEL), interval(1, None):
        buffer = in_field + 6

    with computation(PARALLEL), interval(1, None):
        out_field = buffer[K - 1] + 7


class OrchestratedCode:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        hint: OptimizationHint,
    ) -> None:
        config = OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=True,
                merger=OptimizationConfig.Tree.Merger(enabled=True),
                kernelize=OptimizationOption.DO_NOT_APPLY,
            ),
            hint=hint,
        )
        orchestratable_methods = [
            "trivial_merge",
            "missing_merge_of_forscope_and_map",
            "overcompute_merge",
            "push_non_cartesian_for",
            "block_merge_read_after_write_with_offset",
            "block_merge_write_after_read_with_offset",
            "block_merge_write_after_write_with_different_offset",
        ]
        for method in orchestratable_methods:
            orchestrate(
                obj=self,
                config=stencil_factory.config.dace_config,
                method_to_orchestrate=method,
                optimization_config=config,
            )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="no_overcompute_merge",
            optimization_config=OptimizationConfig(
                stree=OptimizationConfig.Tree(
                    enabled=True,
                    merger=OptimizationConfig.Tree.Merger(
                        enabled=True,
                        overcompute=False,
                    ),
                    kernelize=OptimizationOption.DO_NOT_APPLY,
                )
            ),
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="overcompute_merge_with_auto_kernalize",
            optimization_config=OptimizationConfig(
                stree=OptimizationConfig.Tree(
                    enabled=True,
                    merger=OptimizationConfig.Tree.Merger(
                        enabled=True,
                        overcompute=False,
                    ),
                    kernelize=OptimizationOption.AUTO,
                ),
                hint=hint,
            ),
        )
        self.hint = hint
        self.stencil = stencil_factory.from_dims_halo(
            func=stencil,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self.stencil_with_forward_K = stencil_factory.from_dims_halo(
            func=stencil_with_forward_K,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self.stencil_with_buffer_read_offset_in_Km1 = stencil_factory.from_dims_halo(
            func=stencil_with_buffer_read_offset_in_Km1,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self.stencil_with_different_intervals = stencil_factory.from_dims_halo(
            func=stencil_with_different_intervals,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._buffer = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], units="")

    def trivial_merge(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil(in_field, out_field)
        self.stencil(in_field, out_field)

    def missing_merge_of_forscope_and_map(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil(in_field, out_field)
        self.stencil_with_forward_K(in_field, out_field)
        self.stencil(in_field, out_field)

    def overcompute_merge(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil(in_field, out_field)
        self.stencil_with_different_intervals(in_field, out_field)

    def overcompute_merge_with_auto_kernalize(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil(in_field, out_field)
        self.stencil_with_different_intervals(in_field, out_field)

    def no_overcompute_merge(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil(in_field, out_field)
        self.stencil_with_different_intervals(in_field, out_field)

    def push_non_cartesian_for(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil(in_field, out_field)
        for _ in dace.nounroll(range(2)):
            self.stencil(in_field, out_field)

    def block_merge_read_after_write_with_offset(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil(in_field, out_field)
        self.stencil_with_buffer_read_offset_in_Km1(in_field, out_field, self._buffer)

    def block_merge_write_after_read_with_offset(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil_with_buffer_read_offset_in_Km1(in_field, out_field, self._buffer)
        self.stencil(in_field, out_field)

    def block_merge_write_after_write_with_different_offset(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil_with_buffer_read_offset_in_Km1(in_field, out_field, self._buffer)
        self.stencil(in_field, out_field)


class TestStreeMergeMaps:
    @pytest.fixture(
        params=[Backend("orch:dace:cpu:IJK"), Backend("orch:dace:cpu:KJI")],
        ids=["orch:dace:cpu:IJK", "orch:dace:cpu:KJI"],
    )
    def backend(self, request) -> Backend:
        return request.param

    @pytest.fixture
    def factories(self, backend: Backend) -> Factories:
        domain = (3, 3, 4)
        return get_factories_single_tile_orchestrated(
            domain[0], domain[1], domain[2], 0, backend=backend
        )

    @pytest.fixture(
        params=[OptimizationHint.SERIAL, OptimizationHint.PARALLEL],
        ids=["hint:Serial", "hint:Parallel"],
    )
    def hint(self, request):
        return request.param

    @pytest.fixture
    def code(self, hint: OptimizationHint, factories: Factories) -> OrchestratedCode:
        return OrchestratedCode(*factories, hint)

    def test_trivial_merge(self, code: OrchestratedCode, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        code.trivial_merge(in_qty, out_qty)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]

        assert len(all_maps) == 1  # all merged and collapsed
        assert (out_qty.field[:] == 2).all()

    def test_missing_merge_of_forscope_and_map(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        code.missing_merge_of_forscope_and_map(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            map_entry
            for map_entry, _ in sdfg.all_nodes_recursive()
            if isinstance(map_entry, nodes.MapEntry)
        ]
        all_loops = [
            loop
            for loop, _ in sdfg.all_nodes_recursive()
            if isinstance(loop, LoopRegion)
        ]

        if stencil_factory.backend == Backend("orch:dace:cpu:IJK"):
            assert len(all_maps) == 3  # 1 IJ + 2 Ks
            assert len(all_loops) == 1  # 1 For loop
        elif stencil_factory.backend == Backend("orch:dace:cpu:KJI"):
            assert len(all_maps) == 3  # 2 KJI (all maps) + 1 JI
            assert len(all_loops) == 1  # 1 For loop

    def test_overcompute_merge_with_auto_kernalize(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        code.overcompute_merge_with_auto_kernalize(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]

        if code.hint == OptimizationHint.PARALLEL:
            assert len(all_maps) == 2  # Re-kernalize to two maps post merge
        if code.hint == OptimizationHint.SERIAL:
            if stencil_factory.backend == Backend("orch:dace:cpu:IJK"):
                assert len(all_maps) == 3  # No merge between K and IJ
            elif stencil_factory.backend == Backend("orch:dace:cpu:KJI"):
                assert len(all_maps) == 2  # No merge between K and IJ

    def test_overcompute_merge(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        code.overcompute_merge(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]

        if self.hint == OptimizationHint.PARALLEL:
            assert len(all_maps) == 1  # All maps merged and collapsed
        elif self.hint == OptimizationHint.SERIAL:
            assert (
                len(all_maps) == 3
            )  # K merged - but IJ merge block by overcompute guard

    def test_no_overcompute_merge(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        code.no_overcompute_merge(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg

        all_maps = [
            me for me, _ in sdfg.all_nodes_recursive() if isinstance(me, nodes.MapEntry)
        ]
        k_maps = 0
        ij_maps = 0
        for map_entry in all_maps:
            if len(map_entry.map.params) == 1 and map_entry.map.params[0].startswith(
                "__k"
            ):
                k_maps += 1
            if map_entry.map.params == ["__i", "__j"]:
                ij_maps += 1

        if stencil_factory.backend == Backend("orch:dace:cpu:IJK"):
            assert ij_maps == 1
            assert k_maps == 2
        elif stencil_factory.backend == Backend("orch:dace:cpu:KJI"):
            assert len(all_maps) == 2  # 2 IJKs (un-merged since K is not merging)

    def test_block_merge_read_after_write_with_offset(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        # Forbid merging when data dependencies are detected
        code.block_merge_read_after_write_with_offset(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]

        if stencil_factory.backend == Backend("orch:dace:cpu:IJK"):
            assert len(all_maps) == 3  # 1 IJ + 2 Ks (un-merged)
        elif stencil_factory.backend == Backend("orch:dace:cpu:KJI"):
            if self.hint == OptimizationHint.PARALLEL:
                assert len(all_maps) == 2  # 2 IJKs (un-merged)
            elif self.hint == OptimizationHint.SERIAL:
                assert (
                    len(all_maps) == 4
                )  # 1 IJKs and 1 K + IJs merging block by overcompute if-guard

    def test_block_merge_write_after_read_with_offset(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        # Forbid merging when data dependencies are detected
        code.block_merge_write_after_read_with_offset(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]

        if stencil_factory.backend == Backend("orch:dace:cpu:IJK"):
            assert len(all_maps) == 3  # 1 IJ + 2 Ks (un-merged)
        elif stencil_factory.backend == Backend("orch:dace:cpu:KJI"):
            if self.hint == OptimizationHint.PARALLEL:
                assert len(all_maps) == 2  # 2 IJKs (un-merged)
            elif self.hint == OptimizationHint.SERIAL:
                assert (
                    len(all_maps) == 4
                )  # 1 IJKs and 1 K + IJs merging block by overcompute if-guard

    def test_block_merge_write_after_write_with_different_offset(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        # Forbid merging when data dependencies are detected
        code.block_merge_write_after_write_with_different_offset(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]

        if stencil_factory.backend == Backend("orch:dace:cpu:IJK"):
            assert len(all_maps) == 3  # 1 IJ + 2 Ks (un-merged)
        elif stencil_factory.backend == Backend("orch:dace:cpu:KJI"):
            if self.hint == OptimizationHint.PARALLEL:
                assert len(all_maps) == 2  # 2 IJKs (un-merged)
            elif self.hint == OptimizationHint.SERIAL:
                assert (
                    len(all_maps) == 4
                )  # 1 IJKs and 1 K + IJs merging block by overcompute if-guard

    def test_push_non_cartesian_for(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        # Push non-cartesian ForScope inwards, which allow to potentially
        # merge cartesian maps
        code.push_non_cartesian_for(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, nodes.MapEntry)
        ]
        for_loops = [
            node
            for node, _ in sdfg.all_nodes_recursive()
            if isinstance(node, LoopRegion) and tn.loop_variant(node) == "for"
        ]

        assert len(all_maps) == 1  # All merged & collapsed
        assert len(for_loops) == 1  # 1 For loop
