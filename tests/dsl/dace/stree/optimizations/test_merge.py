from typing import TypeAlias

import dace
import pytest
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.config import Backend
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, K, computation, interval
from ndsl.dsl.typing import FloatField
from tests.dsl.dace.stree import StreeOptimization, get_SDFG_and_purge


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


def stencil_with_buffer_read_offset_in_K(
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
    ) -> None:
        orchestratable_methods = [
            "trivial_merge",
            "missing_merge_of_forscope_and_map",
            "overcompute_merge",
            "block_merge_when_dependencies_are_found",
            "push_non_cartesian_for",
        ]
        for method in orchestratable_methods:
            orchestrate(
                obj=self,
                config=stencil_factory.config.dace_config,
                method_to_orchestrate=method,
            )

        self.stencil = stencil_factory.from_dims_halo(
            func=stencil,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self.stencil_with_forward_K = stencil_factory.from_dims_halo(
            func=stencil_with_forward_K,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self.stencil_with_buffer_read_offset_in_K = stencil_factory.from_dims_halo(
            func=stencil_with_buffer_read_offset_in_K,
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

    def block_merge_when_dependencies_are_found(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.stencil(in_field, out_field)
        self.stencil_with_buffer_read_offset_in_K(in_field, out_field, self._buffer)

    def overcompute_merge(
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


Factories: TypeAlias = tuple[StencilFactory, QuantityFactory]


class TestStreeMergeMapsIJK:
    @pytest.fixture
    def factories(self) -> Factories:
        domain = (3, 3, 4)
        return get_factories_single_tile_orchestrated(
            domain[0], domain[1], domain[2], 0, backend=Backend("orch:dace:cpu:IJK")
        )

    @pytest.fixture
    def code(self, factories: Factories) -> OrchestratedCode:
        return OrchestratedCode(*factories)

    def test_trivial_merge(self, code: OrchestratedCode, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            code.trivial_merge(in_qty, out_qty)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]

        assert len(all_maps) == 3
        assert (out_qty.field[:] == 2).all()

    def test_missing_merge_of_forscope_and_map(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            code.missing_merge_of_forscope_and_map(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            map_entry
            for map_entry, _ in sdfg.all_nodes_recursive()
            if isinstance(map_entry, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 4  # 2 IJ + 2 Ks
        all_loops = [
            loop
            for loop, _ in sdfg.all_nodes_recursive()
            if isinstance(loop, dace.sdfg.state.LoopRegion)
        ]
        assert len(all_loops) == 1  # 1 For loop

    def test_overcompute_merge(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            code.overcompute_merge(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        # ⚠️ WE EXPECT A FAILURE TO MERGE K (because of index) ⚠️
        assert len(all_maps) == 4  # Should be all merged = 3

    def test_block_merge_when_dependencies_are_found(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            # Forbid merging when data dependencies are detected
            code.block_merge_when_dependencies_are_found(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        # ⚠️ WE EXPECT A FAILURE TO MERGE K (because of index) ⚠️
        assert len(all_maps) == 5  # Should be 4 = 2 IJ + 2 Ks (un-merged)

    def test_push_non_cartesian_for(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            # Push non-cartesian ForScope inwards, which allow to potentially
            # merge cartesian maps
            code.push_non_cartesian_for(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 3  # All merged
        for_loops = [
            node
            for node, _ in sdfg.all_nodes_recursive()
            if isinstance(node, LoopRegion) and tn.loop_variant(node) == "for"
        ]
        assert len(for_loops) == 1  # 1 For loop


class TestStreeMergeMapsKJI:
    @pytest.fixture
    def factories(self) -> Factories:
        domain = (3, 3, 4)
        return get_factories_single_tile_orchestrated(
            domain[0], domain[1], domain[2], 0, backend=Backend("orch:dace:cpu:KJI")
        )

    @pytest.fixture
    def code(self, factories: Factories) -> OrchestratedCode:
        return OrchestratedCode(*factories)

    def test_trivial_merge(self, code: OrchestratedCode, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            code.trivial_merge(in_qty, out_qty)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]

        assert len(all_maps) == 3
        assert (out_qty.field[:] == 2).all()

    def test_missing_merge_of_forscope_and_map(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        code.missing_merge_of_forscope_and_map(in_qty, out_qty)
        sdfg = get_SDFG_and_purge(stencil_factory).sdfg

        with StreeOptimization():
            # K iterative loop - blocks all merges
            code.missing_merge_of_forscope_and_map(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            map_entry
            for map_entry, _ in sdfg.all_nodes_recursive()
            if isinstance(map_entry, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 8  # 2 KJI (all maps) + 1 for scope
        all_loops = [
            loop
            for loop, _ in sdfg.all_nodes_recursive()
            if isinstance(loop, dace.sdfg.state.LoopRegion)
        ]
        assert len(all_loops) == 1  # 1 For loop

    def test_overcompute_merge(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            # Overcompute merge in K - we merge and introduce an If guard
            code.overcompute_merge(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        # ⚠️ WE EXPECT A FAILURE TO MERGE K (because of index) ⚠️
        assert len(all_maps) == 6

    def test_block_merge_when_dependencies_are_found(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            # Forbid merging when data dependencies are detected
            code.block_merge_when_dependencies_are_found(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        # ⚠️ WE EXPECT A FAILURE TO MERGE K (because of index) ⚠️
        assert len(all_maps) == 9

    def test_push_non_cartesian_for(
        self, code: OrchestratedCode, factories: Factories
    ) -> None:
        stencil_factory, quantity_factory = factories
        in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            # Push non-cartesian ForScope inwards, which allow to potentially
            # merge cartesian maps
            code.push_non_cartesian_for(in_qty, out_qty)

        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 3  # All merged
        for_loops = [
            node
            for node, _ in sdfg.all_nodes_recursive()
            if isinstance(node, LoopRegion) and tn.loop_variant(node) == "for"
        ]
        assert len(for_loops) == 1  # 1 For loop
