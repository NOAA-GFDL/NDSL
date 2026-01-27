import dace

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.config import Backend
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, K, computation, interval
from ndsl.dsl.typing import FloatField

from .sdfg_stree_tools import StreeOptimization, get_SDFG_and_purge


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
            "block_merge_when_depandencies_is_found",
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
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.stencil_with_forward_K = stencil_factory.from_dims_halo(
            func=stencil_with_forward_K,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.stencil_with_buffer_read_offset_in_K = stencil_factory.from_dims_halo(
            func=stencil_with_buffer_read_offset_in_K,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.stencil_with_different_intervals = stencil_factory.from_dims_halo(
            func=stencil_with_different_intervals,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self._buffer = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="")

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

    def block_merge_when_depandencies_is_found(
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


def test_stree_merge_maps_IJK() -> None:
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend=Backend.performance_cpu()
    )

    code = OrchestratedCode(stencil_factory, quantity_factory)
    in_qty = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "")
    out_qty = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "")

    with StreeOptimization():
        # Trivial merge
        code.trivial_merge(in_qty, out_qty)
        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]

        assert len(all_maps) == 3
        assert (out_qty.field[:] == 2).all()

        # Merge IJ - but do not merge K map & for (missing feature)
        code.missing_merge_of_forscope_and_map(in_qty, out_qty)
        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 4  # 2 IJ + 2 Ks
        all_loop_guard_state = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.SDFGState) and me.name.startswith("loop_guard")
        ]
        assert len(all_loop_guard_state) == 1  # 1 For loop

        # Overcompute merge in K - we merge and introduce an If guard
        code.overcompute_merge(in_qty, out_qty)
        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        # ⚠️ WE EXPECT A FAILURE TO MERGE K (because of index) ⚠️
        assert len(all_maps) == 4  # Should be all dmerged = 3

        # Forbid merging when data dependancy is detected
        code.block_merge_when_depandencies_is_found(in_qty, out_qty)
        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            me.params[0]
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        # ⚠️ WE EXPECT A FAILURE TO MERGE K (because of index) ⚠️
        assert len(all_maps) == 5  # Should be 4 = 2 IJ + 2 Ks (un-merged)

        # Push non-cartesian ForScope inwward, which allow to potentially
        # merge cartesian maps
        code.push_non_cartesian_for(in_qty, out_qty)
        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 3  # All merged
        all_loop_guard_state = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.SDFGState) and me.name.startswith("loop_guard")
        ]
        assert len(all_loop_guard_state) == 1  # 1 For loop


def test_stree_merge_maps_KJI() -> None:
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, Backend("st:dace:cpu:KJI")
    )

    code = OrchestratedCode(stencil_factory, quantity_factory)
    in_qty = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "")
    out_qty = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "")

    with StreeOptimization():
        # Trivial merge
        code.trivial_merge(in_qty, out_qty)
        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]

        assert len(all_maps) == 3
        assert (out_qty.field[:] == 2).all()

        # K iterative loop - blocks all merges
        code.missing_merge_of_forscope_and_map(in_qty, out_qty)
        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 8  # 2 KJI (all maps) + 1 for scope
        all_loop_guard_state = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.SDFGState) and me.name.startswith("loop_guard")
        ]
        assert len(all_loop_guard_state) == 1  # 1 For loop

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

        # Forbid merging when data dependancy is detected
        code.block_merge_when_depandencies_is_found(in_qty, out_qty)
        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        # ⚠️ WE EXPECT A FAILURE TO MERGE K (because of index) ⚠️
        assert len(all_maps) == 9

        # Push non-cartesian ForScope inwward, which allow to potentially
        # merge cartesian maps
        code.push_non_cartesian_for(in_qty, out_qty)
        sdfg = get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 3  # All merged
        all_loop_guard_state = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.SDFGState) and me.name.startswith("loop_guard")
        ]
        assert len(all_loop_guard_state) == 1  # 1 For loop
