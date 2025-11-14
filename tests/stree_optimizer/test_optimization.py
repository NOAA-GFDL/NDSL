from types import TracebackType

import dace

import ndsl.dsl.dace.orchestration as orch
from ndsl import NDSLRuntime, Quantity, QuantityFactory, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, K, computation, interval
from ndsl.dsl.typing import FloatField


def _get_SDFG_and_purge(stencil_factory: StencilFactory) -> dace.CompiledSDFG:
    """Get the Precompiled SDFG from the dace config dict where they are cached post
    compilation and flush the cache in order for next build to re-use the function."""
    sdfg_repo = stencil_factory.config.dace_config.loaded_precompiled_SDFG

    if len(sdfg_repo.values()) != 1:
        raise RuntimeError("Failure to compile SDFG")
    sdfg = list(sdfg_repo.values())[0]

    sdfg_repo.clear()

    return sdfg


class StreeOptimization:
    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = False


def copy_stencil(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        out_field = in_field + 1


def copy_stencil_with_self_assign(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        out_field = out_field + in_field + 2


def copy_stencil_with_forward_K(in_field: FloatField, out_field: FloatField) -> None:
    with computation(FORWARD), interval(...):
        out_field = in_field + 3


def copy_stencil_with_different_intervals(
    in_field: FloatField,
    out_field: FloatField,
) -> None:
    with computation(PARALLEL), interval(1, None):
        out_field = in_field + 5


def copy_stencil_with_buffer_read_offset_in_K(
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
        ]
        for method in orchestratable_methods:
            orchestrate(
                obj=self,
                config=stencil_factory.config.dace_config,
                method_to_orchestrate=method,
            )

        self.copy_stencil = stencil_factory.from_dims_halo(
            func=copy_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.copy_stencil_with_forward_K = stencil_factory.from_dims_halo(
            func=copy_stencil_with_forward_K,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.copy_stencil_with_buffer_read_offset_in_K = stencil_factory.from_dims_halo(
            func=copy_stencil_with_buffer_read_offset_in_K,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.copy_stencil_with_different_intervals = stencil_factory.from_dims_halo(
            func=copy_stencil_with_different_intervals,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self._buffer = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="")

    def trivial_merge(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.copy_stencil(in_field, out_field)
        self.copy_stencil(in_field, out_field)

    def missing_merge_of_forscope_and_map(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.copy_stencil(in_field, out_field)
        self.copy_stencil_with_forward_K(in_field, out_field)
        self.copy_stencil(in_field, out_field)

    def block_merge_when_depandencies_is_found(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.copy_stencil(in_field, out_field)
        self.copy_stencil_with_buffer_read_offset_in_K(
            in_field, out_field, self._buffer
        )

    def overcompute_merge(
        self,
        in_field: FloatField,
        out_field: FloatField,
    ) -> None:
        self.copy_stencil(in_field, out_field)
        self.copy_stencil_with_different_intervals(in_field, out_field)


def test_stree_merge_maps() -> None:
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend="dace:cpu_kfirst"
    )

    code = OrchestratedCode(stencil_factory, quantity_factory)
    in_qty = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "")
    out_qty = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "")

    with StreeOptimization():
        # Trivial merge
        code.trivial_merge(in_qty, out_qty)
        precompiled_sdfg = _get_SDFG_and_purge(stencil_factory)
        all_maps = [
            (me, state)
            for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]

        assert len(all_maps) == 3
        assert (out_qty.field[:] == 2).all()

        # Merge IJ - but do not merge K map & for (missing feature)
        code.missing_merge_of_forscope_and_map(in_qty, out_qty)
        sdfg = _get_SDFG_and_purge(stencil_factory).sdfg
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
        sdfg = _get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 3

        # Forbid merging when data dependancy is detected
        code.block_merge_when_depandencies_is_found(in_qty, out_qty)
        sdfg = _get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 4  # 2 IJ + 2 Ks (un-merged)


class LocalRefineableCode(NDSLRuntime):
    def __init__(
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ) -> None:
        super().__init__(stencil_factory.config.dace_config)
        self.stencil_A = stencil_factory.from_dims_halo(
            func=copy_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.tmp = self.make_local(quantity_factory, [X_DIM, Y_DIM, Z_DIM])

    def __call__(self, in_field: Quantity, out_field: Quantity) -> None:
        self.stencil_A(in_field, self.tmp)
        self.stencil_A(self.tmp, out_field)


def test_stree_roundtrip_transient_is_refined() -> None:
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend="dace:cpu"
    )

    code = LocalRefineableCode(stencil_factory, quantity_factory)
    in_qty = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "")
    out_qty = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "")

    with StreeOptimization():
        code(in_qty, out_qty)

        precompiled_sdfg = _get_SDFG_and_purge(stencil_factory)

        for array in precompiled_sdfg.sdfg.arrays.values():
            if array.transient:
                assert array.shape == (1, 1, 1)
