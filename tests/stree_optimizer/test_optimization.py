from types import TracebackType

import dace

import ndsl.dsl.dace.orchestration as orch
from ndsl import NDSLRuntime, Quantity, QuantityFactory, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, K, computation, interval
from ndsl.dsl.typing import FloatField


def _get_SDFG_and_purge(stencil_factory: StencilFactory) -> dace.CompiledSDFG:
    """Get the Precompiled SDFG from and flush it out from the cache in order
    for the function to be reused."""
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


def copy_stencil_with_if(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        if K > 3:
            out_field = in_field + 4


def copy_stencil_with_temp_and_offset_in_K(
    in_field: FloatField,
    out_field: FloatField,
) -> None:
    with computation(PARALLEL), interval(...):
        temporary_field = in_field + 5

    with computation(PARALLEL), interval(0, -1):
        out_field = temporary_field[K - 1] + 6


class TriviallyMergeableCode:
    def __init__(self, stencil_factory: StencilFactory) -> None:
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self.stencil = stencil_factory.from_dims_halo(
            func=copy_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, in_field: FloatField, out_field: FloatField) -> None:
        self.stencil(in_field, out_field)
        self.stencil(in_field, out_field)


class NonTrivialMergingCode:
    def __init__(self, stencil_factory: StencilFactory) -> None:
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self.copy_stencil = stencil_factory.from_dims_halo(
            func=copy_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.copy_stencil_with_forward_K = stencil_factory.from_dims_halo(
            func=copy_stencil_with_forward_K,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.copy_stencil_with_if = stencil_factory.from_dims_halo(
            func=copy_stencil_with_if,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.copy_stencil_with_temp_and_offset_in_K = stencil_factory.from_dims_halo(
            func=copy_stencil_with_temp_and_offset_in_K,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, in_field: FloatField, out_field: FloatField) -> None:
        self.copy_stencil(in_field, out_field)
        self.copy_stencil_with_if(in_field, out_field)
        self.copy_stencil_with_forward_K(in_field, out_field)
        # self.copy_stencil_with_temp_and_offset_in_K(in_field, out_field)


def test_stree_merge_maps() -> None:
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend="dace:cpu_kfirst"
    )

    trivial_code = TriviallyMergeableCode(stencil_factory)
    in_qty = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "")
    out_qty = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "")

    # with StreeOptimization():
    #     trivial_code(in_qty, out_qty)
    #     precompiled_sdfg = _get_SDFG_and_purge(stencil_factory)
    #     all_maps = [
    #         (me, state)
    #         for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
    #         if isinstance(me, dace.nodes.MapEntry)
    #     ]

    #     assert len(all_maps) == 3
    #     assert (out_qty.field[:] == 1).all()

    complex_code = NonTrivialMergingCode(stencil_factory)
    with StreeOptimization():
        complex_code(in_qty, out_qty)
        sdfg = _get_SDFG_and_purge(stencil_factory).sdfg
        all_maps = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.nodes.MapEntry)
        ]
        assert len(all_maps) == 3
        all_loop_guard_state = [
            (me, state)
            for me, state in sdfg.all_nodes_recursive()
            if isinstance(me, dace.SDFGState) and me.name.startswith("loop_guard")
        ]
        assert len(all_loop_guard_state) == 1


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
