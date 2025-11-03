import dace

from ndsl import NDSLRuntime, Quantity, QuantityFactory, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField


def stencil_A(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        out_field = in_field


def stencil_B(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        out_field = out_field + in_field * 3


class TriviallyMergeableCode:
    def __init__(self, stencil_factory: StencilFactory) -> None:
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self.stencil_A = stencil_factory.from_dims_halo(
            func=stencil_A,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.stencil_B = stencil_factory.from_dims_halo(
            func=stencil_B,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, in_field: FloatField, out_field: FloatField) -> None:
        self.stencil_A(in_field, out_field)
        self.stencil_B(in_field, out_field)


def test_stree_merge_maps() -> None:
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend="dace:cpu"
    )

    code = TriviallyMergeableCode(stencil_factory)
    in_qty = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "")
    out_qty = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "")

    # Temporarily flip the internal switch
    import ndsl.dsl.dace.orchestration as orch

    orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = True

    code(in_qty, out_qty)

    assert len(stencil_factory.config.dace_config.loaded_precompiled_SDFG.values()) == 1
    sdfg = list(stencil_factory.config.dace_config.loaded_precompiled_SDFG.values())[0]
    all_maps = [
        (me, state)
        for me, state in sdfg.sdfg.all_nodes_recursive()
        if isinstance(me, dace.nodes.MapEntry)
    ]

    assert len(all_maps) == 3
    assert (out_qty.field[:] == 4).all()

    orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = False


class LocalRefineableCode(NDSLRuntime):
    def __init__(
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ) -> None:
        super().__init__(stencil_factory.config.dace_config)
        self.stencil_A = stencil_factory.from_dims_halo(
            func=stencil_A,
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

    # Temporarily flip the internal switch
    import ndsl.dsl.dace.orchestration as orch

    orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = True

    code(in_qty, out_qty)

    assert len(stencil_factory.config.dace_config.loaded_precompiled_SDFG.values()) == 1
    sdfg = list(stencil_factory.config.dace_config.loaded_precompiled_SDFG.values())[0]

    for array in sdfg.sdfg.arrays.values():
        if array.transient:
            assert array.shape == (1, 1, 1)

    orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = False
