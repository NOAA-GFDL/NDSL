from ndsl import StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField


def double_map(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(...):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = out_field + in_field * 3


class TriviallyMergeableCode:
    def __init__(self, stencil_factory: StencilFactory):
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self.stencil = stencil_factory.from_dims_halo(
            func=double_map,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, in_field: FloatField, out_field: FloatField):
        self.stencil(in_field, out_field)


def test_stree_roundtrip_no_opt():
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend="dace:cpu_kfirst"
    )

    code = TriviallyMergeableCode(stencil_factory)
    in_qty = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "")
    out_qty = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "")

    # Temporarily flip the internal switch
    import ndsl.dsl.dace.orchestration as orch

    orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = True
    orch._INTERNAL__SCHEDULE_TREE_PASSES = []

    code(in_qty, out_qty)

    assert (out_qty.field[:] == 4).all()

    orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = False
