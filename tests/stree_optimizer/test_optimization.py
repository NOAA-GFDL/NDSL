from ndsl import StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.dace.stree.optimizations import AxisIterator, CartesianAxisMerge
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


def test_stree_roundtrip_no_opt() -> None:
    """Dev Note:

    The below code successfully merges top level K loop (2 loops)
    How do we test it?! Running doesn't test merging and the compilation
    is a near-black box. We could reach in the `dace_config.compiled_sdfg`
    cache but it's keyed on the dace.program and if we can reach the program
    well we can reach the SDFG and turn it into an stree for verification
    Should we run orchestration "by hand"?
    Can we intercept the `stree` ? After all we just want to check that!

    Test is deactivated for now"""

    return
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
    orch._INTERNAL__SCHEDULE_TREE_PASSES = [CartesianAxisMerge(AxisIterator._K)]

    code(in_qty, out_qty)

    assert (out_qty.field[:] == 4).all()

    orch._INTERNAL__SCHEDULE_TREE_OPTIMIZATION = False
