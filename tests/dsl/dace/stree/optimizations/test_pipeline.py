from dace.sdfg.state import LoopRegion
from gt4py.cartesian.gtscript import FORWARD

from ndsl import OptimizationConfig, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.config import Backend
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField
from tests.dsl.dace.stree import get_SDFG_and_purge


def double_map(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(...):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = out_field + in_field * 3


class TriviallyMergeableCode:
    def __init__(self, stencil_factory: StencilFactory):
        config = OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=True, merger=OptimizationConfig.Tree.Merger(enabled=True)
            )
        )
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            optimization_config=config,
        )
        self.stencil = stencil_factory.from_dims_halo(
            func=double_map,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

    def __call__(self, in_field: FloatField, out_field: FloatField):
        self.stencil(in_field, out_field)


def test_stree_roundtrip():
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend=Backend.cpu()
    )

    code = TriviallyMergeableCode(stencil_factory)
    in_qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
    out_qty = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

    code(in_qty, out_qty)

    assert (out_qty.field[:] == 4).all()

    get_SDFG_and_purge(stencil_factory)


def single_K_map(field: FloatField):
    with computation(FORWARD), interval(0, 1):
        field = 2.0


class LocalOptimizationsCode_Child:
    def __init__(self, stencil_factory: StencilFactory):
        config = OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=True,
                merger=OptimizationConfig.Tree.Merger(enabled=False),
                kernelize=False,
                inline_K_loops_size_one=True,
            )
        )
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            optimization_config=config,
        )
        self.stencil = stencil_factory.from_dims_halo(
            func=single_K_map,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

    def __call__(self, field):
        self.stencil(field)


class LocalOptimizationsCode_ChildNoOpt:
    def __init__(self, stencil_factory: StencilFactory):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )
        self.stencil = stencil_factory.from_dims_halo(
            func=single_K_map,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

    def __call__(self, field):
        self.stencil(field)


class LocalOptimizationsCode_TopLevel:
    def __init__(self, stencil_factory: StencilFactory):
        config = OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=True,
                merger=OptimizationConfig.Tree.Merger(enabled=True),
                kernelize=False,
                inline_K_loops_size_one=False,
            )
        )
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            optimization_config=config,
        )
        self.stencil = stencil_factory.from_dims_halo(
            func=single_K_map,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self.child = LocalOptimizationsCode_Child(stencil_factory)
        self.no_opt_child = LocalOptimizationsCode_ChildNoOpt(stencil_factory)

    def __call__(self, field: FloatField):
        self.stencil(field)
        self.child(field)
        self.no_opt_child(field)


def test_stree_local_optimization():
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend=Backend.cpu()
    )

    code = LocalOptimizationsCode_TopLevel(stencil_factory)
    qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")

    code(qty)

    precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

    all_loop_region = [
        (me, state)
        for me, state in precompiled_sdfg.sdfg.all_nodes_recursive()
        if isinstance(me, LoopRegion)
    ]

    # The above code has three K loops of over a single unit element
    # Child level optimization `LocalOptimizationsCode_Child` does `inline_K_loops_size_one` but
    # the top level doesn't leading to a single loop left
    assert len(all_loop_region) == 2
