from ndsl import NDSLRuntime, Quantity, QuantityFactory, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.config import Backend
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import IJK, PARALLEL, Field, J, K, computation, interval
from ndsl.dsl.typing import Float, FloatField

from .sdfg_stree_tools import StreeOptimization, get_SDFG_and_purge


DATADIM_SIZE = 8
DDIM_NAME = "DDIM"
DDIM_TYPE = Field[IJK, (Float, (DATADIM_SIZE))]


def stencil(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        out_field = in_field + 1


def stencil_with_K_offset(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(0, -1):
        out_field = in_field[K + 1] + 2


def stencil_with_JK_offset(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        out_field = in_field[J + 1, K + 1] + 3


def stencil_with_ddim(in_field: DDIM_TYPE, out_field: DDIM_TYPE) -> None:
    with computation(PARALLEL), interval(...):
        n = 0
        while n < DATADIM_SIZE:
            out_field[0, 0, 0][n] = in_field[0, 0, 0][n] + 4
            n = n + 1


class TransientRefineableCode(NDSLRuntime):
    def __init__(
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ) -> None:
        super().__init__(stencil_factory)
        orchestratable_methods = [
            "refine_to_scalar",
            "refine_to_K_buffer",
            "refine_to_JK_buffer",
            "do_not_refine_datadims",
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
        self.stencil_with_K_offset = stencil_factory.from_dims_halo(
            func=stencil_with_K_offset,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.stencil_with_JK_offset = stencil_factory.from_dims_halo(
            func=stencil_with_JK_offset,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.stencil_with_ddim = stencil_factory.from_dims_halo(
            func=stencil_with_ddim,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.tmp = self.make_local(quantity_factory, [X_DIM, Y_DIM, Z_DIM])
        self.tmp_ddim = self.make_local(
            quantity_factory, [X_DIM, Y_DIM, Z_DIM, DDIM_NAME]
        )

    def refine_to_scalar(self, in_field: Quantity, out_field: Quantity) -> None:
        self.stencil(in_field, self.tmp)
        self.stencil(self.tmp, out_field)

    def refine_to_K_buffer(self, in_field: Quantity, out_field: Quantity) -> None:
        self.stencil(in_field, self.tmp)
        self.stencil_with_K_offset(self.tmp, out_field)

    def refine_to_JK_buffer(self, in_field: Quantity, out_field: Quantity) -> None:
        self.stencil(in_field, self.tmp)
        self.stencil_with_JK_offset(self.tmp, out_field)

    def do_not_refine_datadims(self, in_field: Quantity, out_field: Quantity) -> None:
        self.stencil_with_ddim(in_field, self.tmp_ddim)
        self.stencil_with_ddim(self.tmp_ddim, out_field)


def test_stree_roundtrip_transient_is_refined() -> None:
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend=Backend.performance_cpu()
    )

    in_qty = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM], "")
    out_qty = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "")

    quantity_factory.add_data_dimensions({DDIM_NAME: DATADIM_SIZE})
    in_qty_ddim = quantity_factory.ones([X_DIM, Y_DIM, Z_DIM, DDIM_NAME], "")
    out_qty_ddim = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM, DDIM_NAME], "")

    code = TransientRefineableCode(stencil_factory, quantity_factory)

    with StreeOptimization():
        # Refine to scalar
        code.refine_to_scalar(in_qty, out_qty)
        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        for array in precompiled_sdfg.sdfg.arrays.values():
            if array.transient:
                assert array.shape == (1, 1, 1)

        # Refine cartesian axis to buffers
        #   IJ merges - K is a buffer
        code.refine_to_K_buffer(in_qty, out_qty)
        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        for array in precompiled_sdfg.sdfg.arrays.values():
            if array.transient:
                assert array.shape == (
                    1,
                    1,
                    domain[2] + 1,  # Quantity are domain size + 1
                )

        # I merges - JK buffer
        code.refine_to_JK_buffer(in_qty, out_qty)
        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        for array in precompiled_sdfg.sdfg.arrays.values():
            if array.transient:
                assert array.shape == (
                    1,
                    domain[1] + 1,  # Quantity are domain size + 1
                    domain[2] + 1,
                )

        # Refine to remaining data dimensions
        code.do_not_refine_datadims(in_qty_ddim, out_qty_ddim)
        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        for array in precompiled_sdfg.sdfg.arrays.values():
            if array.transient:
                assert array.shape == (1, 1, 1, DATADIM_SIZE) or len(array.shape) == 1
