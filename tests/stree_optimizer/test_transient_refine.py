from types import TracebackType

import dace

import ndsl.dsl.dace.orchestration as orch
from ndsl import NDSLRuntime, Quantity, QuantityFactory, StencilFactory, orchestrate
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval, K, J, Field, IJK
from ndsl.dsl.typing import FloatField, Float


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


DATADIM_SIZE = 8
DDIM_NAME = "DDIM"
DDIM_TYPE = Field[IJK, (Float, (DATADIM_SIZE))]


def copy_stencil(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        out_field = in_field + 1


def copy_stencil_with_K_offset(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(0, -1):
        out_field = in_field[K + 1]


def copy_stencil_with_J_offset(in_field: FloatField, out_field: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        out_field = in_field[J + 1]


def copy_stencil_with_ddim(in_field: DDIM_TYPE, out_field: DDIM_TYPE) -> None:
    with computation(PARALLEL), interval(...):
        n = 0
        while n < DATADIM_SIZE:
            out_field[0, 0, 0][n] = in_field[0, 0, 0][n] + 1
            n = n + 1


class TransientRefineableCode(NDSLRuntime):
    def __init__(
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ) -> None:
        super().__init__(stencil_factory.config.dace_config)
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
        self.copy = stencil_factory.from_dims_halo(
            func=copy_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.copy_with_K_offset = stencil_factory.from_dims_halo(
            func=copy_stencil_with_K_offset,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.copy_with_J_offset = stencil_factory.from_dims_halo(
            func=copy_stencil_with_J_offset,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.copy_stencil_with_ddim = stencil_factory.from_dims_halo(
            func=copy_stencil_with_ddim,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.tmp = self.make_local(quantity_factory, [X_DIM, Y_DIM, Z_DIM])
        self.tmp_ddim = self.make_local(
            quantity_factory, [X_DIM, Y_DIM, Z_DIM, DDIM_NAME]
        )

    def refine_to_scalar(self, in_field: Quantity, out_field: Quantity) -> None:
        self.copy(in_field, self.tmp)
        self.copy(self.tmp, out_field)

    def refine_to_K_buffer(self, in_field: Quantity, out_field: Quantity) -> None:
        self.copy(in_field, self.tmp)
        self.copy_with_K_offset(self.tmp, out_field)

    def refine_to_JK_buffer(self, in_field: Quantity, out_field: Quantity) -> None:
        self.copy(in_field, self.tmp)
        self.copy_with_J_offset(self.tmp, out_field)

    def do_not_refine_datadims(self, in_field: Quantity, out_field: Quantity) -> None:
        self.copy_stencil_with_ddim(in_field, self.tmp_ddim)
        self.copy_stencil_with_ddim(self.tmp_ddim, out_field)


def test_stree_roundtrip_transient_is_refined() -> None:
    domain = (3, 3, 4)
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        domain[0], domain[1], domain[2], 0, backend="dace:cpu_kfirst"
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
        precompiled_sdfg = _get_SDFG_and_purge(stencil_factory)
        for array in precompiled_sdfg.sdfg.arrays.values():
            if array.transient:
                assert array.shape == (1, 1, 1)

        # Refine cartesian axis to buffers
        #   IJ merges - K is a buffer
        code.refine_to_K_buffer(in_qty, out_qty)
        precompiled_sdfg = _get_SDFG_and_purge(stencil_factory)
        for array in precompiled_sdfg.sdfg.arrays.values():
            if array.transient:
                assert array.shape == (
                    1,
                    1,
                    domain[2] + 1,  # Quantity are domain size + 1
                )

        # I merges - JK buffer
        code.refine_to_JK_buffer(in_qty, out_qty)
        precompiled_sdfg = _get_SDFG_and_purge(stencil_factory)
        for array in precompiled_sdfg.sdfg.arrays.values():
            if array.transient:
                assert array.shape == (
                    1,
                    domain[1] + 1,  # Quantity are domain size + 1
                    domain[2] + 1,
                )

        # Refine to remaining data dimensions
        code.do_not_refine_datadims(in_qty_ddim, out_qty_ddim)
        precompiled_sdfg = _get_SDFG_and_purge(stencil_factory)
        for array in precompiled_sdfg.sdfg.arrays.values():
            if array.transient:
                assert array.shape == (1, 1, 1, DATADIM_SIZE) or len(array.shape) == 1
