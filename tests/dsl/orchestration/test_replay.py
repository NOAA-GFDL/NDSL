import os

from ndsl import Backend, NDSLRuntime, StencilFactory
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.dace.dace_executable import DACE_EXECUTABLE_CACHE, DaceExecutable
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField


def stencil_42(qty: FloatField):
    with computation(PARALLEL), interval(...):
        qty = 42.42


class OrchestratedProgram(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
        self.stencil = stencil_factory.from_dims_halo(stencil_42, [I_DIM, J_DIM, K_DIM])

    def __call__(self, out_qty):  # no typehint out_qty on purpose
        self.stencil(out_qty)


def test_record_and_replay():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        5, 5, 2, 0, backend=Backend.cpu()
    )

    qty = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
    code = OrchestratedProgram(stencil_factory)

    record_orginal_value = os.getenv("NDSL_RECORD_ORCHESTRATION", "False")
    os.environ["NDSL_RECORD_ORCHESTRATION"] = "True"

    code(qty)

    assert len(DACE_EXECUTABLE_CACHE.values()) == 1
    exe = next(iter(DACE_EXECUTABLE_CACHE.values()))

    loaded_exe = DaceExecutable.from_serialized_bundle(
        exe.compiled_sdfg.sdfg.build_folder
    )

    loaded_exe.replay()

    os.environ["NDSL_RECORD_ORCHESTRATION"] = record_orginal_value
