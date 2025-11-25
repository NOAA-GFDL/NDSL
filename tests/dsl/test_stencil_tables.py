import numpy as np
from gt4py.storage import ones, zeros

from ndsl import (
    CompilationConfig,
    DaceConfig,
    DaCeOrchestration,
    FrozenStencil,
    GridIndexing,
    StencilConfig,
    StencilFactory,
    orchestrate,
)
from ndsl.dsl.gt4py import FORWARD, PARALLEL, Field, GlobalTable, computation, interval
from ndsl.dsl.stencil import CompareToNumpyStencil
from tests.dsl import utils


def _stencil(inp: GlobalTable[np.int32, (5,)], out: Field[np.float64]) -> None:
    with computation(PARALLEL), interval(0, -1):
        out[0, 0, 0] = inp.A[1]
    with computation(FORWARD), interval(-1, None):
        out[0, 0, 0] = inp.A[1] + inp.A[2]


def _build_stencil(
    backend: str, orchestrated: DaCeOrchestration
) -> tuple[FrozenStencil | CompareToNumpyStencil, GridIndexing, StencilConfig]:
    # Make stencil and verify it ran
    grid_indexing = GridIndexing(
        domain=(5, 5, 5),
        n_halo=2,
        south_edge=True,
        north_edge=True,
        west_edge=True,
        east_edge=True,
    )

    stencil_config = StencilConfig(
        compilation_config=CompilationConfig(backend=backend, rebuild=True),
        dace_config=DaceConfig(None, backend, 5, 5, orchestrated),
    )

    stencil_factory = StencilFactory(stencil_config, grid_indexing)

    built_stencil = stencil_factory.from_origin_domain(
        _stencil, origin=(0, 0, 0), domain=grid_indexing.domain
    )

    return built_stencil, grid_indexing, stencil_config


class OrchestratedProgram:
    def __init__(self, backend, orchestration: DaCeOrchestration):
        self.stencil, grid_indexing, stencil_config = _build_stencil(
            backend, orchestration
        )
        orchestrate(obj=self, config=stencil_config.dace_config)

        self.inp = ones(shape=(5,), dtype=np.int32, backend=backend)
        self.inp[1] = 42
        self.out = utils.make_storage(zeros, grid_indexing, stencil_config, dtype=float)

    def __call__(self):
        self.stencil(self.inp, self.out)


def test_stecil_with_table_orchestrated() -> None:
    program = OrchestratedProgram(
        backend="dace:cpu", orchestration=DaCeOrchestration.BuildAndRun
    )

    # run the orchestrated stencil
    program()

    # validate output
    for k in range(4):
        assert (program.out[:, :, k] == 42).all()
    assert (program.out[:, :, 4] == 43).all()
