import os
import shutil
import sys
from pathlib import Path

import pytest
from gt4py.cartesian import config as gt_config
from gt4py.storage import empty, ones

from ndsl import (
    CompilationConfig,
    DaceConfig,
    DaCeOrchestration,
    GridIndexing,
    StencilConfig,
    StencilFactory,
)
from ndsl.comm.mpi import MPI
from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.gt4py import PARALLEL, Field, computation, interval
from ndsl.dsl.stencil import CompareToNumpyStencil, FrozenStencil


def _make_storage(
    func,
    grid_indexing: GridIndexing,
    stencil_config: StencilConfig,
    *,
    dtype=float,
    aligned_index=(0, 0, 0),
):
    return func(
        backend=stencil_config.compilation_config.backend,
        shape=grid_indexing.domain,
        dtype=dtype,
        aligned_index=aligned_index,
    )


def _stencil(inp: Field[float], out: Field[float], scalar: float):
    with computation(PARALLEL), interval(...):
        out = inp


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
        _stencil, (0, 0, 0), domain=grid_indexing.domain
    )

    return built_stencil, grid_indexing, stencil_config


class OrchestratedProgram:
    def __init__(self, backend, orchestration: DaCeOrchestration):
        self.stencil, grid_indexing, stencil_config = _build_stencil(
            backend, orchestration
        )
        orchestrate(obj=self, config=stencil_config.dace_config)
        self.inp = _make_storage(ones, grid_indexing, stencil_config, dtype=float)
        self.out = _make_storage(empty, grid_indexing, stencil_config, dtype=float)

    def __call__(self):
        self.stencil(self.inp, self.out, self.inp[0, 0, 0])


@pytest.mark.parametrize("backend", [pytest.param("dace:cpu")])
@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1, reason="relocatibility checked with a one-rank setup"
)
def test_relocatability_orchestration(backend: str) -> None:
    original_root_directory: str = gt_config.cache_settings["root_path"]
    cwd = Path.cwd()

    # Compile on default
    p0 = OrchestratedProgram(backend, DaCeOrchestration.BuildAndRun)
    p0()
    expected_cache_dir = (
        cwd
        / ".gt_cache_FV3_A"
        / "dacecache"
        / "test_caches_OrchestratedProgram___call__"
    )
    assert expected_cache_dir.exists()

    # Compile in another directory

    custom_path = cwd / ".my_cache_path"
    gt_config.cache_settings["root_path"] = custom_path
    p1 = OrchestratedProgram(backend, DaCeOrchestration.BuildAndRun)
    p1()
    expected_cache_dir = (
        custom_path
        / ".gt_cache_FV3_A"
        / "dacecache"
        / "test_caches_OrchestratedProgram___call__"
    )
    assert expected_cache_dir.exists()

    # Check relocability by copying the second cache directory,
    # changing the path of gt_config.cache_settings and trying to Run on it
    relocated_path = cwd / ".my_relocated_cache_path"
    shutil.copytree(custom_path, relocated_path, dirs_exist_ok=True)
    gt_config.cache_settings["root_path"] = relocated_path
    p2 = OrchestratedProgram(backend, DaCeOrchestration.Run)
    p2()

    # Generate a file exists error to check for bad path
    bogus_path = "./nope/not_at_all/not_happening"
    gt_config.cache_settings["root_path"] = bogus_path
    with pytest.raises(RuntimeError):
        OrchestratedProgram(backend, DaCeOrchestration.Run)

    # Restore cache settings
    gt_config.cache_settings["root_path"] = original_root_directory


@pytest.mark.parametrize("backend", [pytest.param("dace:cpu")])
@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1, reason="relocatibility checked with a one-rank setup"
)
def test_relocatability(backend: str) -> None:
    # Restore original dir name
    gt_config.cache_settings["dir_name"] = os.environ.get(
        "GT_CACHE_DIR_NAME", f".gt_cache_{MPI.COMM_WORLD.Get_rank():06}"
    )

    cwd = Path.cwd()
    backend_sanitized = backend.replace(":", "")
    python_version = f"py{sys.version_info[0]}{sys.version_info[1]}"
    expected_cache_path = (
        cwd
        / ".gt_cache_000000"
        / f"{python_version}_1013"
        / f"{backend_sanitized}"
        / "test_caches"
        / "_stencil"
    )

    # Compile on default
    p0 = OrchestratedProgram(backend, DaCeOrchestration.Python)
    p0()
    assert expected_cache_path.exists()

    # Compile in another directory

    custom_path = cwd / ".my_cache_path"
    expected_cache_path = (
        custom_path
        / ".gt_cache_000000"
        / f"{python_version}_1013"
        / f"{backend_sanitized}"
        / "test_caches"
        / "_stencil"
    )
    gt_config.cache_settings["root_path"] = custom_path
    p1 = OrchestratedProgram(backend, DaCeOrchestration.Python)
    p1()
    assert expected_cache_path.exists()

    # Check relocability by copying the first cache directory,
    # changing the path of gt_config.cache_settings and trying to Run on it
    relocated_path = cwd / ".my_relocated_cache_path"
    shutil.copytree(cwd / ".gt_cache_000000", relocated_path, dirs_exist_ok=True)
    gt_config.cache_settings["root_path"] = relocated_path
    expected_cache_path = (
        relocated_path
        / ".gt_cache_000000"
        / f"{python_version}_1013"
        / f"{backend_sanitized}"
        / "test_caches"
        / "_stencil"
    )
    p2 = OrchestratedProgram(backend, DaCeOrchestration.Python)
    p2()
    assert expected_cache_path.exists()
