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
from ndsl.config import Backend
from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.gt4py import PARALLEL, Field, computation, interval
from ndsl.dsl.stencil import CompareToNumpyStencil, FrozenStencil
from tests.dsl import utils


def _stencil(inp: Field[float], out: Field[float]) -> None:
    with computation(PARALLEL), interval(...):
        out = inp


def _build_stencil(
    backend: Backend, orchestrated: DaCeOrchestration | None
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
    def __init__(self, backend: Backend, orchestration: DaCeOrchestration | None):
        self.stencil, grid_indexing, stencil_config = _build_stencil(
            backend, orchestration
        )
        orchestrate(obj=self, config=stencil_config.dace_config)
        self.inp = utils.make_storage(ones, grid_indexing, stencil_config, dtype=float)
        self.out = utils.make_storage(empty, grid_indexing, stencil_config, dtype=float)

    def __call__(self) -> None:
        self.stencil(self.inp, self.out)


def test_relocatability_orchestration() -> None:
    # Compile on default
    p0 = OrchestratedProgram(Backend.cpu(), DaCeOrchestration.BuildAndRun)
    p0()

    expected_cache_dir = (
        Path.cwd()
        / ".gt_cache_FV3_A"
        / "dacecache"
        / "tests_dsl_test_caches_OrchestratedProgram___call__"
    )
    assert expected_cache_dir.exists()


def test_relocatability_orchestration_tmpdir(tmp_path: Path) -> None:
    gt_config.cache_settings["root_path"] = tmp_path

    # Compile in temporary directory that is only available in this test session.
    p1 = OrchestratedProgram(Backend.cpu(), DaCeOrchestration.BuildAndRun)
    p1()

    expected_cache_dir = (
        tmp_path
        / ".gt_cache_FV3_A"
        / "dacecache"
        / "tests_dsl_test_caches_OrchestratedProgram___call__"
    )
    assert expected_cache_dir.exists()

    # Check relocatability by copying the second cache directory,
    # changing the path of gt_config.cache_settings and trying to Run on it
    relocated_path = tmp_path / ".my_relocated_cache_path"
    shutil.copytree(tmp_path, relocated_path, dirs_exist_ok=False)
    gt_config.cache_settings["root_path"] = relocated_path
    p2 = OrchestratedProgram(Backend.cpu(), DaCeOrchestration.Run)
    p2()

    # Generate a file exists error to check for bad path
    bogus_path = "./nope/not_at_all/not_happening"
    gt_config.cache_settings["root_path"] = bogus_path
    with pytest.raises(RuntimeError):
        OrchestratedProgram(Backend.cpu(), DaCeOrchestration.Run)


def test_relocatability() -> None:
    gt_config.cache_settings["dir_name"] = os.environ.get(
        "GT_CACHE_DIR_NAME", f".gt_cache_{MPI.COMM_WORLD.Get_rank():06}"
    )
    gt_config.cache_settings["root_path"] = Path.cwd()

    # Compile on default
    backend = Backend("st:dace:cpu:KIJ")
    p0 = OrchestratedProgram(backend, None)
    p0()

    backend_sanitized = backend.as_gt4py().replace(":", "")
    python_version = f"py{sys.version_info[0]}{sys.version_info[1]}"
    expected_cache_path = (
        Path.cwd()
        / ".gt_cache_000000"
        / f"{python_version}_1013"
        / f"{backend_sanitized}"
        / "tests"
        / "dsl"
        / "test_caches"
        / "_stencil"
    )
    assert expected_cache_path.exists()


def test_relocatability_tmpdir(tmp_path: Path) -> None:
    gt_config.cache_settings["dir_name"] = os.environ.get(
        "GT_CACHE_DIR_NAME", f".gt_cache_{MPI.COMM_WORLD.Get_rank():06}"
    )
    gt_config.cache_settings["root_path"] = tmp_path

    # Compile in another directory
    backend = "dace:cpu"
    p1 = OrchestratedProgram(Backend("st:dace:cpu:KIJ"), None)
    p1()

    backend_sanitized = backend.replace(":", "")
    python_version = f"py{sys.version_info[0]}{sys.version_info[1]}"
    expected_cache_path = (
        tmp_path
        / ".gt_cache_000000"
        / f"{python_version}_1013"
        / f"{backend_sanitized}"
        / "tests"
        / "dsl"
        / "test_caches"
        / "_stencil"
    )
    assert expected_cache_path.exists()

    # Check relocatability by copying the first cache directory,
    # changing the path of gt_config.cache_settings and trying to Run on it
    relocated_path = tmp_path / ".my_relocated_cache_path"
    shutil.copytree(tmp_path / ".gt_cache_000000", relocated_path, dirs_exist_ok=False)
    gt_config.cache_settings["root_path"] = relocated_path

    p2 = OrchestratedProgram(Backend("st:dace:cpu:KIJ"), None)
    p2()

    relocated_cache_path = (
        relocated_path
        / ".gt_cache_000000"
        / f"{python_version}_1013"
        / f"{backend_sanitized}"
        / "tests"
        / "dsl"
        / "test_caches"
        / "_stencil"
    )
    assert relocated_cache_path.exists()
