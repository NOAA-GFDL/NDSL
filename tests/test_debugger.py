from pathlib import Path

import numpy as np
import yaml

from ndsl import NDSLRuntime
from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import Backend
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.debug import config as debug_config
from ndsl.debug import get_debugger
from ndsl.debug.debugger import Debugger, DebuggerStartFrom
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField


def test_debug_config_loader_reads_yaml(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "ndsl_debug_config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "stencils_or_class": ["StencilA"],
                "track_parameter_by_name": ["alpha"],
                "save_compute_domain_only": True,
                "dir_name": str(tmp_path),
                "save_all": False,
                "save_from": {"ndslruntime_name": "StencilA", "start_from_call": 2},
            }
        )
    )

    monkeypatch.setenv("NDSL_DEBUG_CONFIG", str(config_path))
    debugger = debug_config._set_debugger_from_config()

    assert isinstance(debugger, Debugger)
    assert debugger.rank == 0
    assert debugger.stencils_or_class == ["StencilA"]
    assert debugger.track_parameter_by_name == ["alpha"]
    assert debugger.save_compute_domain_only is True
    assert debugger.dir_name == str(tmp_path)
    assert debugger.save_all is False
    assert debugger.save_from.ndslruntime_name == "StencilA"
    assert debugger.save_from.start_from_call == 2


def test_debugger_start_from_threshold() -> None:
    start_from = DebuggerStartFrom(ndslruntime_name="StencilA", start_from_call=2)

    assert start_from.call_count == -1
    assert not start_from.can_run()

    start_from.record("other_runtime")
    assert start_from.call_count == -1
    assert not start_from.can_run()

    start_from.record("StencilA")
    assert start_from.call_count == 0
    assert not start_from.can_run()

    start_from.record("StencilA")
    start_from.record("StencilA")
    assert start_from.call_count == 2
    assert start_from.can_run()


def test_debugger_track_data_creates_track_file(tmp_path: Path) -> None:
    debugger = Debugger(
        track_parameter_by_name=["x"],
        dir_name=str(tmp_path),
        rank=1,
    )

    debugger.track_data({"x": np.arange(3), "y": np.arange(2)}, "MyStencil", is_in=True)

    track_file = tmp_path / "debug" / "tracks" / "x" / "R1" / "0_x_MyStencil-In.nc4"
    assert track_file.exists()
    assert debugger.track_parameter_count["x"] == 1


def test_debugger_save_from_waits_until_threshold(tmp_path: Path) -> None:
    debugger = Debugger(
        stencils_or_class=["StencilA"],
        save_all=False,
        dir_name=str(tmp_path),
        save_from=DebuggerStartFrom(ndslruntime_name="StencilA", start_from_call=1),
        rank=0,
    )

    output_file = (
        tmp_path / "debug" / "savepoints" / "R0" / "S000000_StencilA-Call0-In.nc4"
    )

    debugger.save_as_dataset({"x": np.arange(3)}, "StencilA", is_in=True)
    assert (
        not output_file.exists()
    ), "Debugger should not save before the configured call threshold"

    debugger.save_as_dataset({"x": np.arange(3)}, "StencilA", is_in=True)
    assert (
        output_file.exists()
    ), "Debugger should save once the start_from threshold is reached"
    assert debugger.step == 1


def test_ndslruntime_saves_data_for_python_backend(tmp_path: Path, monkeypatch) -> None:
    # Set configuration to be read by the global debugger
    config_path = tmp_path / "ndsl_debug_config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "stencils_or_class": ["DebugCode", "OtherStencil"],
                "track_parameter_by_name": ["A", "B"],
                "dir_name": str(tmp_path),
                "save_all": True,
            }
        )
    )
    monkeypatch.setenv("NDSL_DEBUG_CONFIG", str(config_path))
    debugger = get_debugger(True)

    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=5,
        ny=5,
        nz=3,
        nhalo=0,
        backend=Backend.python(),
    )

    A_ = quantity_factory.ones(dims=[I_DIM, J_DIM, K_DIM], units="n/a")
    B_ = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="n/a")

    # Code definition needs to be done after `_set_debugger_from_config` so that the class
    # gets the correct monkey patching of it's __call__ method
    def copy_stencil(A: FloatField, B: FloatField):
        with computation(PARALLEL), interval(...):
            B = A

    class DebugCode(NDSLRuntime):
        def __init__(self, stencil_factory, quantity_factory):
            super().__init__(stencil_factory)
            self.copy = stencil_factory.from_dims_halo(
                copy_stencil, compute_dims=[I_DIM, J_DIM, K_DIM]
            )

        def __call__(self, A: FloatField, B: FloatField):
            self.copy(A, B)

    code = DebugCode(stencil_factory, quantity_factory)
    code(A_, B_)

    print(tmp_path)

    qualname_of_locals = "test_ndslruntime_saves_data_for_python_backend.<locals>"

    input_file = (
        tmp_path
        / "debug"
        / "savepoints"
        / "R0"
        / f"S000000_{qualname_of_locals}.DebugCode-Call0-In.nc4"
    )
    output_file = (
        tmp_path
        / "debug"
        / "savepoints"
        / "R0"
        / f"S000003_{qualname_of_locals}.DebugCode-Call0-Out.nc4"
    )
    track_file_a = (
        tmp_path
        / "debug"
        / "tracks"
        / "A"
        / "R0"
        / f"0_A_{qualname_of_locals}.DebugCode-In.nc4"
    )
    track_file_b = (
        tmp_path
        / "debug"
        / "tracks"
        / "B"
        / "R0"
        / f"0_B_{qualname_of_locals}.DebugCode-In.nc4"
    )

    assert input_file.exists(), "NDSLRuntime should save input data"
    assert output_file.exists(), "NDSLRuntime should save output data"
    assert track_file_a.exists(), "Debugger should track A input data"
    assert track_file_b.exists(), "Debugger should track B input data"
    assert debugger
    assert debugger.step == 4
    assert (A_.field[:] == B_.field[:]).all()
