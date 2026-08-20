import dataclasses
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import dace
from dace.frontend.python.parser import DaceProgram
from dace.sdfg.sdfg import SDFG
from gt4py import storage as gt_storage
from mpi4py import MPI

from ndsl.comm.local_comm import LocalComm
from ndsl.config.backend import Backend
from ndsl.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from ndsl.dsl.dace.utils import DaCeProgress, upload_to_device
from ndsl.performance.collector import (
    AbstractPerformanceCollector,
    PerformanceCollector,
)
from ndsl.quantity import State

DaceExecutables = dict[DaceProgram, "DaceExecutable"]
DACE_EXECUTABLE_CACHE: DaceExecutables = {}

_BUNDLE_DIRECTORY_NAME = "NDSLRecording"
_PARSED_SDFG_NAME = "parsed_sdfg"
_OPTIMIZED_SDFG_NAME = "optimized_sdfg"


def _download_results_from_dace(
    backend: Backend, dace_result: list | None
) -> list | None:
    """Move all data from DaCe memory space to GT4Py"""
    if dace_result is None:
        return None

    return [
        gt_storage.from_array(result, backend=backend.as_gt4py())
        for result in dace_result
    ]


@dataclasses.dataclass
class DaceExecutable:
    """Translate GT4Py-frozen parsed SDFG into an executable (dynamics lib)
    and its marshalled arguments for fast execution.
    """

    compiled_sdfg: dace.CompiledSDFG | None
    """Loaded compiled SDFG. Allowed to be None only when using `from_serialized_bundle`"""

    performance_collector: AbstractPerformanceCollector
    """Performance timer used to time runtime operations (overhead and numerics)"""

    mode: DaCeOrchestration
    """Orchestration mode the executable is built under"""

    name: str
    """DaCe program name"""

    backend: Backend
    """Backend the executable is build for"""

    arguments: dict[str, Any] | None = None
    """Arguments as C-ready pointers"""

    original_unoptimized_sdfg: SDFG | None = None
    """Optional: Unoptimized SDFG coming from GT4Py-frozen stencils + parsing."""

    _arguments_hash: int = 0
    """Internal: hash reflecting the python/C pointers arguments"""

    _skip_hash: bool = False
    """Internal: skip hash computation because some
    arguments where detected to be un-hashable last time"""

    _record: bool = False
    """Internal: next execution will be recorded for replayability"""

    def run(self, dace_program: DaceProgram, args: Any, kwargs: Any) -> list | None:
        """Execute the loaded executable with as little overhead as possible"""
        assert self.compiled_sdfg
        with self.performance_collector.timestep_timer.clock(f"{self.name}.Call"):
            if self.mode not in [DaCeOrchestration.BuildAndRun, DaCeOrchestration.Run]:
                raise ValueError(f"Unexpected DaceOrchestration mode `{self.mode}`.")

            with DaCeProgress(self.mode, "Run"):
                if self.backend.is_gpu_backend():
                    upload_to_device(list(args) + list(kwargs.values()))

                # Marshall given arguments into C-binding ready memory
                with self.performance_collector.timestep_timer.clock(
                    f"{self.name}.ArgMarshalling"
                ):
                    hash_ = self._hash_expected_dsl_args(args, kwargs)
                    if self.arguments is None or hash_ != self._arguments_hash:
                        marshalled_sdfg_args = dace_program._create_sdfg_args(
                            self.compiled_sdfg.sdfg,
                            args,
                            kwargs,
                        )
                        self._arguments_hash = hash_
                        self.arguments = marshalled_sdfg_args

                if self._record:
                    self.serialize()
                    self._record = False

                # Calling into the C
                with self.performance_collector.timestep_timer.clock(
                    f"{self.name}.Runtime"
                ):
                    results = self.compiled_sdfg(**self.arguments)

        self.performance_collector.collect_performance()

        return _download_results_from_dace(self.backend, results)

    @classmethod
    def from_compiled(
        cls,
        dace_program: DaceProgram,
        config: DaceConfig,
        compiled_sdfg: dace.CompiledSDFG,
        original_unoptimized_sdfg: SDFG | None = None,
    ) -> "DaceExecutable":
        return cls(
            name=dace_program.name,
            compiled_sdfg=compiled_sdfg,
            performance_collector=config.performance_collector,
            mode=config.get_orchestrate(),
            backend=config.get_backend(),
            arguments={},
            original_unoptimized_sdfg=original_unoptimized_sdfg,
            _record=os.getenv("NDSL_RECORD_ORCHESTRATION", "False").lower() == "true",
        )

    def serialize(self) -> None:
        """Serialize arguments and code for blind replayability using `replay`

        Only serialize the rank 0
        """
        if MPI.COMM_WORLD.Get_rank() != 0:
            return  # only save rank 0.

        assert self.compiled_sdfg
        bundle_dir = Path(
            self.compiled_sdfg.sdfg.build_folder + "/" + _BUNDLE_DIRECTORY_NAME
        )
        bundle_dir.mkdir(exist_ok=True, parents=True)

        with open(bundle_dir / "de_args.pickle", "wb") as f:
            pickle.dump(self.arguments, f)

        if self.original_unoptimized_sdfg:
            self.original_unoptimized_sdfg.save(
                f"{bundle_dir}/{_PARSED_SDFG_NAME}.sdfgz", compress=True
            )

        self.compiled_sdfg.sdfg.save(
            f"{bundle_dir}/{_OPTIMIZED_SDFG_NAME}.sdfgz", compress=True
        )
        with open(bundle_dir / "backend.txt", "w") as f:
            f.write(self.backend.as_humanly_readable())

    @classmethod
    def from_serialized_bundle(
        cls, bundle_dir: str, *, do_compile: bool = True
    ) -> "DaceExecutable":
        """Read a serialized bundle and ready the system for replay."""

        bundle_path = Path(bundle_dir) / _BUNDLE_DIRECTORY_NAME

        with open(bundle_path / "de_args.pickle", "rb") as f:
            arguments = pickle.load(f)

        gt4py_sdfg_bundle_sdfg = bundle_path / f"{_PARSED_SDFG_NAME}.sdfgz"
        if gt4py_sdfg_bundle_sdfg.exists():
            original_unoptimized_sdfg = SDFG.from_file(str(gt4py_sdfg_bundle_sdfg))

        sdfg = SDFG.from_file(f"{bundle_path}/{_OPTIMIZED_SDFG_NAME}.sdfgz")
        with open(bundle_path / "backend.txt", "r") as f:
            backend = Backend(f.readlines()[0])

        csdfg = sdfg.compile() if do_compile else None

        return cls(
            name=sdfg.name,
            compiled_sdfg=csdfg,
            performance_collector=PerformanceCollector("replay", LocalComm(0, 1, {})),
            mode=DaCeOrchestration.Run,
            backend=backend,
            arguments=arguments,
            original_unoptimized_sdfg=original_unoptimized_sdfg,
            _record=False,
        )

    def _hash_expected_dsl_args(self, args: tuple[Any], kwargs: dict[str, Any]) -> int:
        """Hash direct memory of NDSL expected types.

        Handling the following types:
            - quantity | Numpy.ndarray | cupy.ndarray: we hash the C pointer through the array interface,
            - state: called into a bespoke function,
            - everything else is passed as-is to `hash` which _can_ fail.
        """
        if self._skip_hash:
            self.arguments = None  # Flush arguments to force recompute
            return 0

        to_hash = []
        for arg in list(args) + list(kwargs.values()):
            if hasattr(arg, "__array_interface__"):
                to_hash.append(arg.__array_interface__["data"][0])
            elif hasattr(arg, "__cuda_array_interface__"):
                to_hash.append(arg.__cuda_array_interface__["data"][0])
            elif isinstance(arg, State):
                to_hash.append(arg._hash())
            else:
                to_hash.append(arg)

        try:
            h = hash(tuple(to_hash))
        except TypeError as e:
            warnings.warn(
                f"[NDSL|Orchestration] argument type aren't hashable: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.arguments = None  # Flush arguments to force recompute
            self._skip_hash = True  # Skip future checks
            return 0

        return h

    def replay(self, *, bench: bool = False) -> None:
        """Replay executable using last cached arguments"""
        if not self.arguments:
            raise RuntimeError(f"Cannot replay {self.name} - no arguments available")

        self.performance_collector.start_cuda_profiler()

        if not self.compiled_sdfg:
            raise RuntimeError("Replay impossible, CompiledSDFG is not set.")

        self.compiled_sdfg(**self.arguments)

        if bench:
            with self.performance_collector.total_timer.clock("all"):
                for _ in range(1000):
                    with self.performance_collector.clock_timestep("ts"):
                        self.compiled_sdfg(**self.arguments)

            self.performance_collector.write_out_rank_0(
                self.backend, True, dt_atmos=-1.0, sim_status="done"
            )

        self.performance_collector.stop_cuda_profiler()
