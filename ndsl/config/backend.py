from __future__ import annotations

from enum import Enum
from typing import Any, Final

import gt4py.cartesian.backend as gt_backend


class BackendStrategy(Enum):
    """Strategy for the code execution"""

    STENCIL = "st"
    ORCHESTRATION = "orch"


class BackendTargetDevice(Enum):
    """Target device"""

    CPU = "cpu"
    GPU = "gpu"


class BackendFramework(Enum):
    """Main lower-level framework (or language) backend relies on"""

    GRIDTOOLS = "gt"
    DACE = "dace"
    PYTHON = "python"


_NDSL_TO_GT4PY_BACKEND_NAMING = {
    "st:python:cpu:debug": "debug",
    "st:python:cpu:numpy": "numpy",
    "st:gt:cpu:IJK": "gt:cpu_kfirst",
    "st:gt:cpu:KJI": "gt:cpu_ifirst",
    "st:gt:gpu:KJI": "gt:gpu",
    "st:dace:cpu:IJK": "dace:cpu_kfirst",
    "orch:dace:cpu:IJK": "dace:cpu_kfirst",
    "st:dace:cpu:KIJ": "dace:cpu",
    "orch:dace:cpu:KIJ": "dace:cpu",
    "st:dace:cpu:KJI": "dace:cpu_KJI",
    "orch:dace:cpu:KJI": "dace:cpu_KJI",
    "st:dace:gpu:KJI": "dace:gpu",
    "orch:dace:gpu:KJI": "dace:gpu",
}
"""Internal: match the NDSL backend names with the GT4Py names"""

_FORTRAN_LOOP_LAYOUT = (2, 1, 0)
"""Fortran is a column-first (or stride-first) memory system,
which in the internal gt4py loop layout means I (or axis[0]) has
the higher value, e.g. "higher importance to run first":

for k # Layout=0
    for j # Layout=1
        for i # Layout=2

"""


class Backend:
    """Backend for NDSL.

    The backend is a string concatenating information on the intent of the user
    for a given execution separated by a ':'.

    It describes to NDSL the strategy, device and framework to be used
    on the frontend code. Additionally, it gives a hint toward the macro-strategy
    for loop ordering (IJK, KJI, etc.) or a more broad intent (debug, numpy).

    For convenience, shorcuts are given to the most common needs (
    `backend_python`, `backend_cpu`, `backend_gpu`).
    """

    def __init__(self, ndsl_backend: str) -> None:
        # Checks for existence and form
        if ndsl_backend not in _NDSL_TO_GT4PY_BACKEND_NAMING:
            raise ValueError(
                f"Unknown {ndsl_backend}, options are {list(_NDSL_TO_GT4PY_BACKEND_NAMING.keys())}"
            )
        parts = ndsl_backend.split(":")
        if len(parts) != 4:
            raise ValueError(f"Backend {ndsl_backend} is ill-formed.")

        # Breakdown and save into internal parameters
        self._humanly_readable = ndsl_backend
        self._strategy = BackendStrategy(parts[0].lower())
        self._framework = BackendFramework(parts[1].lower())
        self._device = BackendTargetDevice(parts[2].lower())
        self._loop_order = parts[3]

        # Check GPU capacity
        if (
            self._device == BackendTargetDevice.GPU
            and gt_backend.from_name(self.as_gt4py()).storage_info["device"] != "gpu"
        ):
            raise ValueError(
                f"NDSL backend requested ({self._humanly_readable}) tagets GPU,"
                f"but requests a non-GPU backend from GT4Py ({self.as_gt4py()})."
            )

    def __str__(self) -> str:
        return self.as_humanly_readable()

    def __repr__(self) -> str:
        return self.as_humanly_readable()

    def __eq__(self, other: object) -> bool:
        return self._humanly_readable == self._humanly_readable

    def __hash__(self) -> int:
        return hash(self._humanly_readable)

    def __add__(self, other: str) -> Any:
        """Concatenation operators"""
        if isinstance(other, Backend):
            raise TypeError("OperationError: Backend cannot add to another Backend")
        return str(self) + other

    def __radd__(self, other: str) -> Any:
        """Concatenation operators"""
        if isinstance(other, Backend):
            raise TypeError("OperationError: Backend cannot add to another Backend")
        return other + str(self)

    @staticmethod
    def python() -> Backend:
        """Default backend for quick iterative work."""
        return backend_python

    @staticmethod
    def cpu() -> Backend:
        """Default performance backend targeting CPU device"""
        return backend_cpu

    @staticmethod
    def gpu() -> Backend:
        """Default performance backend targeting GPU device"""
        return backend_gpu

    @property
    def device(self) -> BackendTargetDevice:
        return self._device

    @property
    def framework(self) -> BackendFramework:
        return self._framework

    def as_gt4py(self) -> str:
        """Given an NDSL backend, give back a GT4Py equivalent"""
        return _NDSL_TO_GT4PY_BACKEND_NAMING[self._humanly_readable]

    def as_humanly_readable(self) -> str:
        return self._humanly_readable

    def as_safe_for_path(self) -> str:
        return self._humanly_readable.replace(":", "_")

    def is_orchestrated(self) -> bool:
        return self._strategy == BackendStrategy.ORCHESTRATION

    def is_stencil(self) -> bool:
        return self._strategy == BackendStrategy.STENCIL

    def is_gpu_backend(self) -> bool:
        return self._device == BackendTargetDevice.GPU

    def is_fortran_aligned(self) -> bool:
        """Check that the standard 3D field on cartesian axis is memory-aligned with Fortran
        striding."""
        return _FORTRAN_LOOP_LAYOUT == gt_backend.from_name(
            self.as_gt4py()
        ).storage_info["layout_map"](("I", "J", "K"))


backend_python: Final[Backend] = Backend("st:python:cpu:debug")
"""Default backend for quick iterative work."""

backend_cpu: Final[Backend] = Backend("orch:dace:cpu:IJK")
"""Default performance backend targeting CPU device"""

backend_gpu: Final[Backend] = Backend("orch:dace:gpu:KJI")
"""Default performance backend targeting GPU device"""
