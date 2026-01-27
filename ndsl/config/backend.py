from __future__ import annotations

from enum import Enum
from typing import Any

import gt4py.cartesian.backend as gt_backend


class BackendStrategy(Enum):
    """Strategy for the code execution"""

    STENCIL = "st"
    ORCHESTRATION = "orch"


class BackendTargetDevice(Enum):
    """Targeted device"""

    CPU = "cpu"
    GPU = "gpu"


class BackendFramework(Enum):
    """Main framework (or language) backend relies on"""

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

_NDSL_SHORT_TO_LONG_NAME = {
    "numpy": "st:python:cpu:numpy",
    "debug": "st:python:cpu:debug",
}
"""Internal: match short name to long name form"""


class Backend:
    """Backend for NDSL"""

    def __init__(self, ndsl_backend: str) -> None:
        # Swap short form to long form
        if ndsl_backend in _NDSL_SHORT_TO_LONG_NAME.keys():
            ndsl_backend = _NDSL_SHORT_TO_LONG_NAME[ndsl_backend]

        # Checks for existence and form
        if ndsl_backend not in _NDSL_TO_GT4PY_BACKEND_NAMING.keys():
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
                f"Coding error: NDSL backend requested {self._humanly_readable} "
                f"translate to non-GPU {self.as_gt4py()} GT4Py backend"
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
    def debug() -> Backend:
        return Backend("st:python:cpu:debug")

    @staticmethod
    def python() -> Backend:
        return Backend("st:python:cpu:numpy")

    @staticmethod
    def performance_cpu() -> Backend:
        return Backend("orch:dace:cpu:IJK")

    @staticmethod
    def hybrid_fortran_cpu() -> Backend:
        return Backend("orch:dace:cpu:KJI")

    @staticmethod
    def performance_gpu() -> Backend:
        return Backend("orch:dace:gpu:KJI")

    @property
    def device(self) -> BackendTargetDevice:
        return self._device

    @property
    def framework(self) -> BackendFramework:
        return self._framework

    def as_gt4py(self) -> str:
        if self._humanly_readable in _NDSL_TO_GT4PY_BACKEND_NAMING.keys():
            return _NDSL_TO_GT4PY_BACKEND_NAMING[self._humanly_readable]
        raise ValueError(
            f"Backend {self._humanly_readable} cannot be translate to GT4Py"
        )

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


# Those two internal values are used for default parameters values
# as it is bad practice to call a function in default argument value
_BACKEND_PERFORMANCE_CPU = Backend.performance_cpu()
"""Internal: cache performance CPU. Please use Backend.performance_cpu()."""
_BACKEND_PYTHON = Backend.python()
"""Internal: cache performance CPU. Please use Backend.python()."""
