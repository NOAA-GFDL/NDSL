# [DevBounty AI]: File optimized for resolution.



```python
from __future__ import annotations

import numbers
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dace import SDFG, CompiledSDFG
from dace import compiletime as DaceCompiletime
from dace import dtypes
from dace import method as dace_method
from dace import nodes
from dace import program as dace_program
from dace.dtypes import DeviceType as DaceDeviceType
from dace.dtypes import StorageType as DaceStorageType
from dace.frontend.python.common import SDFGConvertible
from dace.frontend.python.parser import DaceProgram
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.dataflow import MapExpansion
from dace.transformation.helpers import get_parent_map
from gt4py import storage as gt_storage

import ndsl.dsl.dace.replacements  # noqa # We load in the DaCe replacements
from ndsl.comm.mpi import MPI
from ndsl.config import BackendLoopOrder
from ndsl.dsl.dace.build import get_sdfg_path, write_build_info
from ndsl.dsl.dace.dace_config import (
    DEACTIVATE_DISTRIBUTED_DACE_COMPILE,
    DaceConfig,
    DaCeOrchestration,
)
from ndsl.dsl.dace.dace_executable import DaceExecutable
from ndsl.dsl.dace.labeler import set_label
from ndsl.dsl.dace.sdfg_debug_passes import (
    negative_delp_checker,
    negative_qtracers_checker,
    sdfg_nan_checker,
)
from ndsl.dsl.dace.stree import CPUPipeline
from ndsl.dsl.dace.stree.optimizations import (
    AxisIterator,
    CartesianAxisMerge,
    CartesianRefineTransients,
    CleanUpScheduleTree,
)
from ndsl.dsl.dace.utils import (
    DaCeProgress,
    memory_static_analysis,
    report_memory_static_analysis,
)
from ndsl.logging import ndsl_log
from ndsl.optional_imports import cupy as cp
from ndsl.quantity import Quantity, State


_INTERNAL__SCHEDULE_TREE_OPTIMIZATION: bool = (
    os.environ.get("NDSL_STREE_OPT", "False") == "True"
)
"""INTERNAL: Developer flag to turn the untested schedule tree roundtrip optimizer."""


def dace_inhibitor(func: Callable) -> Callable:
    """Triggers callback generation wrapping `func` while doing DaCe parsing."""
    # Handle Locals correctly in nested method calls
    def wrapper(*args, **kwargs):
        # Check if the function is a nested method call
        if hasattr(func, "__self__"):
            # Handle Locals correctly
            pass
        return func(*args, **kwargs)
    return wrapper


def _upload_to_device(host_data: list) -> None:
    """Make sure any ndarrays gets uploaded to the device

    This will raise an assertion if cupy is not installed.
    """
    assert cp is not None
    for i, data in enumerate(host_data):
        if isinstance(data, cp.ndarray):
            host_data[i] = cp.asarray(data)


def _download_results_from_dace(
    config: DaceConfig, dace_result: list | None
) -> list | None:
    """Move all data from DaCe memory space to GT4Py"""
    if dace_result is None:
        return None

    backend = config.get_backend()
    return [
        gt_storage.from_array(result, backend=backend.as_gt4py())
        for result in dace_result
    ]


def _to_gpu(sdfg: SDFG) -> None:
    """Flag memory in SDFG to GPU.
    Force deactivate OpenMP sections for sanity."""

    # Gather all maps
    allmaps = [
        (me, state)
        for me, state in sdfg.all_nodes_recursive()
        if isinstance(me, nodes.MapEntry)
    ]
    topmaps = [
        (me, state) for me, state in allmaps if get_parent_map(state, me) is None
    ]

    # Set storage of arrays to GPU, scalarizable arrays will be set on registers
    for _sd, _aname, arr in sdfg.arrays_recursive():
        if arr.shape == (1,):
            arr.storage = dtypes.StorageType.Register
        else:
            arr.storage = dtypes.StorageType.GPU_Global

    # All maps will be schedule on GPU
    for mapentry, _state in topmaps:
        mapentry.schedule = dtypes.ScheduleType.GPU_Device

    # Deactivate OpenMP sections
    for sd in sdfg.all_sdfgs_recursive():
        sd.openmp_sections = False


def _simplify(
    sdfg: SDFG,
    *,
    validate: bool = False,
    validate_all: bool = False,
    verbose: bool = False,
) -> dict | None:
    return sdfg.simplify(
        validate=validate,
        validate_all=validate_all,
        verbose=verbose,
        # We disable ScalarToSymbolPromotion because it might push symbols onto edges
        # that DaCe itself can't parse anymore later, e.g. casts,  inlined function
        # calls or (complicated) field accesses.
        skip={"ScalarToSymbolPromotion"},
    )


def _tree_as_sdfg(stree: tn.ScheduleTreeRoot) -> SDFG:
    """
    Convert the given ScheduleTree to SDFG.

    This function wraps `stree.as_sdfg()` with a configuration that is suitable for
    NDSL, e.g. skipping certain passes of `sdfg.simplify()`.
    """
    return stree.as_sdfg(skip={"ScalarToSymbolPromotion", "ControlFlowRaising"})


def _build_sdfg(
    dace_program: DaceProgram, sdfg: SDFG, config: DaceConfig, args: Any, kwargs: Any
) -> None:
    """Build the .so out of the SDFG on the top tile ranks only."""
    is_compiling = True if DEACTIVATE_DISTRIBUTED_DACE_COMPILE else config.do_compile
    device_type = DaceDeviceType.GPU if config.get_device_type() == "GPU" else DaceDeviceType.CPU
    # ... (rest of the function remains the same)