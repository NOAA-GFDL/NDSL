from __future__ import annotations

import copy
import numbers
import os
import sys
from pathlib import Path
from pprint import pformat
from typing import Any

from dace import SDFG, DeviceType, dtypes, nodes
from dace.dtypes import DeviceType as DaceDeviceType
from dace.dtypes import ScheduleType
from dace.dtypes import StorageType as DaceStorageType
from dace.frontend.python.parser import DaceProgram
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.dataflow import MapCollapse, MapExpansion
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
from dace.transformation.helpers import get_parent_map
from gt4py import storage as gt_storage

import ndsl.dsl.dace.replacements  # noqa # We load in the DaCe replacements
from ndsl import Backend, OptimizationConfig, ndsl_log
from ndsl.comm.mpi import MPI
from ndsl.dsl.dace.build import get_sdfg_path, write_build_info
from ndsl.dsl.dace.builder.sdfg.debug_passes import (
    negative_delp_checker,
    negative_qtracers_checker,
    sdfg_nan_checker,
)
from ndsl.dsl.dace.dace_config import (
    DEACTIVATE_DISTRIBUTED_DACE_COMPILE,
    DaceConfig,
    DaCeOrchestration,
)
from ndsl.dsl.dace.dace_executable import DACE_EXECUTABLE_CACHE, DaceExecutable
from ndsl.dsl.dace.hardware_config import get_gpu_hardware_defaults
from ndsl.dsl.dace.stree import CPUPipeline, GPUPipeline
from ndsl.dsl.dace.stree.pipeline import StreePipeline
from ndsl.dsl.dace.utils import (
    DaCeProgress,
    memory_static_analysis,
    report_memory_static_analysis,
)
from ndsl.optional_imports import cupy as cp

_INTERNAL__SCHEDULE_TREE_OPTIMIZATION_PASSES: list[tn.ScheduleNodeVisitor] | None = None


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
        # We disable LiftTrivialIf because it takes forever on larger graphs
        skip={"ScalarToSymbolPromotion", "LiftTrivialIf"},
    )


def _tree_as_sdfg(stree: tn.ScheduleTreeRoot) -> SDFG:
    """
    Convert the given ScheduleTree to SDFG.

    This function wraps `stree.as_sdfg()` with a configuration that is suitable for
    NDSL, e.g. skipping certain passes of `sdfg.simplify()`.
    """
    return stree.as_sdfg(
        validate=False,
        simplify=True,
        skip={"ScalarToSymbolPromotion", "ControlFlowRaising"},
    )


def _optimization_pipeline(
    config: OptimizationConfig,
    device_type: DeviceType,
    backend: Backend,
    *,
    passes: list[tn.ScheduleNodeVisitor] | None = None,
    cache_directory: Path | None = None,
) -> StreePipeline:
    if device_type == DeviceType.CPU:
        return CPUPipeline(
            config, backend, passes=passes, cache_directory=cache_directory
        )

    if device_type == DeviceType.GPU:
        return GPUPipeline(
            config, backend, passes=passes, cache_directory=cache_directory
        )

    raise ValueError(
        f"Unknown device type `{device_type}`, expected {DeviceType.CPU} or {DeviceType.GPU}."
    )


def optimize_full_program_sdfg(
    dace_program: DaceProgram,
    parsed_sdfg: SDFG,
    config: DaceConfig,
    optimization_config: OptimizationConfig | None,
    args: Any,
    kwargs: Any,
) -> None:
    """Optimize and compile the .so from the parsed SDFG. Build on the top tile ranks only."""
    is_compiling = True if DEACTIVATE_DISTRIBUTED_DACE_COMPILE else config.do_compile
    device_type = DaceDeviceType.GPU if config.is_gpu_backend() else DaceDeviceType.CPU
    backend_name = config.get_backend()
    mode = config.get_orchestrate()

    if is_compiling:

        # Enforce cache directory made so all downstream caching file
        # won't hit an non existing directory
        Path(parsed_sdfg.build_folder).mkdir(parents=True, exist_ok=True)

        unoptimized_sdfg = copy.copy(parsed_sdfg)

        if optimization_config is None:
            ndsl_log.debug(
                f"Using default optimization config for {parsed_sdfg.label}."
            )
            optimization_config = OptimizationConfig()

        ndsl_log.debug(f"Compiling config:\n{pformat(optimization_config, indent=2)}")
        # Fully specialize all known symbols and then propagate these changes in the simplify
        # pass that follows. This is not only a smart idea in general, but also simplifies (haha)
        # the schedule tree (optimization) roundtrip.
        with DaCeProgress(mode, "Fully specialize symbols"):
            for my_sdfg in parsed_sdfg.all_sdfgs_recursive():
                if my_sdfg.parent_nsdfg_node is not None:
                    repl_dict: dict[str, str] = {}
                    for sym, val in my_sdfg.parent_nsdfg_node.symbol_mapping.items():
                        if isinstance(val, numbers.Number):
                            repl_dict[sym] = str(val)
                    my_sdfg.replace_dict(repl_dict)

            if config.verbose_orchestration:
                ndsl_log.debug("saving 00-combined_from_stencils.sdfgz")
                parsed_sdfg.save(
                    os.path.abspath(
                        f"{parsed_sdfg.build_folder}/00-combined_from_stencils.sdfgz"
                    ),
                    compress=True,
                )

        if config.is_gpu_backend():
            with DaCeProgress(mode, "Configure maps to run on GPU"):
                for this_sdfg in parsed_sdfg.all_sdfgs_recursive():
                    for state in this_sdfg.states():
                        for node in state.nodes():
                            if (
                                isinstance(node, nodes.EntryNode)
                                and node.schedule != ScheduleType.Sequential
                            ):
                                node.schedule = ScheduleType.GPU_Device

            if config.verbose_orchestration:
                ndsl_log.debug("saving 00-gpu-maps.sdfgz")
                parsed_sdfg.save(
                    os.path.abspath(f"{parsed_sdfg.build_folder}/00-gpu-maps.sdfgz"),
                    compress=True,
                )

        with DaCeProgress(mode, "Simplify (1)"):
            _simplify(parsed_sdfg)
            if config.verbose_orchestration:
                ndsl_log.debug("saving 01-simplify.sdfgz")
                parsed_sdfg.save(
                    os.path.abspath(f"{parsed_sdfg.build_folder}/01-simplify_1.sdfgz"),
                    compress=True,
                )

        if optimization_config.stree.enabled:
            # Here be 🐉 - but tests exists in test_optimization.py
            with DaCeProgress(mode, "Schedule Tree: generate from SDFG"):
                # Break all loops into uni-dimensional loops to simplify optimizations
                parsed_sdfg.apply_transformations_repeated(
                    MapExpansion,
                    options={
                        "inner_schedule": (
                            ScheduleType.GPU_Device
                            if device_type is DeviceType.GPU
                            else ScheduleType.Default
                        )
                    },
                    validate=True,
                )
                stree = parsed_sdfg.as_schedule_tree()
                if config.verbose_orchestration:
                    ndsl_log.debug("saving 02-pre_opt.stree.txt")
                    with open(
                        os.path.abspath(
                            f"{parsed_sdfg.build_folder}/02-pre_opt.stree.txt"
                        ),
                        "w+",
                    ) as f:
                        f.write(stree.as_string())

            with DaCeProgress(mode, "Schedule Tree: optimization"):
                pipeline = _optimization_pipeline(
                    optimization_config,
                    device_type,
                    backend_name,
                    cache_directory=Path(parsed_sdfg.build_folder),
                    passes=_INTERNAL__SCHEDULE_TREE_OPTIMIZATION_PASSES,
                )
                pipeline.run(stree, verbose=config.verbose_schedule_tree_optimizations)
                if config.verbose_orchestration:
                    ndsl_log.debug("saving 03-post_opt.stree.txt")
                    with open(
                        os.path.abspath(
                            f"{parsed_sdfg.build_folder}/03-post_opt.stree.txt"
                        ),
                        "w+",
                    ) as f:
                        f.write(stree.as_string())

            with DaCeProgress(mode, "Schedule Tree: go back to SDFG"):
                parsed_sdfg = _tree_as_sdfg(stree)
                if config.verbose_orchestration:
                    ndsl_log.debug("saving 04-from_stree.sdfgz")
                    parsed_sdfg.save(
                        os.path.abspath(
                            f"{parsed_sdfg.build_folder}/04-from_stree.sdfgz"
                        ),
                        compress=True,
                    )

        # We want all maps properly collapse to make sure the codegen will see nD parallel
        # axis as a single kernelizable map
        with DaCeProgress(mode, "Collapse maps"):
            # permissive: allow `MapCollapse` to collapse maps with different schedules
            # progress: do not print intermediate transformations applied
            # validate: do not validate after applying all transformations
            parsed_sdfg.apply_transformations_repeated(
                MapCollapse, permissive=True, progress=False, validate=False
            )

        with DaCeProgress(mode, "Make transient persistents"):
            # Make the transients array persistents
            if config.is_gpu_backend():
                # TODO
                # The following should happen on the stree level
                _to_gpu(parsed_sdfg)
                make_transients_persistent(sdfg=parsed_sdfg, device=device_type)

                # Upload args to device
                _upload_to_device(list(args) + list(kwargs.values()))
            else:
                # TODO
                # The following should happen on the stree level
                for _sd, _aname, arr in parsed_sdfg.arrays_recursive():
                    if arr.shape == (1,):
                        arr.storage = DaceStorageType.Register
                make_transients_persistent(sdfg=parsed_sdfg, device=device_type)

        if config.is_gpu_backend():
            with DaCeProgress(mode, "Apply GPU transformations"):
                # Set block size on GPU maps and collect callback
                # tasklets to exclude next
                gpu_defaults = get_gpu_hardware_defaults()
                exclude_taskslets_list = []

                for me, _state in parsed_sdfg.all_nodes_recursive():
                    if (
                        isinstance(me, nodes.MapEntry)
                        and me.map.schedule == ScheduleType.GPU_Device
                    ) and me.map.gpu_block_size is None:
                        me.map.gpu_block_size = gpu_defaults.block_size

                    if isinstance(me, nodes.Tasklet) and "callback_" in me.label:
                        exclude_taskslets_list.append(me.label)

                parsed_sdfg.apply_transformations_repeated(
                    AddThreadBlockMap, print_report=False
                )

                if optimization_config.gpu.common_gpu_xforms:
                    with DaCeProgress(mode, "Apply common GPU xforms"):
                        # Apply common GPU transforms (includes a simplify)
                        # while making sure tasklet remain on the host
                        from dace.transformation.interstate import GPUTransformSDFG

                        parsed_sdfg.apply_transformations(
                            GPUTransformSDFG,
                            options={
                                "exclude_tasklets": ",".join(exclude_taskslets_list),
                                "host_data": ["__pystate"],
                            },
                        )
                else:
                    with DaCeProgress(mode, "GPU simplify"):
                        _simplify(parsed_sdfg)

                if config.verbose_orchestration:
                    ndsl_log.debug("saving 05-apply_gpu_xforms.sdfgz")
                    parsed_sdfg.save(
                        os.path.abspath(
                            f"{parsed_sdfg.build_folder}/05-apply_gpu_xforms.sdfgz"
                        ),
                        compress=True,
                    )
        else:
            with DaCeProgress(mode, "Simplify (2)"):
                _simplify(parsed_sdfg)
                if config.verbose_orchestration:
                    ndsl_log.debug("saving 05-simplify_2.sdfgz")
                    parsed_sdfg.save(
                        os.path.abspath(
                            f"{parsed_sdfg.build_folder}/05-simplify_2.sdfgz"
                        ),
                        compress=True,
                    )
        # Move all memory that can be into a pool to lower memory pressure for GPU
        # We skip this memory optimization for CPU because we don't have a memory
        # pool available yet (DaCe v1)

        if config.is_gpu_backend():
            with DaCeProgress(mode, "Turn Persistents into pooled Scope"):
                memory_pooled = 0.0
                for _sd, _aname, arr in parsed_sdfg.arrays_recursive():
                    # Change Persistent memory (sub-SDFG) into Scope and flag it.
                    if arr.lifetime == dtypes.AllocationLifetime.Persistent:
                        arr.pool = True
                        memory_pooled += arr.total_size * arr.dtype.bytes
                        arr.lifetime = dtypes.AllocationLifetime.Scope
                memory_pooled = float(memory_pooled) / (1024 * 1024)
                ndsl_log.debug(
                    f"{DaCeProgress.default_prefix(mode)} Pooled {memory_pooled:.2f} mb",
                )

        # Set of debug tools inserted in the SDFG when dace.conf "syncdebug"
        # is turned on.
        if config.get_sync_debug():
            with DaCeProgress(mode, "Tooling the SDFG for debug"):
                sdfg_nan_checker(parsed_sdfg)
                negative_delp_checker(parsed_sdfg)
                negative_qtracers_checker(parsed_sdfg)

        # Compile
        with DaCeProgress(mode, "Codegen & compile"):
            compiled_sdfg = parsed_sdfg.compile()
            DACE_EXECUTABLE_CACHE[dace_program] = DaceExecutable.from_compiled(
                dace_program=dace_program,
                config=config,
                compiled_sdfg=compiled_sdfg,
                original_unoptimized_sdfg=unoptimized_sdfg,
            )

        # Printing analysis of the compiled SDFG
        with DaCeProgress(mode, "Build finished. Running memory static analysis"):
            report = report_memory_static_analysis(
                parsed_sdfg, memory_static_analysis(parsed_sdfg), False
            )
            ndsl_log.info(f"{DaCeProgress.default_prefix(mode)} {report}")

        # Store build info in the common cache directory
        write_build_info(
            parsed_sdfg, config.layout, config.tile_resolution, report, backend_name
        )

    # Compilation done.
    # On Build: all ranks sync, then exit.
    # On BuildAndRun: all ranks sync, then load the SDFG from
    #                 the expected path (made available by build).
    # We use a "CompiledSDFG" which keep the `so` online but _won't_
    # do the marshalling of the arguments at call time. For this we call
    # `dace_program._create_sdfg_args`. There's optimization potential for
    # re-entry cost there.

    mode = config.get_orchestrate()
    # DEV NOTE: we explicitly use MPI.COMM_WORLD here because it is
    # a true multi-machine sync, outside of our own communicator class.
    if mode == DaCeOrchestration.Build:
        MPI.COMM_WORLD.Barrier()  # Protect against early exist which kill SLURM jobs
        ndsl_log.info(f"{DaCeProgress.default_prefix(mode)} Build only, exiting.")
        sys.exit(0)

    if mode == DaCeOrchestration.BuildAndRun:
        if not is_compiling:
            ndsl_log.info(
                f"{DaCeProgress.default_prefix(mode)} Rank is not compiling. "
                "Waiting for compilation to end on all other ranks..."
            )
        MPI.COMM_WORLD.Barrier()

        if not is_compiling:
            with DaCeProgress(mode, "Loading"):
                sdfg_path = get_sdfg_path(
                    dace_program.name, config, override_run_only=True
                )
                if sdfg_path is None:
                    raise ValueError("Couldn't load SDFG post build")
                compiled_sdfg, _ = dace_program.load_precompiled_sdfg(
                    sdfg_path, *args, **kwargs
                )
                DACE_EXECUTABLE_CACHE[dace_program] = DaceExecutable.from_compiled(
                    dace_program=dace_program,
                    config=config,
                    compiled_sdfg=compiled_sdfg,
                    original_unoptimized_sdfg=None,
                )
