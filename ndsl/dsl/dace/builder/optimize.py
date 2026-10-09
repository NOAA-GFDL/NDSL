import gc
import importlib
import numbers
import os
from pathlib import Path
from pprint import pformat
from typing import Any

from dace import SDFG, DeviceType, dtypes, nodes
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.dtypes import DeviceType as DaceDeviceType
from dace.dtypes import ScheduleType
from dace.dtypes import StorageType as DaceStorageType
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.utils import fuse_states, inline_sdfgs
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.dataflow import MapCollapse, MapExpansion
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
from dace.transformation.helpers import get_parent_map

from ndsl import Backend, OptimizationConfig, ndsl_log
from ndsl.dsl.dace.builder.cache import BuildInfo
from ndsl.dsl.dace.builder.sdfg.debug_passes import (
    negative_delp_checker,
    negative_qtracers_checker,
    sdfg_nan_checker,
)
from ndsl.dsl.dace.builder.stree import CPUPipeline, GPUPipeline, StreePipeline
from ndsl.dsl.dace.dace_config import DaceConfig
from ndsl.dsl.dace.hardware_config import get_gpu_hardware_defaults
from ndsl.dsl.dace.utils import (
    DaCeProgress,
    memory_static_analysis,
    report_memory_static_analysis,
    upload_to_device,
)

_INTERNAL__SCHEDULE_TREE_OPTIMIZATION_PASSES: list[tn.ScheduleNodeVisitor] | None = None


def _dynamic_transform_load(module_name: str, class_name: str) -> type | None:
    """Safely load a class from a module specified by string."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError):
        ndsl_log.debug(f"Transform {class_name} not available")
        return None


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
        # We disable LiftTrivialIf because it takes long on bigger graphs and we estimate
        # the potential speed gains to be minimal anyway.
        skip={"ScalarToSymbolPromotion", "LiftTrivialIf"},
    )


def _tree_as_sdfg(stree: tn.ScheduleTreeRoot) -> SDFG:
    """
    Convert the given ScheduleTree to SDFG.

    This function wraps `stree.as_sdfg()` with a configuration that is suitable for
    NDSL, e.g. skipping certain passes of `sdfg.simplify()`.
    """
    # We disable ScalarToSymbolPromotion because it might push symbols onto edges
    # that DaCe itself can't parse anymore later, e.g. casts,  inlined function
    # calls or (complicated) field accesses.
    # We disable ControlFlowRaising because tree -> sdfg outputs control flow graphs.
    # We disable LiftTrivialIf because it takes long on bigger graphs and we estimate
    # the potential speed gains to be minimal anyway.
    return stree.as_sdfg(
        validate=False,
        simplify=False,  # D_SW failed validation on merging
        skip={"ScalarToSymbolPromotion", "ControlFlowRaising", "LiftTrivialIf"},
    )


def _V2_optimize_for_serial_cpu(config: DaceConfig, parsed_sdfg: SDFG) -> SDFG:
    mode = config.get_orchestrate()
    with DaCeProgress(mode, "Schedule Tree: generate from SDFG"):
        stree = parsed_sdfg.as_schedule_tree()

    with DaCeProgress(
        mode,
        "Schedule Tree: cleanup, vectorization friendly pass, transient massaging",
    ):
        PIPELINE = [
            "convert_map_to_loop",
            "fuse_loops_for_reuse",
            "merge_consecutive_loops",
            "pair_complementary_guards",
            "forward_substitute_conditions",
            "remove_dead_assignments",
            "fold_guards",
            "unswitch_invariant_guards",
            "split_iteration_spaces",
            "merge_consecutive_loops",
            "convert_diamonds_to_selects",
            "remove_dead_stores",
            "merge_contiguous_loops",
            "reroll_statements",
            "fuse_rolled_loops",
            "hoist_select_arms",
            "merge_consecutive_loops",
            "hoist_condition_reads",
            "refine_loop_local_transients",
            "reuse_transients",
            "move_small_transients_to_stack",
        ]
        for step in PIPELINE:
            trf = _dynamic_transform_load(
                "dace.sdfg.analysis.schedule_tree.passes", step
            )
            if trf:
                r = trf(stree)
                ndsl_log.debug(f"{trf.__name__}: {r}")

    with DaCeProgress(mode, "Schedule Tree: go back to SDFG"):
        result = _tree_as_sdfg(stree)

    with DaCeProgress(mode, "Replace Bool array by Int array (vectorization)"):
        try:
            from dace.transformation.passes import PredicateToIntegerArray

            PredicateToIntegerArray().apply_pass(result, {})
        except ModuleNotFoundError:
            ndsl_log.debug("The experimental PredicateToIntegerArray is not available")

    return result


def _optimize_for_serial_cpu(
    config: DaceConfig, sdfg: SDFG, validate: bool = True, verbose: bool = True
) -> SDFG:
    """
    Applies the schedule-tree passes to a (frozen) dynamical-core SDFG and returns the resulting SDFG.

    :param sdfg: The SDFG to optimize (left unchanged).
    :param verbose: Print how many changes each pass made.
    :return: The optimized SDFG.
    """
    import dace.sdfg.analysis.schedule_tree.passes as passes

    ndsl_log.debug("Going to tree...")
    stree = sdfg.as_schedule_tree()
    pipeline = [
        # Turn all maps to loops
        ("toloops", passes.convert_map_to_loop),
        # Temporaries: fuse vertical loops that share them, then shrink them (before splitting separates iterations)
        (
            "fusion",
            lambda t: passes.fuse_loops_for_reuse(
                t, cache_bytes=256 * 1024, min_trip_count=2
            ),
        ),
        ("refine", passes.refine_loop_local_transients),
        # Guard cleanup
        ("pair", passes.pair_complementary_guards),
        ("forward", passes.forward_substitute_conditions),
        ("deadassign", passes.remove_dead_assignments),
        ("fold", passes.fold_guards),
        ("unswitch", passes.unswitch_invariant_guards),
        ("split", passes.split_iteration_spaces),
        ("selects", passes.convert_diamonds_to_selects),
        ("deadstores", passes.remove_dead_stores),
        ("merge", passes.merge_contiguous_loops),
        ("reroll", passes.reroll_statements),
        ("fuserolled", passes.fuse_rolled_loops),
        ("hoist", passes.hoist_select_arms),
        # Rarely true guards (before hoist_condition_reads, which adds unguarded statements before conditions)
        ("sink", passes.sink_into_guards),
        (
            "move",
            passes.move_statements_to_readers,
        ),  # Past nests that keep statements from their readers" guards
        ("sink2", passes.sink_into_guards),
        ("coarsen", passes.coarsen_guards),
        # Skip resets of containers nothing wrote since they were last reset (before reuse, which would share them)
        ("resets", passes.skip_redundant_resets),
        # Conditions read transient arrays through memlets, so that the memory passes see those reads
        ("hoistcond", passes.hoist_condition_reads),
        # Memory: share slots between temporaries, then put small ones on the stack
        ("reuse", passes.reuse_transients),
        (
            "stack",
            lambda t: passes.move_small_transients_to_stack(
                t,
                max_array_bytes=4096,
                max_total_bytes=128 * 1024,
                zero_read_before_written=True,
            ),
        ),
    ]
    has_converged = False
    while not has_converged:
        ndsl_log.debug("Apply passes...")
        results = {}
        for name, apply in pipeline:
            results[name] = apply(stree)
        has_converged = all(results.values()) == 0
        if verbose:
            for name, count in results.items():
                ndsl_log.debug(f"{name}: {count}")
    result = stree.as_sdfg(validate=False, simplify=False)
    from dace.transformation.passes import PredicateToIntegerArray

    PredicateToIntegerArray().apply_pass(
        result, {}
    )  # Boolean mask arrays become int32 (vectorizes better)
    if validate:
        result.validate()
    return result


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
    parsed_sdfg: SDFG,
    config: DaceConfig,
    optimization_config: OptimizationConfig | None,
    args: Any,
    kwargs: Any,
) -> CompiledSDFG:
    """Optimize and compile the SDFG (creating the .dacecache and with the source and dynamic library)
    from a parsed SDFG (e.g. python code + gt4py stencils).
    """

    device_type = DaceDeviceType.GPU if config.is_gpu_backend() else DaceDeviceType.CPU
    mode = config.get_orchestrate()

    # Enforce cache directory made so all downstream caching file
    # won't hit an non existing directory
    Path(parsed_sdfg.build_folder).mkdir(parents=True, exist_ok=True)

    if optimization_config is None:
        ndsl_log.debug(f"Using default optimization config for {parsed_sdfg.label}.")
        optimization_config = OptimizationConfig.get_default()
    optimization_config.concretize(config.get_backend())
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
            ndsl_log.debug("Saving parsed_sdfg.sdfgz")
            parsed_sdfg.save(
                os.path.abspath(f"{parsed_sdfg.build_folder}/parsed_sdfg.sdfgz"),
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

    with DaCeProgress(mode, "Simplify (1)"):
        _simplify(parsed_sdfg)
        if config.verbose_orchestration:
            ndsl_log.debug("saving 01-simplify.sdfgz")
            parsed_sdfg.save(
                os.path.abspath(f"{parsed_sdfg.build_folder}/01-simplify_1.sdfgz"),
                compress=True,
            )

    with DaCeProgress(mode, "Expand maps"):
        parsed_sdfg.apply_transformations_repeated(
            [MapExpansion],
            validate=False,
            print_report=False,
        )

    if optimization_config.stree.enabled:
        with DaCeProgress(mode, "Expand maps (pre tree conversion)"):
            parsed_sdfg.apply_transformations_repeated(
                MapExpansion,
                options={
                    "inner_schedule": (
                        ScheduleType.GPU_Device
                        if device_type is DeviceType.GPU
                        else ScheduleType.Default
                    )
                },
                validate=False,
            )
        # Here be 🐉 - but tests exists in test_optimization.py
        with DaCeProgress(mode, "Schedule Tree: generate from SDFG"):
            # Break all loops into uni-dimensional loops to simplify optimizations
            stree = parsed_sdfg.as_schedule_tree()

        with DaCeProgress(mode, "Schedule Tree: optimization"):
            pipeline = _optimization_pipeline(
                optimization_config,
                device_type,
                config.get_backend(),
                cache_directory=Path(parsed_sdfg.build_folder),
                passes=_INTERNAL__SCHEDULE_TREE_OPTIMIZATION_PASSES,
            )
            pipeline.run(stree, verbose=config.verbose_schedule_tree_optimizations)

        with DaCeProgress(mode, "Schedule Tree: go back to SDFG"):
            parsed_sdfg = _tree_as_sdfg(stree)
            if config.verbose_orchestration:
                ndsl_log.debug("saving 04-from_stree.sdfgz")
                parsed_sdfg.save(
                    os.path.abspath(f"{parsed_sdfg.build_folder}/04-from_stree.sdfgz"),
                    compress=True,
                )

        with DaCeProgress(mode, "Simplify (post-stree conversion)"):
            _simplify(parsed_sdfg)

    if optimization_config.loop_vectorization:
        with DaCeProgress(mode, "Serial CPU optimization V3"):
            parsed_sdfg = _optimize_for_serial_cpu(
                config, parsed_sdfg, validate=False, verbose=True
            )

        # with DaCeProgress(mode, "Serial CPU optimization V2"):
        #     parsed_sdfg = _V2_optimize_for_serial_cpu(config, parsed_sdfg)

        with DaCeProgress(mode, "Inline SDFGs & Fuse states"):
            ctr_sdfg = inline_sdfgs(parsed_sdfg)
            ctr_state = fuse_states(parsed_sdfg)
            ndsl_log.debug(
                f"Inline SDFGs | Fuse states counts: {ctr_sdfg} | {ctr_state}"
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
            upload_to_device(list(args) + list(kwargs.values()))
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
            exclude_tasklets_list = []

            for me, _state in parsed_sdfg.all_nodes_recursive():
                if (
                    isinstance(me, nodes.MapEntry)
                    and me.map.schedule == ScheduleType.GPU_Device
                ) and me.map.gpu_block_size is None:
                    me.map.gpu_block_size = gpu_defaults.block_size

                if isinstance(me, nodes.Tasklet) and "callback_" in me.label:
                    exclude_tasklets_list.append(me.label)

            parsed_sdfg.apply_transformations_repeated(
                AddThreadBlockMap, print_report=False, validate=False
            )

            if optimization_config.gpu.common_gpu_xforms:
                with DaCeProgress(mode, "Apply common GPU xforms"):
                    # Apply common GPU transforms (includes a simplify)
                    # while making sure tasklet remain on the host
                    from dace.transformation.interstate import GPUTransformSDFG

                    parsed_sdfg.apply_transformations(
                        GPUTransformSDFG,
                        options={
                            "exclude_tasklets": ",".join(exclude_tasklets_list),
                            "host_data": ["__pystate"],
                        },
                        validate=False,
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
        ndsl_log.debug(
            "Stats on SDFG before compilation:\n"
            f"  Arrays: {len([name for _, name, _ in parsed_sdfg.arrays_recursive(include_nested_data=True)])}\n"
            f"  SDFGs : {len([sdfg for sdfg in parsed_sdfg.all_sdfgs_recursive()])}\n"
            f"  States: {len(parsed_sdfg.states())}\n"
            f"  CFGs  : {len([cfg for cfg in parsed_sdfg.all_control_flow_regions(recursive=True)])}\n"
        )
        parsed_sdfg.save("pre-compile.sdfgz", compress=True)
        compiled_sdfg = parsed_sdfg.compile()

    # Printing analysis of the compiled SDFG
    with DaCeProgress(mode, "Build finished. Running memory static analysis"):
        report = report_memory_static_analysis(
            parsed_sdfg, memory_static_analysis(parsed_sdfg), False
        )
        ndsl_log.info(f"{DaCeProgress.default_prefix(mode)} {report}")

    # Store build info in the common cache directory
    BuildInfo.save(
        parsed_sdfg, config.layout, config.tile_resolution, report, config.get_backend()
    )

    # The full process of orchestration and compile is very memory hungry, since we are about to
    # begin runtime execution in earnest, let's give python a beat to clean up
    gc.collect(2)

    return compiled_sdfg
