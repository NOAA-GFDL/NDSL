import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import dace
import gt4py.storage
from dace import compiletime as DaceCompiletime
from dace.dtypes import DeviceType as DaceDeviceType
from dace.dtypes import StorageType as DaceStorageType
from dace.frontend.python.common import SDFGConvertible
from dace.frontend.python.parser import DaceProgram
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.helpers import get_parent_map
from dace.transformation.passes.simplify import SimplifyPass

from ndsl.comm.mpi import MPI
from ndsl.dsl.dace.build import get_sdfg_path, write_build_info
from ndsl.dsl.dace.dace_config import (
    DEACTIVATE_DISTRIBUTED_DACE_COMPILE,
    DaceConfig,
    DaCeOrchestration,
    FrozenCompiledSDFG,
)
from ndsl.dsl.dace.sdfg_debug_passes import (
    negative_delp_checker,
    negative_qtracers_checker,
    sdfg_nan_checker,
)
from ndsl.dsl.dace.sdfg_opt_passes import splittable_region_expansion
from ndsl.dsl.dace.utils import (
    DaCeProgress,
    memory_static_analysis,
    report_memory_static_analysis,
)
from ndsl.logging import ndsl_log


try:
    import cupy as cp
except ImportError:
    cp = None


def dace_inhibitor(func: Callable) -> Callable:
    """Triggers callback generation wrapping `func` while doing DaCe parsing."""
    return func


def _upload_to_device(host_data: List[Any]) -> None:
    """Make sure any ndarrays gets uploaded to the device

    This will raise an assertion if cupy is not installed.
    """
    assert cp is not None
    for i, data in enumerate(host_data):
        if isinstance(data, cp.ndarray):
            host_data[i] = cp.asarray(data)


def _download_results_from_dace(
    config: DaceConfig, dace_result: Optional[List[Any]], args: List[Any]
):
    """Move all data from DaCe memory space to GT4Py"""
    if dace_result is None:
        return None

    backend = config.get_backend()
    return [gt4py.storage.from_array(result, backend=backend) for result in dace_result]


def _to_gpu(sdfg: dace.SDFG):
    """Flag memory in SDFG to GPU.
    Force deactivate OpenMP sections for sanity."""

    # Gather all maps
    allmaps = [
        (me, state)
        for me, state in sdfg.all_nodes_recursive()
        if isinstance(me, dace.nodes.MapEntry)
    ]
    topmaps = [
        (me, state) for me, state in allmaps if get_parent_map(state, me) is None
    ]

    # Set storage of arrays to GPU, scalarizable arrays will be set on registers
    for sd, _aname, arr in sdfg.arrays_recursive():
        if arr.shape == (1,):
            arr.storage = dace.StorageType.Register
        else:
            arr.storage = dace.StorageType.GPU_Global

    # All maps will be schedule on GPU
    for mapentry, _state in topmaps:
        mapentry.schedule = dace.ScheduleType.GPU_Device

    # Deactivate OpenMP sections
    for sd in sdfg.all_sdfgs_recursive():
        sd.openmp_sections = False


def _simplify(
    sdfg: dace.SDFG,
    *,
    validate: bool = True,
    validate_all: bool = False,
    verbose: bool = False,
):
    """Override of sdfg.simplify to skip failing transformation
    per https://github.com/spcl/dace/issues/1328
    """
    return SimplifyPass(
        validate=validate,
        validate_all=validate_all,
        verbose=verbose,
        skip=["ConstantPropagation"],
    ).apply_pass(sdfg, {})


def _build_sdfg(
    dace_program: DaceProgram, sdfg: dace.SDFG, config: DaceConfig, args, kwargs
):
    """Build the .so out of the SDFG on the top tile ranks only"""
    is_compiling = True if DEACTIVATE_DISTRIBUTED_DACE_COMPILE else config.do_compile

    if is_compiling:
        # Make the transients array persistents
        if config.is_gpu_backend():
            _to_gpu(sdfg)
            make_transients_persistent(sdfg=sdfg, device=DaceDeviceType.GPU)

            # Upload args to device
            _upload_to_device(list(args) + list(kwargs.values()))
        else:
            for _sd, _aname, arr in sdfg.arrays_recursive():
                if arr.shape == (1,):
                    arr.storage = DaceStorageType.Register
            make_transients_persistent(sdfg=sdfg, device=DaceDeviceType.CPU)

        # Build non-constants & non-transients from the sdfg_kwargs
        sdfg_kwargs = dace_program._create_sdfg_args(sdfg, args, kwargs)
        for k in dace_program.constant_args:
            if k in sdfg_kwargs:
                del sdfg_kwargs[k]
        sdfg_kwargs = {k: v for k, v in sdfg_kwargs.items() if v is not None}
        for k, tup in dace_program.resolver.closure_arrays.items():
            if k in sdfg_kwargs and tup[1].transient:
                del sdfg_kwargs[k]

        with DaCeProgress(config, "Simplify (1/2)"):
            _simplify(sdfg, validate=False, verbose=True)

        # Perform pre-expansion fine tuning
        with DaCeProgress(config, "Split regions"):
            splittable_region_expansion(sdfg, verbose=True)

        # Expand the stencil computation Library Nodes with the right expansion
        with DaCeProgress(config, "Expand"):
            sdfg.expand_library_nodes()

        with DaCeProgress(config, "Simplify (2/2)"):
            _simplify(sdfg, validate=False, verbose=True)

        # Move all memory that can be into a pool to lower memory pressure.
        # Change Persistent memory (sub-SDFG) into Scope and flag it.
        with DaCeProgress(config, "Turn Persistents into pooled Scope"):
            memory_pooled = 0.0
            for _sd, _aname, arr in sdfg.arrays_recursive():
                if arr.lifetime == dace.AllocationLifetime.Persistent:
                    arr.pool = True
                    memory_pooled += arr.total_size * arr.dtype.bytes
                    arr.lifetime = dace.AllocationLifetime.Scope
            memory_pooled = float(memory_pooled) / (1024 * 1024)
            ndsl_log.debug(
                f"{DaCeProgress.default_prefix(config)} Pooled {memory_pooled} mb",
            )

        # Set of debug tools inserted in the SDFG when dace.conf "syncdebug"
        # is turned on.
        if config.get_sync_debug():
            with DaCeProgress(config, "Tooling the SDFG for debug"):
                sdfg_nan_checker(sdfg)
                negative_delp_checker(sdfg)
                negative_qtracers_checker(sdfg)

        # Compile
        with DaCeProgress(config, "Codegen & compile"):
            sdfg.compile()
        write_build_info(sdfg, config.layout, config.tile_resolution, config._backend)

        # Printing analysis of the compiled SDFG
        with DaCeProgress(config, "Build finished. Running memory static analysis"):
            report = report_memory_static_analysis(
                sdfg, memory_static_analysis(sdfg), False
            )
            ndsl_log.info(f"{DaCeProgress.default_prefix(config)} {report}")

    # Compilation done.
    # On Build: all ranks sync, then exit.
    # On BuildAndRun: all ranks sync, then load the SDFG from
    #                 the expected path (made available by build).
    # We use a "FrozenCompiledSDFG" to minimize re-entry cost at call time

    mode = config.get_orchestrate()
    # DEV NOTE: we explicitly use MPI.COMM_WORLD here because it is
    # a true multi-machine sync, outside of our own communicator class.
    if mode == DaCeOrchestration.Build:
        MPI.COMM_WORLD.Barrier()  # Protect against early exist which kill SLURM jobs
        ndsl_log.info(f"{DaCeProgress.default_prefix(config)} Build only, exiting.")
        exit(0)

    if mode == DaCeOrchestration.BuildAndRun:
        if not is_compiling:
            ndsl_log.info(
                f"{DaCeProgress.default_prefix(config)} Rank is not compiling."
                "Waiting for compilation to end on all other ranks..."
            )
        MPI.COMM_WORLD.Barrier()

        with DaCeProgress(config, "Loading"):
            sdfg_path = get_sdfg_path(dace_program.name, config, override_run_only=True)
            compiledSDFG, _ = dace_program.load_precompiled_sdfg(
                sdfg_path, *args, **kwargs
            )
            config.loaded_precompiled_SDFG[dace_program] = FrozenCompiledSDFG(
                dace_program, compiledSDFG, args, kwargs
            )

        return _call_sdfg(dace_program, sdfg, config, args, kwargs)


def _call_sdfg(
    dace_program: DaceProgram, sdfg: dace.SDFG, config: DaceConfig, args, kwargs
):
    """Dispatch the SDFG execution and/or build"""
    # Pre-compiled SDFG code path does away with any data checks and
    # cached the marshalling - leading to almost direct C call
    # DaceProgram performs argument transformation & checks for a cost ~200ms
    # of overhead
    if dace_program in config.loaded_precompiled_SDFG:
        with DaCeProgress(config, "Run"):
            if config.is_gpu_backend():
                _upload_to_device(list(args) + list(kwargs.values()))
            res = config.loaded_precompiled_SDFG[dace_program]()
            res = _download_results_from_dace(
                config, res, list(args) + list(kwargs.values())
            )
        return res

    mode = config.get_orchestrate()
    if mode in [DaCeOrchestration.Build, DaCeOrchestration.BuildAndRun]:
        ndsl_log.info("Building DaCe orchestration")
        return _build_sdfg(dace_program, sdfg, config, args, kwargs)

    if mode == DaCeOrchestration.Run:
        # We should never hit this, it should be caught by the
        # loaded_precompiled_SDFG check above
        raise RuntimeError("Unexpected call - pre-compiled SDFG failed to load")
    else:
        raise NotImplementedError(f"Mode '{mode}' unimplemented at call time")


def _parse_sdfg(
    dace_program: DaceProgram,
    config: DaceConfig,
    *args,
    **kwargs,
) -> Optional[dace.SDFG]:
    """Return an SDFG depending on cache existence.
    Either parses, load a .sdfg or load .so (as a compiled sdfg)

    Attributes:
        dace_program: the DaceProgram carrying reference to the original method/function
        config: the DaceConfig configuration for this execution
    """
    # Check cache for already loaded SDFG
    if dace_program in config.loaded_precompiled_SDFG:
        return config.loaded_precompiled_SDFG[dace_program]

    # Build expected path
    sdfg_path = get_sdfg_path(dace_program.name, config)
    if sdfg_path is None:
        is_compiling = (
            True if DEACTIVATE_DISTRIBUTED_DACE_COMPILE else config.do_compile
        )

        if not is_compiling:
            # We can not parse the SDFG since we will load the proper
            # compiled SDFG from the compiling rank
            return None

        with DaCeProgress(config, f"Parse code of {dace_program.name} to SDFG"):
            sdfg = dace_program.to_sdfg(
                *args,
                **dace_program.__sdfg_closure__(),
                **kwargs,
                save=False,
                simplify=False,
            )
        return sdfg

    if os.path.isfile(sdfg_path):
        with DaCeProgress(config, "Load .sdfg"):
            sdfg, _ = dace_program.load_sdfg(sdfg_path, *args, **kwargs)
        return sdfg

    with DaCeProgress(config, "Load precompiled .sdfg (.so)"):
        compiledSDFG, _ = dace_program.load_precompiled_sdfg(sdfg_path, *args, **kwargs)
        config.loaded_precompiled_SDFG[dace_program] = FrozenCompiledSDFG(
            dace_program, compiledSDFG, args, kwargs
        )
    return compiledSDFG


class _LazyComputepathFunction(SDFGConvertible):
    """JIT wrapper around a function for DaCe orchestration.

    Attributes:
        func: function to either orchestrate or directly execute
        load_sdfg: folder path to a pre-compiled SDFG or file path to a .sdfg graph
                   that will be compiled but not regenerated.
    """

    def __init__(self, func: Callable, config: DaceConfig):
        self.func = func
        self.config = config
        self.daceprog: DaceProgram = dace.program(self.func)
        self._sdfg = None

    def __call__(self, *args, **kwargs):
        assert self.config.is_dace_orchestrated()
        sdfg = _parse_sdfg(
            self.daceprog,
            self.config,
            *args,
            **kwargs,
        )
        return _call_sdfg(
            self.daceprog,
            sdfg,
            self.config,
            args,
            kwargs,
        )

    @property
    def global_vars(self):
        return self.daceprog.global_vars

    @global_vars.setter
    def global_vars(self, value):
        self.daceprog.global_vars = value

    def __sdfg__(self, *args, **kwargs):
        return _parse_sdfg(self.daceprog, self.config, *args, **kwargs)

    def __sdfg_closure__(self, *args, **kwargs):
        return self.daceprog.__sdfg_closure__(*args, **kwargs)

    def __sdfg_signature__(self):
        return self.daceprog.argnames, self.daceprog.constant_args

    def closure_resolver(self, constant_args, given_args, parent_closure=None):
        return self.daceprog.closure_resolver(constant_args, given_args, parent_closure)


class _LazyComputepathMethod:
    """JIT wrapper around a class method for DaCe orchestration.

    Attributes:
        method: class method to either orchestrate or directly execute
        load_sdfg: folder path to a pre-compiled SDFG or file path to a .sdfg graph
                   that will be compiled but not regenerated.
    """

    # In order to not regenerate SDFG for the same obj.method callable
    # we cache the SDFGEnabledCallable we have already init
    bound_callables: Dict[Tuple[int, int], "SDFGEnabledCallable"] = dict()

    class SDFGEnabledCallable(SDFGConvertible):
        def __init__(self, lazy_method: "_LazyComputepathMethod", obj_to_bind):
            methodwrapper = dace.method(lazy_method.func)
            self.obj_to_bind = obj_to_bind
            self.lazy_method = lazy_method
            self.daceprog: DaceProgram = methodwrapper.__get__(obj_to_bind)

        @property
        def global_vars(self):
            return self.daceprog.global_vars

        @global_vars.setter
        def global_vars(self, value):
            self.daceprog.global_vars = value

        def __call__(self, *args, **kwargs):
            assert self.lazy_method.config.is_dace_orchestrated()
            sdfg = _parse_sdfg(
                self.daceprog,
                self.lazy_method.config,
                *args,
                **kwargs,
            )
            return _call_sdfg(
                self.daceprog,
                sdfg,
                self.lazy_method.config,
                args,
                kwargs,
            )

        def __sdfg__(self, *args, **kwargs):
            return _parse_sdfg(self.daceprog, self.lazy_method.config, *args, **kwargs)

        def __sdfg_closure__(self, reevaluate=None):
            return self.daceprog.__sdfg_closure__(reevaluate)

        def __sdfg_signature__(self):
            return self.daceprog.argnames, self.daceprog.constant_args

        def closure_resolver(self, constant_args, given_args, parent_closure=None):
            return self.daceprog.closure_resolver(
                constant_args, given_args, parent_closure
            )

    def __init__(self, func: Callable, config: DaceConfig):
        self.func = func
        self.config = config

    def __get__(self, obj, objtype=None) -> SDFGEnabledCallable:
        """Return SDFGEnabledCallable wrapping original obj.method from cache.
        Update cache first if need be"""
        if (id(obj), id(self.func)) not in _LazyComputepathMethod.bound_callables:
            _LazyComputepathMethod.bound_callables[
                (id(obj), id(self.func))
            ] = _LazyComputepathMethod.SDFGEnabledCallable(self, obj)

        return _LazyComputepathMethod.bound_callables[(id(obj), id(self.func))]


def orchestrate(
    *,
    obj: object,
    config: Optional[DaceConfig],
    method_to_orchestrate: str = "__call__",
    dace_compiletime_args: Optional[Sequence[str]] = None,
):
    """
    Orchestrate a method of an object with DaCe.
    The method object is patched in place, replacing the original Callable with
    a wrapper that will trigger orchestration at call time.
    If the model configuration doesn't demand orchestration, this won't do anything.

    Args:
        obj: object which methods is to be orchestrated
        config: DaceConfig carrying model configuration
        method_to_orchestrate: string representing the name of the method
        dace_compiletime_args: list of names of arguments to be flagged has
                               dace.compiletime for orchestration to behave
    """
    if config is None:
        raise ValueError("DaCe config cannot be None")

    if dace_compiletime_args is None:
        dace_compiletime_args = []

    if config.is_dace_orchestrated():
        if not hasattr(obj, method_to_orchestrate):
            raise RuntimeError(
                f"Could not orchestrate, "
                f"{type(obj).__name__}.{method_to_orchestrate} "
                "does not exists"
            )

        func = type.__getattribute__(type(obj), method_to_orchestrate)

        # Flag argument as dace.constant
        for argument in dace_compiletime_args:
            func.__annotations__[argument] = DaceCompiletime

        # Build DaCe orchestrated wrapper
        # This is a JIT object, e.g. DaCe compilation will happen on call
        wrapped = _LazyComputepathMethod(func, config).__get__(obj)

        if method_to_orchestrate == "__call__":
            # Grab the function from the type of the child class
            # Dev note: we need to use type for dunder call because:
            #   a = A()
            #   a()
            # resolved to: type(a).__call__(a)
            # therefore patching the instance call (e.g a.__call__) is not enough.
            # We could patch the type(self), ergo the class itself
            # but that would patch _every_ instance of A.
            # What we can do is patch the instance.__class__ with a local made class
            # in order to keep each instance with it's own patch.
            #
            # Re: type:ignore
            # Mypy is unhappy about dynamic class name and the devs (per github
            # issues discussion) is to make a plugin. Too much work -> ignore mypy

            class _(type(obj)):  # type: ignore
                __qualname__ = f"{type(obj).__qualname__}_patched"
                __name__ = f"{type(obj).__name__}_patched"

                def __call__(self, *arg, **kwarg):
                    return wrapped(*arg, **kwarg)

                def __sdfg__(self, *args, **kwargs):
                    return wrapped.__sdfg__(*args, **kwargs)

                def __sdfg_closure__(self, reevaluate=None):
                    return wrapped.__sdfg_closure__(reevaluate)

                def __sdfg_signature__(self):
                    return wrapped.__sdfg_signature__()

                def closure_resolver(
                    self, constant_args, given_args, parent_closure=None
                ):
                    return wrapped.closure_resolver(
                        constant_args, given_args, parent_closure
                    )

            # We keep the original class type name to not perturb
            # the workflows that uses it to build relevant info (path, hash...)
            previous_cls_name = type(obj).__name__
            obj.__class__ = _
            type(obj).__name__ = previous_cls_name
        else:
            # For regular attribute - we can just patch as usual
            setattr(obj, method_to_orchestrate, wrapped)


def orchestrate_function(
    config: DaceConfig = None,
    dace_compiletime_args: Optional[Sequence[str]] = None,
) -> Union[Callable[..., Any], _LazyComputepathFunction]:
    """
    Decorator orchestrating a method of an object with DaCe.
    If the model configuration doesn't demand orchestration, this won't do anything.

    Args:
        config: DaceConfig carrying model configuration
        dace_compiletime_args: list of names of arguments to be flagged has
                               dace.compiletime for orchestration to behave
    """

    if dace_compiletime_args is None:
        dace_compiletime_args = []

    def _decorator(func: Callable[..., Any]):
        def _wrapper(*args, **kwargs):
            for argument in dace_compiletime_args:
                func.__annotations__[argument] = DaceCompiletime
            return _LazyComputepathFunction(func, config)

        return _wrapper(func) if config.is_dace_orchestrated() else func

    return _decorator
