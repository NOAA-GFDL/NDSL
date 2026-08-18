from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from dace import SDFG
from dace import compiletime as DaceCompiletime
from dace import method as dace_method
from dace import program as dace_program_wrapper
from dace.frontend.python.common import SDFGConvertible
from dace.frontend.python.parser import DaceProgram
from dace.sdfg.analysis.schedule_tree import treenodes as tn

import ndsl.dsl.dace.replacements  # noqa # We load in the DaCe replacements
from ndsl import OptimizationConfig
from ndsl.dsl.dace.builder import (
    get_dace_executable,
    optimize_full_program_sdfg,
    parse_sdfg,
)
from ndsl.dsl.dace.dace_config import (
    DaceConfig,
    DaCeOrchestration,
)
from ndsl.dsl.dace.dace_executable import DACE_EXECUTABLE_CACHE, DaceExecutable
from ndsl.dsl.dace.labeler import set_label
from ndsl.quantity import Quantity, State

_INTERNAL__SCHEDULE_TREE_OPTIMIZATION_PASSES: list[tn.ScheduleNodeVisitor] | None = None


def dace_inhibitor(func: Callable) -> Callable:
    """Triggers callback generation wrapping `func` while doing DaCe parsing."""
    return func


def _call_sdfg(
    dace_program: DaceProgram,
    sdfg: SDFG,
    config: DaceConfig,
    optimization_config: OptimizationConfig | None,
    args: Any,
    kwargs: Any,
) -> DaceExecutable:
    """Dispatch to either SDFG execution and/or build."""

    mode = config.get_orchestrate()
    if (
        mode in [DaCeOrchestration.Build, DaCeOrchestration.BuildAndRun]
        and dace_program not in DACE_EXECUTABLE_CACHE  # already cached
    ):
        optimize_full_program_sdfg(
            dace_program, sdfg, config, optimization_config, args, kwargs
        )

    if dace_program not in DACE_EXECUTABLE_CACHE:
        raise RuntimeError(
            "Dace program not found in cache. Are you running `DaCeOrchestration.Run` "
            "without a pre-filled cache folder? Try `DacCeOrchestration.BuildAndRun` instead."
        )

    return DACE_EXECUTABLE_CACHE[dace_program].run(dace_program, args, kwargs)


class _LazyComputepathFunction(SDFGConvertible):
    """JIT wrapper around a function for DaCe orchestration.

    Attributes:
        func: function to either orchestrate or directly execute
        load_sdfg: folder path to a pre-compiled SDFG or file path to a .sdfg graph
                   that will be compiled but not regenerated.
    """

    def __init__(
        self,
        func: Callable,
        config: DaceConfig,
        optimization_config: OptimizationConfig | None,
    ) -> None:
        self.func = func
        self.config = config
        self.optimization_config = optimization_config
        self.daceprog: DaceProgram = dace_program_wrapper(self.func)
        self._sdfg = None

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        assert self.config.is_dace_orchestrated()
        sdfg = parse_sdfg(
            self.daceprog,
            self.config,
            self.optimization_config,
            *args,
            **kwargs,
        )
        exe = get_dace_executable(
            self.daceprog,
            sdfg,
            self.config,
            self.optimization_config,
            args,
            kwargs,
        )
        return exe.run(self.daceprog, args, kwargs)

    @property
    def global_vars(self):  # type: ignore[no-untyped-def]
        return self.daceprog.global_vars

    @global_vars.setter
    def global_vars(self, value):  # type: ignore[no-untyped-def]
        self.daceprog.global_vars = value

    def __sdfg__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return parse_sdfg(
            self.daceprog, self.config, self.optimization_config, *args, **kwargs
        )

    def __sdfg_closure__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.daceprog.__sdfg_closure__(*args, **kwargs)

    def __sdfg_signature__(self):  # type: ignore[no-untyped-def]
        return self.daceprog.argnames, self.daceprog.constant_args

    def closure_resolver(self, constant_args, given_args, parent_closure=None):  # type: ignore[no-untyped-def]
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
    bound_callables: dict[tuple[int, int], SDFGEnabledCallable] = dict()

    class SDFGEnabledCallable(SDFGConvertible):
        def __init__(
            self, lazy_method: _LazyComputepathMethod, obj_to_bind: object
        ) -> None:
            methodwrapper = dace_method(lazy_method.func)
            self.obj_to_bind = obj_to_bind
            self.lazy_method = lazy_method
            self.daceprog: DaceProgram = methodwrapper.__get__(obj_to_bind)

        @property
        def global_vars(self):  # type: ignore[no-untyped-def]
            return self.daceprog.global_vars

        @global_vars.setter
        def global_vars(self, value):  # type: ignore[no-untyped-def]
            self.daceprog.global_vars = value

        def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            assert self.lazy_method.config.is_dace_orchestrated()
            sdfg = parse_sdfg(
                self.daceprog,
                self.lazy_method.config,
                self.lazy_method.optimization_config,
                *args,
                **kwargs,
            )
            exe = get_dace_executable(
                self.daceprog,
                sdfg,
                self.lazy_method.config,
                self.lazy_method.optimization_config,
                args,
                kwargs,
            )
            return exe.run(self.daceprog, args, kwargs)

        def __sdfg__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            sdfg = parse_sdfg(
                self.daceprog,
                self.lazy_method.config,
                self.lazy_method.optimization_config,
                *args,
                **kwargs,
            )
            # Label the code
            if (
                sdfg is not None
                and self.lazy_method.optimization_config is not None
                and self.lazy_method.optimization_config.stree.enabled
            ):
                set_label(
                    sdfg,
                    type(self.obj_to_bind).__qualname__,
                    is_top_sdfg=False,
                    local_optimizations=self.lazy_method.optimization_config,
                )
            return sdfg

        def __sdfg_closure__(self, reevaluate=None):  # type: ignore[no-untyped-def]
            return self.daceprog.__sdfg_closure__(reevaluate)

        def __sdfg_signature__(self):  # type: ignore[no-untyped-def]
            return self.daceprog.argnames, self.daceprog.constant_args

        def closure_resolver(self, constant_args, given_args, parent_closure=None):  # type: ignore[no-untyped-def]
            return self.daceprog.closure_resolver(
                constant_args, given_args, parent_closure
            )

    def __init__(
        self,
        func: Callable,
        config: DaceConfig,
        optimization_config: OptimizationConfig | None,
    ) -> None:
        self.func = func
        self.config = config
        self.optimization_config = optimization_config

    def __get__(self, obj: object, objtype: Any = None) -> SDFGEnabledCallable:
        """Return SDFGEnabledCallable wrapping original obj.method from cache.
        Update cache first if need be"""
        if (id(obj), id(self.func)) not in _LazyComputepathMethod.bound_callables:
            _LazyComputepathMethod.bound_callables[(id(obj), id(self.func))] = (
                _LazyComputepathMethod.SDFGEnabledCallable(self, obj)
            )

        return _LazyComputepathMethod.bound_callables[(id(obj), id(self.func))]


def orchestrate(
    *,
    obj: object,
    config: DaceConfig,
    method_to_orchestrate: str = "__call__",
    dace_compiletime_args: Sequence[str] | None = None,
    optimization_config: OptimizationConfig | None = None,
) -> None:
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
    if hasattr(obj, "_ndsl_orchestrated_methods"):
        # Automatically register all orchestrated methods of NDSLRuntime classes
        # to track where Locals can be used.
        # See __post_init__() of NDSLRuntime.
        obj._ndsl_orchestrated_methods.append(method_to_orchestrate)

    if config is None:
        raise ValueError("DaCe config cannot be None")

    if not config.is_dace_orchestrated():
        return

    # We have to un-monkey patch the __call__ (from the debugger)
    if method_to_orchestrate == "__call__" and hasattr(type(obj), "_original__call__"):
        type(obj).__call__ = type(obj)._original__call__  # type: ignore[method-assign,attr-defined]

    if not hasattr(obj, method_to_orchestrate):
        raise RuntimeError(
            f"Could not orchestrate, "
            f"{type(obj).__name__}.{method_to_orchestrate} "
            "does not exist."
        )

    if dace_compiletime_args is None:
        dace_compiletime_args = []

    func: Callable = type.__getattribute__(type(obj), method_to_orchestrate)

    # Flag argument as dace.constant
    for argument in dace_compiletime_args:
        func.__annotations__[argument] = DaceCompiletime

    # Swap State and subclass into compile time
    for arg_name, annotation in func.__annotations__.items():
        if annotation in [State] or (
            isinstance(annotation, type) and issubclass(annotation, State)
        ):
            func.__annotations__[arg_name] = DaceCompiletime

    # Remove type hint of Quantity to allow for __descriptor__ to be read in JIT
    for arg_name, annotation in func.__annotations__.items():
        if annotation in [Quantity] or (
            isinstance(annotation, type) and issubclass(annotation, Quantity)
        ):
            func.__annotations__[arg_name] = None

    # Build DaCe orchestrated wrapper
    # This is a JIT object, e.g. DaCe compilation will happen on call
    wrapped = _LazyComputepathMethod(func, config, optimization_config).__get__(obj)

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

            def __call__(self, *arg, **kwarg):  # type: ignore[no-untyped-def]
                return wrapped(*arg, **kwarg)

            def __sdfg__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return wrapped.__sdfg__(*args, **kwargs)

            def __sdfg_closure__(self, reevaluate=None):  # type: ignore[no-untyped-def]
                return wrapped.__sdfg_closure__(reevaluate)

            def __sdfg_signature__(self):  # type: ignore[no-untyped-def]
                return wrapped.__sdfg_signature__()

            def closure_resolver(self, constant_args, given_args, parent_closure=None):  # type: ignore[no-untyped-def]
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
    config: DaceConfig,
    dace_compiletime_args: Sequence[str] | None = None,
    optimization_config: OptimizationConfig | None = None,
) -> Callable[..., Any] | _LazyComputepathFunction:
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

    def _decorator(func: Callable[..., Any]):  # type: ignore[no-untyped-def]
        def _wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            for argument in dace_compiletime_args:
                func.__annotations__[argument] = DaceCompiletime
            return _LazyComputepathFunction(func, config, optimization_config)

        return _wrapper(func) if config.is_dace_orchestrated() else func

    return _decorator
