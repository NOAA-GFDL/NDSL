from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import Float
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Local, Quantity


_TOP_LEVEL: object | None = None


class NDSLRuntime:
    """Base class to tool runtime code, allows use of Locals, orchestration and
    debug tools.

    The __call__ function will automatically be orchestrated."""

    def __init__(self, stencil_factory: StencilFactory) -> None:
        self._stencil_factory = stencil_factory
        # Use this flag to detect that the init wasn't done properly
        self._base_class_was_properly_super_init = True

    def __init_subclass__(cls: type[NDSLRuntime], **kwargs: dict[str, Any]) -> None:
        # WARNING: no code outside the `init_decorator` this is cls
        # function, it will be called ONLY ONCE for monkey-patching the
        # Class - not the instance !

        def init_decorator(previous_init: Callable) -> Callable:
            def new_init(
                self: NDSLRuntime,
                *args: list[Any],
                **kwargs: dict[str, Any],
            ) -> None:
                global _TOP_LEVEL
                if _TOP_LEVEL is None:
                    _TOP_LEVEL = self
                previous_init(self, *args, **kwargs)
                self.__post_init__()

            return new_init

        cls.__init__ = init_decorator(cls.__init__)  # type: ignore[method-assign]

    def __post_init__(self) -> None:
        if not hasattr(self, "_base_class_was_properly_super_init"):
            raise RuntimeError(
                f"Class {type(self).__name__} inherit from NDSLRuntime but didn't call super().__init__."
            )

        # Check quantity allocation of NDSLRuntime supervised code
        if _TOP_LEVEL == self:

            def check_for_quantity(object_: object) -> None:
                for key, value in object_.__dict__.items():
                    if isinstance(value, Quantity) and not isinstance(value, Local):
                        warnings.warn(
                            f"{type(self).__name__}.{key} is a Quantity instead of a Locals"
                            " on a NDSLRuntime - our eyebrows are frowned.",
                            UserWarning,
                            stacklevel=2,
                        )
                    elif isinstance(value, NDSLRuntime):
                        check_for_quantity(value)

            check_for_quantity(self)

        # Orchestrate __call__ by default
        if callable(self):
            orchestrate(
                obj=self,
                config=self._stencil_factory.config.dace_config,
            )

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        # We look at the direct caller frame for our own `self`
        # in the locals.
        # All other cases are forbidden.
        if isinstance(attr, Local):
            frame = inspect.currentframe()
            if frame is None:
                raise NotImplementedError(
                    "Locals check cannot locate frame. Talk to the team."
                )
            caller_frame = frame.f_back
            if (
                not caller_frame
                or "self" not in caller_frame.f_locals
                or not isinstance(caller_frame.f_locals["self"], type(self))
            ):
                # We expect the original class to have been monkey-patched
                # See `dace.dsl.orchestration.orchestrate`
                class_name = type(self).__name__
                raise RuntimeError(
                    f"Forbidden Local access: {name} called outside of {class_name}."
                )

        return attr

    def make_local(
        self,
        quantity_factory: QuantityFactory,
        dims: list[str],
        dtype: type = Float,
        units: str = "unspecified",
        *,
        allow_mismatch_float_precision: bool = False,
    ) -> Local:
        quantity = quantity_factory.zeros(
            dims,
            units,
            dtype,
            allow_mismatch_float_precision=allow_mismatch_float_precision,
        )
        return Local(
            data=quantity.data,
            dims=quantity.dims,
            units=quantity.units,
            origin=quantity.origin,
            extent=quantity.extent,
            backend=quantity.backend,
            allow_mismatch_float_precision=allow_mismatch_float_precision,
        )
