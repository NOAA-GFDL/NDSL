from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Hashable, Self, TypeAlias

import dacite
import xarray as xr
from numpy.typing import ArrayLike

from ndsl.comm.mpi import MPI
from ndsl.types import Number


from ndsl.quantity import Quantity, Local  # isort:skip


if TYPE_CHECKING:
    from ndsl import QuantityFactory

import warnings


StateMemoryMapping: TypeAlias = dict[str, dict | ArrayLike | None]
OptionalQuantityType: TypeAlias = Quantity | None
StateElementType: TypeAlias = dict[
    str, Quantity | OptionalQuantityType | Local | dict[str, Any]
]


@dataclasses.dataclass
class State:
    """Base class for state objects in models.

    A State groups a collection of (possibly nested) Quantities in a dataclass.

    This baseclass implements common initialization functions and serialization.

    Typical usage example:

    ```python
        class MyState(State):
            pass

        my_state = MyState.zeros(quantity_factory)

        # ...

        my_state.to_netcdf()
    ```
    """

    @classmethod
    def _init(
        cls,
        quantity_factory_allocator: Callable,
        *,
        type_check: bool = True,
    ) -> Self:
        """Allocate memory and init with a blind quantity init operation

        Args:
            quantity_factory_allocator: the allocator function from
                a quantity_factory (zeros, ones, empty)
            type_check: have strict type checking when running dacite. This flag
                exists to allow overriding types on dataclasses for Locals
        """

        def _init_recursive(cls: Any) -> StateElementType:
            initial_quantities: StateElementType = {}
            for _field in dataclasses.fields(cls):
                if not _field.init:
                    continue

                if dataclasses.is_dataclass(_field.type):
                    initial_quantities[_field.name] = _init_recursive(_field.type)
                elif _field.type in [Quantity, OptionalQuantityType, Local]:
                    if "dims" not in _field.metadata.keys():
                        raise ValueError(
                            f"Malformed state - no dims to init {_field.name} of type {_field.type}"
                        )

                    allow_mismatch_float_precision = True
                    quantity = quantity_factory_allocator(
                        _field.metadata["dims"],
                        _field.metadata["units"],
                        dtype=_field.metadata["dtype"],
                        allow_mismatch_float_precision=allow_mismatch_float_precision,
                    )
                    if _field.type == Local:
                        local_ = Local(
                            data=quantity.data,
                            dims=quantity.dims,
                            units=quantity.units,
                            origin=quantity.origin,
                            extent=quantity.extent,
                            backend=quantity.backend,
                            allow_mismatch_float_precision=allow_mismatch_float_precision,
                        )
                        initial_quantities[_field.name] = local_
                    else:
                        initial_quantities[_field.name] = quantity
                else:
                    raise TypeError(
                        "State attributes needs to be Quantity, Scalar or nested dataclasses "
                        f"got {_field.type} for {_field.name}."
                    )

            return initial_quantities

        dict_of_quantities = _init_recursive(cls)
        return dacite.from_dict(
            data_class=cls,
            data=dict_of_quantities,
            config=dacite.Config(check_types=type_check),
        )

    def __post_init__(self) -> None:
        def _flag_optional_recursive(cls: Any) -> None:
            for _field in dataclasses.fields(cls):
                if dataclasses.is_dataclass(_field.type):
                    _flag_optional_recursive(_field.type)
                elif _field.type == OptionalQuantityType:
                    self.optional_quantities[_field.name] = True
                else:
                    self.optional_quantities[_field.name] = False

        self.optional_quantities: dict[str, bool] = {}
        _flag_optional_recursive(type(self))

    class _FactorySwapDimensionsDefinitions:
        """INTERNAL: QuantityFactory carry a sizer which has a full definition of the dimensions.
        It's this sizer that is leveraged for the factory to figure out allocations.
        In a regular pattern of use, data dimensions fields tend to be _the exception_ rather
        than the rule and therefore would need a Factory defined _for a few cases_.
        We bring this tool to override temporarily the allocations based on a single descriptions of
        the data dimensions at allocation time.
        """

        def __init__(self, factory: QuantityFactory, ddims: dict[str, int]):
            self._ddims = ddims
            self._factory = factory

        def __enter__(self) -> None:
            self._original_dims = self._factory.sizer.data_dimensions
            self._factory.sizer.data_dimensions = self._ddims

        def __exit__(
            self,
            type: type[BaseException] | None,
            value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            self._factory.sizer.data_dimensions = self._original_dims

    @classmethod
    def _check_no_locals(cls) -> None:
        def _check_no_locals_recursive(cls: Any) -> None:
            for _field in dataclasses.fields(cls):
                if dataclasses.is_dataclass(_field.type):
                    _check_no_locals_recursive(_field.type)
                elif _field.type == Local:
                    raise TypeError(
                        f"State contains Local {_field.name}, you need to allocate using `make_local`. "
                        "State with Locals can _only_ contain Locals."
                    )

        _check_no_locals_recursive(cls)

    @classmethod
    def empty(
        cls,
        quantity_factory: QuantityFactory,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        """Allocate all quantities. Do not expect 0 on values, values are random.

        Args:
            quantity_factory: factory, expected to be defined on the Grid dimensions
                e.g. without data dimensions.
            data_dimensions: extra data dimensions required for any field with data dimensions.
                Dict of name/size pair.
        """
        cls._check_no_locals()

        if data_dimensions is None:
            data_dimensions = {}

        with State._FactorySwapDimensionsDefinitions(quantity_factory, data_dimensions):
            state = cls._init(quantity_factory.empty)
        return state

    @classmethod
    def zeros(
        cls,
        quantity_factory: QuantityFactory,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        """Allocate all quantities and fill their value to zeros

        Args:
            quantity_factory: factory, expected to be defined on the Grid dimensions
                e.g. without data dimensions.
            data_dimensions: extra data dimensions required for any field with data dimensions.
                Dict of name/size pair.
        """
        cls._check_no_locals()

        if data_dimensions is None:
            data_dimensions = {}

        with State._FactorySwapDimensionsDefinitions(quantity_factory, data_dimensions):
            state = cls._init(quantity_factory.zeros)
        return state

    @classmethod
    def ones(
        cls,
        quantity_factory: QuantityFactory,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        """Allocate all quantities and fill their value to ones

        Args:
            quantity_factory: factory, expected to be defined on the Grid dimensions
                e.g. without data dimensions.
            data_dimensions: extra data dimensions required for any field with data dimensions.
                Dict of name/size pair.
        """
        cls._check_no_locals()

        if data_dimensions is None:
            data_dimensions = {}

        with State._FactorySwapDimensionsDefinitions(quantity_factory, data_dimensions):
            state = cls._init(quantity_factory.ones)
        return state

    @classmethod
    def full(
        cls,
        quantity_factory: QuantityFactory,
        value: Number,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        """Allocate all quantities and fill them with the input value

        Args:
            quantity_factory: factory, expected to be defined on the Grid dimensions
                e.g. without data dimensions.
            value: number to initialize the buffers with.
            data_dimensions: extra data dimensions required for any field with data dimensions.
                Dict of name/size pair.
        """
        cls._check_no_locals()

        if data_dimensions is None:
            data_dimensions = {}

        with State._FactorySwapDimensionsDefinitions(quantity_factory, data_dimensions):
            state = cls._init(quantity_factory.empty)
        state.fill(value)
        return state

    @classmethod
    def copy_memory(
        cls,
        quantity_factory: QuantityFactory,
        memory_map: StateMemoryMapping,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        """Allocate all quantities and fill their value based
        on the given memory map. See `update_from_memory`.

        Args:
            quantity_factory: factory, expected to be defined on the Grid dimensions
                e.g. without data dimensions.
            memory_map: Dict of name/buffer. See `update_from_memory`.
            data_dimensions: extra data dimensions required for any field with data dimensions.
                Dict of name/size pair.
        """
        cls._check_no_locals()

        if data_dimensions is None:
            data_dimensions = {}

        state = cls.zeros(quantity_factory, data_dimensions=data_dimensions)
        state.update_copy_memory(memory_map)

        return state

    @classmethod
    def move_memory(
        cls,
        quantity_factory: QuantityFactory,
        memory_map: StateMemoryMapping,
        *,
        data_dimensions: dict[str, int] | None = None,
        check_shape_and_strides: bool = True,
    ) -> Self:
        """Allocate all quantities and move memory based on
        on the given memory map. See `update_move_memory`.

        Args:
            quantity_factory: factory, expected to be defined on the Grid dimensions
                e.g. without data dimensions.
            memory_map: Dict of name/buffer. See `update_from_memory`.
            data_dimensions: extra data dimensions required for any field with data dimensions.
                Dict of name/size pair.
            check_shape_and_strides: Check for every given buffer that the shape & strides match the
                previously allocated memory.
        """
        cls._check_no_locals()

        if data_dimensions is None:
            data_dimensions = {}

        state = cls.zeros(quantity_factory, data_dimensions=data_dimensions)
        state.update_move_memory(
            memory_map,
            check_shape_and_strides=check_shape_and_strides,
        )

        return state

    def fill(self, value: Number) -> None:
        def _fill_recursive(
            state: State,
            value: Number,
        ) -> None:
            for _field in dataclasses.fields(state):
                if dataclasses.is_dataclass(_field.type):
                    _fill_recursive(state.__getattribute__(_field.name), value)
                else:
                    state.__getattribute__(_field.name).field[:] = value

        _fill_recursive(self, value)

    def update_copy_memory(self, memory_map: dict[str, Any]) -> None:
        """Copy data into the Quantities carried by the state.

        The memory map must follow the dataclass naming convention, e.g.

        ```python
        @dataclass
        class MyState:
            @dataclass
            class InnerA
                a: Quantity

            inner_a: InnerA
            b: Quantity
        ```
        will update with a dictionary looking like
        ```python
        {
            "inner_a":
            {
                "a": Quantity(...)
            }
            "b": Quantity(...)

        }
        ```

        The memory map can be sparse.
        """

        def _update_from_memory_recursive(
            state: State,
            memory_map: StateMemoryMapping,
        ) -> None:
            for name, array in memory_map.items():
                if array is None:
                    if self.optional_quantities[name]:
                        state.__setattr__(name, None)
                    else:
                        raise TypeError(
                            f"State memory copy: illegal copy from None for attribute {name}"
                        )
                elif isinstance(array, dict):
                    _update_from_memory_recursive(state.__getattribute__(name), array)
                else:
                    try:
                        state.__getattribute__(name).field[:] = array
                    except Exception as e:
                        e.add_note(
                            f"Error when initializing field {name} on state {type(self)}"
                        )
                        raise e

        _update_from_memory_recursive(self, memory_map)

    def update_move_memory(
        self,
        memory_map: StateMemoryMapping,
        *,
        check_shape_and_strides: bool = True,
    ) -> None:
        """Move memory into the Quantities carried by the state.
        Memory is moved rather than copied (e.g. buffers are swapped)

        The memory map must follow the dataclass naming convention, e.g.

        ```python
        @dataclass
        class MyState:
            @dataclass
            class InnerA
                a: Quantity

            inner_a: InnerA
            b: Quantity
        ```
        will update with a dictionary looking like
        ```python
        {
            "inner_a":
            {
                "a": Quantity(...)
            }
            "b": Quantity(...)

        }
        ```

        The memory map can be sparse.

        Args:
            memory_map: Dictionary of keys to buffers. Buffers must be np.ArrayLike
            check_shape_and_strides: check that the given buffers have the same
                shape and strides as the original quantity
        """

        def _update_zero_copy_recursive(
            state: State, memory_map: StateMemoryMapping
        ) -> None:
            for name, array in memory_map.items():
                if array is None:
                    state.__setattr__(name, None)
                elif isinstance(array, dict):
                    _update_zero_copy_recursive(state.__getattribute__(name), array)
                else:
                    quantity = state.__getattribute__(name)
                    if check_shape_and_strides:
                        if array.shape != quantity.field.shape:
                            e = ValueError("Shape mismatch on zero copy for")
                            e.add_note(f"  Error on {name} for {type(state)}")
                            e.add_note(
                                f"  Shapes: {array.shape} != {quantity.field.shape}"
                            )
                            raise e
                        if array.strides != quantity.data.strides:
                            e = ValueError("Stride mismatch on zero copy for")
                            e.add_note(f"  Error on {name} for {type(state)}")
                            e.add_note(
                                f"  Strides: {array.strides} != {quantity.data.strides}"
                            )
                            raise e
                    try:
                        quantity.data = array
                    except Exception as e:
                        e.add_note(f"  Error on {name} for {type(state)}")
                        raise e

        _update_zero_copy_recursive(self, memory_map)

    def _hash(self) -> int:
        """Custom memory hash.

        We do not use __hash__ because of issues with subclassing.
        """

        def _flatten_elements_for_hash(
            state: State, flatten_hashable_list: list[Hashable]
        ) -> None:
            for _field in dataclasses.fields(state):
                element = state.__getattribute__(_field.name)
                if dataclasses.is_dataclass(_field.type):
                    _flatten_elements_for_hash(element, flatten_hashable_list)
                else:
                    flatten_hashable_list.append(element)

        to_hash: list[Hashable] = []
        _flatten_elements_for_hash(self, to_hash)
        return hash(tuple(to_hash))

    def _netcdf_name(self, directory_path: Path, postfix: str = "") -> Path:
        """Resolve rank-tied postfix if needed"""
        rank_postfix = ""
        if MPI.COMM_WORLD.Get_size() > 1:
            rank_postfix = f"_rank{MPI.COMM_WORLD.Get_rank()}"
        return directory_path / f"{type(self).__name__}{rank_postfix}{postfix}.nc4"

    def to_netcdf(self, directory_path: Path | None = None, postfix: str = "") -> None:
        """
        Save state to NetCDF. Can be reloaded with `update_from_netcdf`.

        If applicable, will save separate NetCDF files for each running rank.

        The file names are deduced from the class name, and post fix with rank number
        in the case of a multi-process use.

        Args:
            directory_path: directory to save the netcdf in
        """
        if directory_path is None:
            directory_path = Path("./")

        def _save_recursive(state: State) -> dict:
            local_data = {}
            for _field in dataclasses.fields(state):
                if dataclasses.is_dataclass(_field.type):
                    local_data[_field.name] = xr.Dataset(
                        data_vars=_save_recursive(state.__getattribute__(_field.name))
                    )
                else:
                    if "dims" not in _field.metadata.keys():
                        raise ValueError(
                            "Malformed state - no dims to init "
                            f"Quantity in  {_field.name} of type {_field.type}"
                        )

                    quantity = state.__getattribute__(_field.name)
                    if quantity is not None:
                        local_data[_field.name] = quantity.field_as_xarray

            return local_data

        datatree = _save_recursive(self)

        # Move top-level into their own dataset in the "/" prefix
        # to match DataTree expected format
        top_level = {}
        for key, value in datatree.items():
            if not isinstance(value, xr.Dataset):
                top_level[key] = value
        for key, _value in top_level.items():
            datatree.pop(key)
        datatree["/"] = xr.Dataset(data_vars=top_level)

        xr.DataTree.from_dict(datatree).to_netcdf(
            self._netcdf_name(directory_path, postfix)
        )

    def update_from_netcdf(self, directory_path: Path, postfix: str = "") -> None:
        """This is a mirror of the `to_netcdf` method NOT a generic
        NetCDF loader. It expects the NetCDF to be named with the auto-naming scheme
        of `to_netcdf`.

        Args:
            directory_path: directory carrying the netcdf saved with `to_netcdf`

        """
        datatree = xr.open_datatree(self._netcdf_name(directory_path, postfix))
        datatree_as_dict = datatree.to_dict()

        # All other cases - recursing downward
        def _load_recursive(
            data_tree_as_dict: dict[str, xr.Dataset] | xr.Dataset,
        ) -> dict:
            local_data_dict = {}
            for name, data_array in data_tree_as_dict.items():
                # Case of the top_level "/"
                if name == "/":
                    for root_name, root_data_array in datatree_as_dict["/"].items():
                        local_data_dict[root_name] = root_data_array.to_numpy()
                else:
                    # Get the leading `/` out
                    if isinstance(data_array, xr.Dataset):
                        local_data_dict[name[1:]] = _load_recursive(data_array)
                    else:
                        local_data_dict[name] = data_array.to_numpy()

            return local_data_dict

        data_as_numpy_dict = _load_recursive(datatree_as_dict)

        self.update_copy_memory(data_as_numpy_dict)


@dataclasses.dataclass
class LocalState(State):
    def __post_init__(self) -> None:
        from ndsl.internal.python_extensions import find_first_NDSLRuntime_caller

        runtime = find_first_NDSLRuntime_caller(inspect.currentframe())
        if runtime is None:
            raise RuntimeError("LocalState allocated outside of NDSLRuntime: forbidden")

        self._parent_runtime = runtime

    def _no_op__post_init__(self) -> None:
        pass

    @classmethod
    def _check_only_locals(cls) -> None:
        def _check_only_locals_recursive(cls: Any) -> None:
            for _field in dataclasses.fields(cls):
                if dataclasses.is_dataclass(_field.type):
                    _check_only_locals_recursive(_field.type)
                elif _field.type == Quantity:
                    raise TypeError(
                        f"State contains Quantity {_field.name} but is allocated as a LocalState. "
                        "LocalState with Locals can _only_ contain Locals."
                    )

        _check_only_locals_recursive(cls)

    @classmethod
    def make_locals(
        cls,
        quantity_factory: QuantityFactory,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        """Allocate all elements as Locals. Do not expect 0 on values, values are random.

        Args:
            quantity_factory: factory, expected to be defined on the Grid dimensions
                e.g. without data dimensions.
            data_dimensions: extra data dimensions required for any field with data dimensions.
                Dict of name/size pair.
        """
        cls._check_only_locals()

        if data_dimensions is None:
            data_dimensions = {}

        with cls._FactorySwapDimensionsDefinitions(quantity_factory, data_dimensions):
            state = cls._init(quantity_factory.empty)
        return state

    @classmethod
    def make_as_state(
        cls,
        quantity_factory: QuantityFactory,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        """Allow LocalState to be allocate as if it was a regular State, e.g. with Quantities.

        This behavior is useful for testing and interfacing, but should not be used in the
        regular numerical workflow, hence the warning."""

        warnings.warn(
            "LocalState is allocated as a regular State (e.g. elements are Quantities instead of Locals)."
            "This is not the intended use and should be only used for testing.",
            category=UserWarning,
            stacklevel=2,
        )
        cls._check_only_locals()

        if data_dimensions is None:
            data_dimensions = {}

        class _DeactivatePostInitMethod:
            """[Here be 🐉] Temporily shadow the __post_init__ method to deactivate the guardrails
            of the LocalState. DO NOT USE OUTSIDE OF THIS FUNCTION."""

            def __init__(self) -> None:
                pass

            def __enter__(self) -> None:
                self._original_post_init = LocalState.__post_init__
                LocalState.__post_init__ = LocalState._no_op__post_init__  # type: ignore[method-assign]

            def __exit__(
                self,
                type: type[BaseException] | None,
                value: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                LocalState.__post_init__ = self._original_post_init  # type: ignore[method-assign]

        with _DeactivatePostInitMethod():

            def _swap_local_recursive(cls: Any) -> None:
                for _field in dataclasses.fields(cls):
                    if dataclasses.is_dataclass(_field.type):
                        _swap_local_recursive(_field.type)
                    elif _field.type is Local:
                        _field.type = Quantity

            _swap_local_recursive(cls)

            with cls._FactorySwapDimensionsDefinitions(
                quantity_factory, data_dimensions
            ):
                state = cls._init(quantity_factory.empty, type_check=False)

            def _restore_local_recursive(cls: Any) -> None:
                for _field in dataclasses.fields(cls):
                    if dataclasses.is_dataclass(_field.type):
                        _swap_local_recursive(_field.type)
                    elif _field.type is Quantity:
                        _field.type = Local

            _restore_local_recursive(cls)

        return state

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)

        # We look for the first NDSLRuntime caller - we should be allocate alongside
        # it - all other allocations are forbidden
        from ndsl.internal.python_extensions import find_all_NDSLRuntime_callers

        if isinstance(attr, Local):
            runtimes = find_all_NDSLRuntime_callers(inspect.currentframe())
            if "_patched" in type(self._parent_runtime).__name__:
                unpatched_name = type(self._parent_runtime).__name__[: -len("_patched")]
            else:
                unpatched_name = type(self._parent_runtime).__name__
            # No frame -> algorithmics breaks down OR locals allocate outside
            # of a NDSLRuntime
            if runtimes == []:
                raise RuntimeError(
                    f"Forbidden Local access: {name} called outside "
                    f"of it's original NDSLRuntime ({unpatched_name})."
                )

            # Check my parent runtime is in the stack
            for runtime in runtimes:
                if runtime == self._parent_runtime:
                    return attr

            # No luck - we are probably called from the wrong NDSLRuntime: forbidden
            raise RuntimeError(
                f"Forbidden Local access: {name} called outside of {unpatched_name}."
            )

        return attr

    @classmethod
    def zeros(
        cls,
        quantity_factory: QuantityFactory,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        raise TypeError("LocalState cannot be allocated to zeros, use `make_locals`")

    @classmethod
    def empty(
        cls,
        quantity_factory: QuantityFactory,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        raise TypeError("LocalState cannot be allocated to empty, use `make_locals`")

    @classmethod
    def ones(
        cls,
        quantity_factory: QuantityFactory,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        raise TypeError("LocalState cannot be allocated to ones, use `make_locals`")

    @classmethod
    def full(
        cls,
        quantity_factory: QuantityFactory,
        value: Number,
        *,
        data_dimensions: dict[str, int] | None = None,
    ) -> Self:
        raise TypeError("LocalState cannot be allocated to full, use `make_locals`")
