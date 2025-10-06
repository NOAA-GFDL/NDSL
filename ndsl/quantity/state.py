from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Self, TypeAlias

import dacite
import xarray as xr
from mpi4py import MPI
from numpy.typing import ArrayLike

from ndsl.types import Number


if TYPE_CHECKING:
    from ndsl import QuantityFactory

StateMemoryMapping: TypeAlias = dict[str, dict | ArrayLike | None]


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
    def _init(cls, quantity_factory_allocator: Callable) -> Self:
        """Allocate memory and init with a blind quantity init operation"""

        def _init_recursive(cls):
            initial_quantities = {}
            for _field in dataclasses.fields(cls):
                if dataclasses.is_dataclass(_field.type):
                    initial_quantities[_field.name] = _init_recursive(_field.type)
                else:
                    if "dims" not in _field.metadata.keys():
                        raise ValueError(
                            "Malformed state - no dims to init "
                            f"Quantity in  {_field.name} of type {_field.type}"
                        )

                    initial_quantities[_field.name] = quantity_factory_allocator(
                        _field.metadata["dims"],
                        _field.metadata["units"],
                        dtype=_field.metadata["dtype"],
                        allow_mismatch_float_precision=True,
                    )

            return initial_quantities

        dict_of_quantities = _init_recursive(cls)
        return dacite.from_dict(data_class=cls, data=dict_of_quantities)

    class _FactorySwapDimensionsDefinitions:
        """INTERNAL: QuantityFactory carry a sizer which has a full definition of the dimensions.
        It's this sizer that is leveraged for the factory to figure out allocations.
        In a regular pattern of use, data dimensions fields tend to be _the exception_ rather
        than the rule and therefor would need a Factory defined _for a few cases_.
        We bring this tool to override temporarly the allocations based on a single descriptions of
        the data dimenions at allocation time.
        """

        def __init__(self, factory: QuantityFactory, ddims: dict[str, int]):
            self._ddims = ddims
            self._factory = factory

        def __enter__(self):
            self._original_dims = self._factory.sizer.extra_dim_lengths
            self._factory.sizer.extra_dim_lengths = self._ddims

        def __exit__(self, type, value, traceback):
            self._factory.sizer.extra_dim_lengths = self._original_dims

    @classmethod
    def empty(
        cls,
        quantity_factory: QuantityFactory,
        data_dimensions: dict[str, int] = {},
    ) -> Self:
        """Allocate all quantities. Do not expect 0 on values, values are random.

        Args:
            quantity_factory: factory, expected to be defined on the Grid dimensions
                e.g. without data dimensions.
            data_dimensions: extra data dimensions required for any field with data dimensions.
                Dict of name/size pair.
        """

        with State._FactorySwapDimensionsDefinitions(quantity_factory, data_dimensions):
            state = cls._init(quantity_factory.empty)
        return state

    @classmethod
    def zeros(
        cls,
        quantity_factory: QuantityFactory,
        data_dimensions: dict[str, int] = {},
    ) -> Self:
        """Allocate all quantities and fill their value to zeros

        Args:
            quantity_factory: factory, expected to be defined on the Grid dimensions
                e.g. without data dimensions.
            data_dimensions: extra data dimensions required for any field with data dimensions.
                Dict of name/size pair.
        """

        with State._FactorySwapDimensionsDefinitions(quantity_factory, data_dimensions):
            state = cls._init(quantity_factory.zeros)
        return state

    @classmethod
    def ones(
        cls,
        quantity_factory: QuantityFactory,
        data_dimensions: dict[str, int] = {},
    ) -> Self:
        """Allocate all quantities and fill their value to ones

        Args:
            quantity_factory: factory, expected to be defined on the Grid dimensions
                e.g. without data dimensions.
            data_dimensions: extra data dimensions required for any field with data dimensions.
                Dict of name/size pair.
        """

        with State._FactorySwapDimensionsDefinitions(quantity_factory, data_dimensions):
            state = cls._init(quantity_factory.ones)
        return state

    @classmethod
    def copy_memory(
        cls,
        quantity_factory: QuantityFactory,
        memory_map: StateMemoryMapping,
        data_dimensions: dict[str, int] = {},
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

        state = cls.zeros(quantity_factory, data_dimensions=data_dimensions)
        state.update_copy_memory(memory_map)

        return state

    @classmethod
    def move_memory(
        cls,
        quantity_factory: QuantityFactory,
        memory_map: StateMemoryMapping,
        data_dimensions: dict[str, int] = {},
        *,
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
            check_shape_and_strides: Check for evey given buffer that the shape & strides match the
                previously allocated memory.
        """

        state = cls.zeros(quantity_factory, data_dimensions)
        state.update_move_memory(
            memory_map,
            check_shape_and_strides=check_shape_and_strides,
        )

        return state

    def fill(self, value: Number):
        def _fill_recursive(
            state: State,
            value: Number,
        ):
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
        ):
            for name, array in memory_map.items():
                if array is None:
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

        def _update_zero_copy_recursive(state: State, memory_map: StateMemoryMapping):
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

    def _netcdf_name(self, directory_path: Path) -> Path:
        """Resolve rank-tied postfix if needed"""
        rank_postfix = ""
        if MPI.COMM_WORLD.Get_size() > 1:
            rank_postfix = f"_rank{MPI.COMM_WORLD.Get_rank()}"
        return directory_path / f"{type(self).__name__}{rank_postfix}.nc4"

    def to_netcdf(self, directory_path: Path = Path("./")) -> None:
        """
        Save state to NetCDF. Can be reloaded with `update_from_netcdf`.

        If applicable, will save seperate NetCDF files for each running rank.

        The file names are deduced from the class name, and post fix with rank number
        in the case of a multi-process use.

        Args:
            directory_path: directory to save the netcdf in
        """

        def _save_recursive(state: State):
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

                    local_data[_field.name] = state.__getattribute__(
                        _field.name
                    ).field_as_xarray

            return local_data

        datatree = _save_recursive(self)

        # Move top-level into their own dataset in the "/" prefix
        # to match DataTree expected format
        top_level = {}
        for key, value in datatree.items():
            if not isinstance(value, xr.Dataset):
                top_level[key] = value
        for key, value in top_level.items():
            datatree.pop(key)
        datatree["/"] = xr.Dataset(data_vars=top_level)

        xr.DataTree.from_dict(datatree).to_netcdf(self._netcdf_name(directory_path))

    def update_from_netcdf(self, directory_path: Path) -> None:
        """This is a mirror of the `to_netcdf` method NOT a generic
        NetCDF loader. It expects the NetCDF to be named with the auto-naming scheme
        of `to_netcdf`.

        Args:
            directory_path: directory carrying the netcdf saved with `to_netcdf`

        """
        datatree = xr.open_datatree(self._netcdf_name(directory_path))
        datatree_as_dict = datatree.to_dict()

        # All other cases - recursing downward
        def _load_recursive(data_tree_as_dict: dict[str, xr.Dataset] | xr.Dataset):
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
