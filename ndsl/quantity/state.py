from typing import Any, Self
from ndsl import QuantityFactory
import dataclasses
import dacite
import xarray as xr
from mpi4py import MPI
from numpy.typing import ArrayLike


@dataclasses.dataclass
class State:
    """Base class for State object in models that bundles a collection
    of functions to deal with nested dataclasses and common usage of States:
        - init (zero, from memory, zero copy buffer swap)
        - IO (save to NetCDF, from NetCDF)

    The State expects Quantities.
    """

    @classmethod
    def zeros(cls, quantity_factory: QuantityFactory) -> Self:
        """Init all quantities to zeros - included nested ones"""

        def _zeros_recursive(cls):
            initial_quantities = {}
            for _field in dataclasses.fields(cls):
                if dataclasses.is_dataclass(_field.type):
                    initial_quantities[_field.name] = _zeros_recursive(_field.type)
                else:
                    if "dims" not in _field.metadata.keys():
                        raise ValueError(
                            "Malformed state - no dims to init "
                            f"Quantity in  {_field.name} of type {_field.type}"
                        )

                    initial_quantities[_field.name] = quantity_factory.zeros(
                        _field.metadata["dims"],
                        _field.metadata["units"],
                        dtype=_field.metadata["dtype"],
                        allow_mismatch_float_precision=True,
                    )

            return initial_quantities

        dict_of_qty = _zeros_recursive(cls)
        return dacite.from_dict(data_class=cls, data=dict_of_qty)

    def init_from_memory(self, memory_map: dict[str, Any]):
        """Will copy data from the memory map if it follows the nested
        naming convention of the dataclass"""

        def _init_from_memory_recursive(dataclss, memory_map: dict[str, Any]):
            for name, array in memory_map.items():
                if isinstance(array, dict):
                    _init_from_memory_recursive(dataclss.__getattribute__(name), array)
                else:
                    try:
                        dataclss.__getattribute__(name).field[:] = array
                    except ValueError as e:
                        e.add_note(
                            f"Error when initializing field {name} on state {type(self)}"
                        )
                        raise e

        _init_from_memory_recursive(self, memory_map)

    def init_zero_copy(self, memory_map: dict[str, Any], check: bool = True):
        """Swap buffers given into the Quantities carried by the state
        by following dataclass naming convention"""

        def _init_zero_copy_recursive(dataclss, memory_map: dict[str, Any | ArrayLike]):
            for name, array in memory_map.items():
                if isinstance(array, dict):
                    _init_zero_copy_recursive(dataclss.__getattribute__(name), array)
                else:
                    qty = dataclss.__getattribute__(name)
                    if check:
                        if array.shape != qty.field.shape:
                            e = ValueError("Shape mismatch on zero copy for")
                            e.add_note(f"  Error on {name} for {type(dataclss)}")
                            e.add_note(f"  Shapes: {array.shape} != {qty.field.shape}")
                            raise e
                        if array.strides != qty.data.strides:
                            e = ValueError("Stride mismatch on zero copy for")
                            e.add_note(f"  Error on {name} for {type(dataclss)}")
                            e.add_note(
                                f"  Strides: {array.strides} != {qty.data.strides}"
                            )
                            raise e

                    qty.data = array

        _init_zero_copy_recursive(self, memory_map)

    def to_netcdf(self, path: str = "./"):
        def _save_recursive(datclss: State):
            local_data = {}
            for _field in dataclasses.fields(datclss):
                if dataclasses.is_dataclass(_field.type):
                    local_data[_field.name] = xr.Dataset(
                        data_vars=_save_recursive(datclss.__getattribute__(_field.name))
                    )
                else:
                    if "dims" not in _field.metadata.keys():
                        raise ValueError(
                            "Malformed state - no dims to init "
                            f"Quantity in  {_field.name} of type {_field.type}"
                        )

                    local_data[_field.name] = datclss.__getattribute__(
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

        # Resolve rank-tied postfix if needed
        rank_postfix = ""
        if MPI.COMM_WORLD.Get_size() > 1:
            rank_postfix = f"_rank{MPI.COMM_WORLD.Get_rank()}"

        xr.DataTree.from_dict(datatree).to_netcdf(
            f"{path}{type(self).__name__}{rank_postfix}.nc4"
        )

    def from_netcdf(self, path: str):
        datatree = xr.open_datatree(path)
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

        self.init_from_memory(data_as_numpy_dict)
