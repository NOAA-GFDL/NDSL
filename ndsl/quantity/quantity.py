import warnings
from typing import Any, Iterable, Optional, Sequence, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import ndsl.constants as constants
from ndsl.dsl.typing import Float, is_float
from ndsl.optional_imports import cupy, dace, gt4py
from ndsl.optional_imports import xarray as xr
from ndsl.quantity.bounds import BoundedArrayView
from ndsl.quantity.metadata import QuantityHaloSpec, QuantityMetadata
from ndsl.types import NumpyModule


if cupy is None:
    import numpy as cupy


class Quantity:
    """
    Data container for physical quantities.
    """

    def __init__(
        self,
        data,
        dims: Sequence[str],
        units: str,
        origin: Optional[Sequence[int]] = None,
        extent: Optional[Sequence[int]] = None,
        gt4py_backend: Union[str, None] = None,
        allow_mismatch_float_precision: bool = False,
    ):
        """
        Initialize a Quantity.

        Args:
            data: ndarray-like object containing the underlying data
            dims: dimension names for each axis
            units: units of the quantity
            origin: first point in data within the computational domain
            extent: number of points along each axis within the computational domain
            gt4py_backend: backend to use for gt4py storages, if not given this will
                be derived from a Storage if given as the data argument, otherwise the
                storage attribute is disabled and will raise an exception. Will raise
                a TypeError if this is given with a gt4py storage type as data
        """

        if (
            not allow_mismatch_float_precision
            and is_float(data.dtype)
            and data.dtype != Float
        ):
            raise ValueError(
                f"Floating-point data type mismatch, asked for {data.dtype}, "
                f"Pace configured for {Float}"
            )
        if origin is None:
            origin = (0,) * len(dims)  # default origin at origin of array
        else:
            origin = tuple(origin)

        if extent is None:
            extent = tuple(length - start for length, start in zip(data.shape, origin))
        else:
            extent = tuple(extent)

        if isinstance(data, (int, float, list)):
            # If converting basic data, use a numpy ndarray.
            data = np.asarray(data)

        if not isinstance(data, (np.ndarray, cupy.ndarray)):
            raise TypeError(
                f"Only supports numpy.ndarray and cupy.ndarray, got {type(data)}"
            )

        if gt4py_backend is not None:
            gt4py_backend_cls = gt4py.cartesian.backend.from_name(gt4py_backend)
            assert gt4py_backend_cls is not None
            is_optimal_layout = gt4py_backend_cls.storage_info["is_optimal_layout"]

            dimensions: Tuple[Union[str, int], ...] = tuple(
                [
                    axis
                    if any(dim in axis_dims for axis_dims in constants.SPATIAL_DIMS)
                    else str(data.shape[index])
                    for index, (dim, axis) in enumerate(
                        zip(dims, ("I", "J", "K", *([None] * (len(dims) - 3))))
                    )
                ]
            )

            self._data = (
                data
                if is_optimal_layout(data, dimensions)
                else self._initialize_data(
                    data,
                    origin=origin,
                    gt4py_backend=gt4py_backend,
                    dimensions=dimensions,
                )
            )
        else:
            if data is None:
                raise TypeError("requires 'data' to be passed")
            # We have no info about the gt4py_backend, so just assign it.
            self._data = data

        _validate_quantity_property_lengths(data.shape, dims, origin, extent)
        self._metadata = QuantityMetadata(
            origin=_ensure_int_tuple(origin, "origin"),
            extent=_ensure_int_tuple(extent, "extent"),
            dims=tuple(dims),
            units=units,
            data_type=type(self._data),
            dtype=data.dtype,
            gt4py_backend=gt4py_backend,
        )
        self._attrs = {}  # type: ignore[var-annotated]
        self._compute_domain_view = BoundedArrayView(
            self.data, self.dims, self.origin, self.extent
        )

    @classmethod
    def from_data_array(
        cls,
        data_array: xr.DataArray,
        origin: Sequence[int] = None,
        extent: Sequence[int] = None,
        gt4py_backend: Union[str, None] = None,
    ) -> "Quantity":
        """
        Initialize a Quantity from an xarray.DataArray.

        Args:
            data_array
            origin: first point in data within the computational domain
            extent: number of points along each axis within the computational domain
            gt4py_backend: backend to use for gt4py storages, if not given this will
                be derived from a Storage if given as the data argument, otherwise the
                storage attribute is disabled and will raise an exception
        """
        if "units" not in data_array.attrs:
            raise ValueError("need units attribute to create Quantity from DataArray")
        return cls(
            data_array.values,
            cast(Tuple[str], data_array.dims),
            data_array.attrs["units"],
            origin=origin,
            extent=extent,
            gt4py_backend=gt4py_backend,
        )

    def to_netcdf(self, path: str, name="var", rank: int = -1) -> None:
        if rank < 0 or MPI.COMM_WORLD.Get_rank() == rank:
            self.data_array.to_dataset(name=name).to_netcdf(f"{path}__r{rank}.nc4")

    def halo_spec(self, n_halo: int) -> QuantityHaloSpec:
        return QuantityHaloSpec(
            n_halo,
            self.data.strides,
            self.data.itemsize,
            self.data.shape,
            self.metadata.origin,
            self.metadata.extent,
            self.metadata.dims,
            self.np,
            self.metadata.dtype,
        )

    def __repr__(self):
        return (
            f"Quantity(\n    data=\n{self.data},\n    dims={self.dims},\n"
            f"    units={self.units},\n    origin={self.origin},\n"
            f"    extent={self.extent}\n)"
        )

    def sel(self, **kwargs: Union[slice, int]) -> np.ndarray:
        """Convenience method to perform indexing on `view` using dimension names
        without knowing dimension order.

        Args:
            **kwargs: slice/index to retrieve for a given dimension name

        Returns:
            view_selection: an ndarray-like selection of the given indices
                on `self.view`
        """
        return self.view[tuple(kwargs.get(dim, slice(None, None)) for dim in self.dims)]

    def _initialize_data(self, data, origin, gt4py_backend: str, dimensions: Tuple):
        """Allocates an ndarray with optimal memory layout, and copies the data over."""
        storage = gt4py.storage.from_array(
            data,
            data.dtype,
            backend=gt4py_backend,
            aligned_index=origin,
            dimensions=dimensions,
        )
        return storage

    @property
    def metadata(self) -> QuantityMetadata:
        return self._metadata

    @property
    def units(self) -> str:
        """units of the quantity"""
        return self.metadata.units

    @property
    def gt4py_backend(self) -> Union[str, None]:
        return self.metadata.gt4py_backend

    @property
    def attrs(self) -> dict:
        return dict(**self._attrs, units=self._metadata.units)

    @property
    def dims(self) -> Tuple[str, ...]:
        """names of each dimension"""
        return self.metadata.dims

    @property
    def values(self) -> np.ndarray:
        warnings.warn(
            "values exists only for backwards-compatibility with "
            "DataArray and will be removed, use .view[:] instead",
            DeprecationWarning,
        )
        return_array = np.asarray(self.view[:])
        return_array.flags.writeable = False
        return return_array

    @property
    def view(self) -> BoundedArrayView:
        """a view into the computational domain of the underlying data"""
        return self._compute_domain_view

    @property
    def data(self) -> Union[np.ndarray, cupy.ndarray]:
        """the underlying array of data"""
        return self._data

    @data.setter
    def data(self, inputData):
        if type(inputData) in [np.ndarray, cupy.ndarray]:
            self._data = inputData

    @property
    def origin(self) -> Tuple[int, ...]:
        """the start of the computational domain"""
        return self.metadata.origin

    @property
    def extent(self) -> Tuple[int, ...]:
        """the shape of the computational domain"""
        return self.metadata.extent

    @property
    def data_array(self, full_data=False) -> xr.DataArray:
        """Returns an Xarray.DataArray of the view (domain)

        Args:
            full_data: Return the entire data (halo included) instead of the view
        """
        if full_data:
            return xr.DataArray(self.data[:], dims=self.dims, attrs=self.attrs)
        else:
            return xr.DataArray(self.view[:], dims=self.dims, attrs=self.attrs)

    @property
    def np(self) -> NumpyModule:
        return self.metadata.np

    @property
    def __array_interface__(self):
        return self.data.__array_interface__

    @property
    def __cuda_array_interface__(self):
        return self.data.__cuda_array_interface__

    @property
    def shape(self):
        return self.data.shape

    def __descriptor__(self) -> Any:
        """The descriptor is a property that dace uses.
        This relies on `dace` capacity to read out data from the buffer protocol.
        If the internal data given doesn't follow the protocol it will most likely
        fail.
        """
        if dace:
            return dace.data.create_datadescriptor(self.data)
        else:
            raise ImportError(
                "Attempt to use DaCe orchestrated backend but "
                "DaCe module is not available."
            )

    def transpose(
        self,
        target_dims: Sequence[Union[str, Iterable[str]]],
        allow_mismatch_float_precision: bool = False,
    ) -> "Quantity":
        """Change the dimension order of this Quantity.

        Args:
            target_dims: a list of output dimensions. Instead of a single dimension
                name, an iterable of dimensions can be used instead for any entries.
                For example, you may want to use X_DIMS to place an
                x-dimension without knowing whether it is on cell centers or interfaces.

        Returns:
            transposed: Quantity with the requested output dimension order

        Raises:
            ValueError: if any of the target dimensions do not exist on this Quantity,
                or if this Quantity contains multiple values from an iterable entry

        Examples:
            Let's say we have a cell-centered variable:

            >>> import ndsl.util
            >>> import numpy as np
            >>> quantity = Quantity(
            ...     data=np.zeros([2, 3, 4]),
            ...     dims=[X_DIM, Y_DIM, Z_DIM],
            ...             units="m",
            ... )

            If you know you are working with cell-centered variables, you can do:

            >>> from ndsl.constants import X_DIM, Y_DIM, Z_DIM
            >>> transposed_quantity = quantity.transpose([X_DIM, Y_DIM, Z_DIM])

            To support re-ordering without checking whether quantities are on
            cell centers or interfaces, the API supports giving a list of dimension
            names for dimensions. For example, to re-order to X-Y-Z dimensions
            regardless of the grid the variable is on, one could do:

            >>> from ndsl.constants import X_DIMS, Y_DIMS, Z_DIMS
            >>> transposed_quantity = quantity.transpose([X_DIMS, Y_DIMS, Z_DIMS])
        """
        target_dims = _collapse_dims(target_dims, self.dims)
        transpose_order = [self.dims.index(dim) for dim in target_dims]
        transposed = Quantity(
            self.np.transpose(self.data, transpose_order),  # type: ignore[attr-defined]
            dims=_transpose_sequence(self.dims, transpose_order),
            units=self.units,
            origin=_transpose_sequence(self.origin, transpose_order),
            extent=_transpose_sequence(self.extent, transpose_order),
            gt4py_backend=self.gt4py_backend,
            allow_mismatch_float_precision=allow_mismatch_float_precision,
        )
        transposed._attrs = self._attrs
        return transposed

    def plot_k_level(self, k_index=0):
        field = self.data
        print(
            "Min and max values:",
            field[:, :, k_index].min(),
            field[:, :, k_index].max(),
        )
        plt.xlabel("I")
        plt.ylabel("J")

        im = plt.imshow(field[:, :, k_index].transpose(), origin="lower")

        plt.colorbar(im)
        plt.title("Plot at K = " + str(k_index))
        plt.show()


def _transpose_sequence(sequence, order):
    return sequence.__class__(sequence[i] for i in order)


def _collapse_dims(target_dims, dims):
    return_list = []
    for target in target_dims:
        if isinstance(target, str):
            if target in dims:
                return_list.append(target)
            else:
                raise ValueError(
                    f"requested dimension {target} is not defined in "
                    f"quantity dimensions {dims}"
                )
        elif isinstance(target, Iterable):
            matches = [d for d in target if d in dims]
            if len(matches) > 1:
                raise ValueError(
                    f"multiple matches for {target} found in quantity dimensions {dims}"
                )
            elif len(matches) == 0:
                raise ValueError(
                    f"no matches for {target} found in quantity dimensions {dims}"
                )
            else:
                return_list.append(matches[0])
    return return_list


def _validate_quantity_property_lengths(shape, dims, origin, extent):
    n_dims = len(shape)
    for var, desc in (
        (dims, "dimension names"),
        (origin, "origins"),
        (extent, "extents"),
    ):
        if len(var) != n_dims:
            raise ValueError(
                f"received {len(var)} {desc} for {n_dims} dimensions: {var}"
            )


def _ensure_int_tuple(arg, arg_name):
    return_list = []
    for item in arg:
        try:
            return_list.append(int(item))
        except ValueError:
            raise TypeError(
                f"tuple arg {arg_name}={arg} contains item {item} of "
                f"unexpected type {type(item)}"
            )
    return tuple(return_list)
