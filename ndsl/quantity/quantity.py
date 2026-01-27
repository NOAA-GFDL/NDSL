from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import Any, cast

import dace
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from gt4py import storage as gt_storage
from gt4py.cartesian import backend as gt_backend

import ndsl.constants as constants
from ndsl.comm.mpi import MPI
from ndsl.dsl.typing import Float, is_float
from ndsl.optional_imports import cupy
from ndsl.quantity.bounds import BoundedArrayView
from ndsl.quantity.metadata import QuantityHaloSpec, QuantityMetadata
from ndsl.types import NumpyModule


if cupy is None:
    import numpy as cupy


class Quantity:
    """Data container for physical quantities."""

    def __init__(
        self,
        data: np.ndarray | cupy.ndarray,
        dims: Sequence[str],
        units: str,
        *,
        backend: str,
        origin: Sequence[int] | None = None,
        extent: Sequence[int] | None = None,
        allow_mismatch_float_precision: bool = False,
        number_of_halo_points: int = 0,
    ):
        """Initialize a Quantity.

        Args:
            data: ndarray-like object containing the underlying data
            dims: dimension names for each axis
            units: units of the quantity
            backend: GT4Py backend name. We ensure that the data is allocated in a
                performance optimal way for that backend and copy if necessary.
            origin: first point in data within the
                computational domain. Defaults to None.
            extent: number of points along each axis
                within the computational domain. Defaults to None.
            allow_mismatch_float_precision: allow for precision that is
                not the simulation-wide default configuration. Defaults to False.
            number_of_halo_points: Number of halo points used. Defaults to 0.

        Raises:
            ValueError: Data-type mismatch between configuration and input-data
            TypeError: Typing of the data that does not fit
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

        if not isinstance(data, (np.ndarray, cupy.ndarray)):
            raise TypeError(
                f"Only supports numpy.ndarray and cupy.ndarray, got {type(data)}"
            )

        _validate_quantity_property_lengths(data.shape, dims, origin, extent)

        gt4py_backend_cls = gt_backend.from_name(backend)
        is_optimal_layout = gt4py_backend_cls.storage_info["is_optimal_layout"]
        device = gt4py_backend_cls.storage_info["device"]

        dimensions: tuple[str | int, ...] = tuple(
            [
                (
                    axis  # type: ignore # mypy can't parse this list construction of hell
                    if any(dim in axis_dims for axis_dims in constants.SPATIAL_DIMS)
                    else str(data.shape[index])
                )
                for index, (dim, axis) in enumerate(
                    zip(dims, ("I", "J", "K", *([None] * (len(dims) - 3))))
                )
            ]
        )

        if isinstance(data, np.ndarray):
            is_correct_device = device == "cpu"
        elif isinstance(data, cupy.ndarray):
            is_correct_device = device == "gpu"
        else:
            raise ValueError(
                f"Unknown device target for quantity allocation {type(data)}"
            )

        if is_optimal_layout(data, dimensions) and is_correct_device:
            self._data = data
        else:
            warnings.warn(
                f"Suboptimal data layout found. Copying data to optimally align for backend '{backend}'.",
                UserWarning,
                stacklevel=2,
            )
            self._data = gt_storage.from_array(
                data,
                data.dtype,
                backend=backend,
                aligned_index=origin,
                dimensions=dimensions,
            )

        self._metadata = QuantityMetadata(
            origin=_ensure_int_tuple(origin, "origin"),
            extent=_ensure_int_tuple(extent, "extent"),
            n_halo=number_of_halo_points,
            dims=tuple(dims),
            units=units,
            data_type=type(self._data),
            dtype=data.dtype,
            backend=backend,
        )
        self._attrs = {}  # type: ignore[var-annotated]
        self._compute_domain_view = BoundedArrayView(
            self.data, self.dims, self.origin, self.extent
        )

    @classmethod
    def from_data_array(
        cls,
        data_array: xr.DataArray,
        *,
        origin: Sequence[int] | None = None,
        extent: Sequence[int] | None = None,
        number_of_halo_points: int = 0,
        backend: str | None = None,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """
        Initialize a Quantity from an xarray.DataArray.

        Args:
            data_array
            origin: first point in data within the computational domain
            extent: number of points along each axis within the computational domain
            allow_mismatch_float_precision: allow for precision that is
                not the simulation-wide default configuration. Defaults to False.
            number_of_halo_points: Number of halo points used. Defaults to 0.
            backend: GT4Py backend name. If given, we allocate data in a performance
                optimal way for this backend. Overrides any potentially saved `backend`
                in `data.attrs["backend"]`.
        """
        if "units" not in data_array.attrs:
            data_array.attrs.update({"units": "unknown"})

        return cls(
            data_array.values,
            cast(tuple[str], data_array.dims),
            data_array.attrs["units"],
            origin=origin,
            extent=extent,
            number_of_halo_points=number_of_halo_points,
            backend=_resolve_backend(data_array, backend),
            allow_mismatch_float_precision=allow_mismatch_float_precision,
        )

    def to_netcdf(
        self, path: str, name: str = "var", rank: int = -1, all_data: bool = False
    ) -> None:
        if rank < 0 or MPI.COMM_WORLD.Get_rank() == rank:
            if rank < 0:
                rank = MPI.COMM_WORLD.Get_rank()
            if all_data:
                self.data_as_xarray.to_dataset(name=name).to_netcdf(
                    f"{path}__r{rank}.nc4"
                )
            else:
                self.field_as_xarray.to_dataset(name=name).to_netcdf(
                    f"{path}__r{rank}.nc4"
                )

    def halo_spec(self, n_halo: int) -> QuantityHaloSpec:
        # This is a preliminary check to see if this is ever triggered.
        # If not, we can remove it down the line and change the call signature.
        if n_halo != self._metadata.n_halo:
            warnings.warn(
                "Found inconsistency with number of halo points in Quantity:"
                + f"{n_halo} vs {self._metadata.n_halo}",
                UserWarning,
                stacklevel=2,
            )
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

    def __repr__(self) -> str:
        return (
            f"Quantity(\n    data=\n{self.data},\n    dims={self.dims},\n"
            f"    units={self.units},\n    origin={self.origin},\n"
            f"    extent={self.extent}\n)"
        )

    def sel(self, **kwargs: slice | int) -> np.ndarray:
        """Convenience method to perform indexing on `view` using dimension names
        without knowing dimension order.

        Args:
            **kwargs: slice/index to retrieve for a given dimension name

        Returns:
            view_selection: an ndarray-like selection of the given indices
                on `self.view`
        """
        return self.view[tuple(kwargs.get(dim, slice(None, None)) for dim in self.dims)]

    @property
    def metadata(self) -> QuantityMetadata:
        return self._metadata

    @property
    def units(self) -> str:
        """Units of the quantity"""
        return self.metadata.units

    @property
    def backend(self) -> str:
        return self.metadata.backend

    @property
    def attrs(self) -> dict:
        return dict(**self._attrs, units=self.units, backend=self.backend)

    @property
    def dims(self) -> tuple[str, ...]:
        """Names of each dimension"""
        return self.metadata.dims

    @property
    def view(self) -> BoundedArrayView:
        """A view into the computational domain of the underlying data"""
        return self._compute_domain_view

    @property
    def field(self) -> np.ndarray | cupy.ndarray:
        return self._compute_domain_view[:]

    @property
    def data(self) -> np.ndarray | cupy.ndarray:
        """The underlying array of data"""
        return self._data

    @data.setter
    def data(self, input_data: np.ndarray | cupy.ndarray) -> None:
        if type(input_data) not in [np.ndarray, cupy.ndarray]:
            raise TypeError(
                "Quantity.data buffer swap failed: "
                f"given data is not an array (type: {type(input_data)})"
            )

        if input_data.shape < self.extent:
            raise ValueError(
                "Quantity.data buffer swap failed: "
                f"new data ({input_data.shape}) is smaller "
                f"than expected extent ({self.extent})."
            )

        self._data = input_data
        self._compute_domain_view = BoundedArrayView(
            self.data, self.dims, self.origin, self.extent
        )

    @property
    def origin(self) -> tuple[int, ...]:
        """The start of the computational domain"""
        return self.metadata.origin

    @property
    def extent(self) -> tuple[int, ...]:
        """The shape of the computational domain"""
        return self.metadata.extent

    @property
    def field_as_xarray(self) -> xr.DataArray:
        """Returns an Xarray.DataArray of the field (domain)"""
        if isinstance(self.field, np.ndarray):
            field = self.field
        else:
            field = self.field.get()
        return xr.DataArray(field, dims=self.dims, attrs=self.attrs)

    @property
    def data_as_xarray(self) -> xr.DataArray:
        """Returns an Xarray.DataArray of the underlying array"""
        if isinstance(self.data, np.ndarray):
            data = self.data
        else:
            data = self.data.get()
        return xr.DataArray(data, dims=self.dims, attrs=self.attrs)

    @property
    def np(self) -> NumpyModule:
        return self.metadata.np

    @property
    def __array_interface__(self):  # type: ignore[no-untyped-def]
        return self.data.__array_interface__

    @property
    def __cuda_array_interface__(self):  # type: ignore[no-untyped-def]
        return self.data.__cuda_array_interface__

    def __hash__(self) -> int:
        """Hash based on underlying memory

        Quantity fundamentally represent a C-held memory on either CPU or GPU device.
        This hash does not cover _all_ of Quantity (metadata, etc.) but it reflects the
        runtime reality of Quantity.
        """
        if isinstance(self.data, np.ndarray):
            return hash(self.data.__array_interface__["data"])
        return hash(self.data.__cuda_array_interface__["data"])

    @property
    def shape(self):  # type: ignore[no-untyped-def]
        return self.data.shape

    def __descriptor__(self) -> Any:
        """The descriptor is a property that dace uses.

        This relies on `dace` capacity to read out data from the buffer protocol.
        If the internal data given doesn't follow the protocol it will most likely
        fail.
        """
        return dace.data.create_datadescriptor(self.data)

    def transpose(
        self,
        target_dims: Sequence[str | Iterable[str]],
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """Change the dimension order of this Quantity.

        Args:
            target_dims: a list of output dimensions. Instead of a single dimension
                name, an iterable of dimensions can be used instead for any entries.
                For example, you may want to use I_DIMS to place an
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
            ...     dims=[I_DIM, J_DIM, K_DIM],
            ...             units="m",
            ... )

            If you know you are working with cell-centered variables, you can do:

            >>> from ndsl.constants import I_DIM, J_DIM, K_DIM
            >>> transposed_quantity = quantity.transpose([I_DIM, J_DIM, K_DIM])

            To support re-ordering without checking whether quantities are on
            cell centers or interfaces, the API supports giving a list of dimension
            names for dimensions. For example, to re-order to X-Y-Z dimensions
            regardless of the grid the variable is on, one could do:

            >>> from ndsl.constants import I_DIMS, J_DIMS, K_DIMS
            >>> transposed_quantity = quantity.transpose([I_DIMS, J_DIMS, K_DIMS])
        """
        target_dims = _collapse_dims(target_dims, self.dims)
        transpose_order = [self.dims.index(dim) for dim in target_dims]
        transposed = Quantity(
            self.np.transpose(self.data, transpose_order),  # type: ignore[attr-defined]
            dims=_transpose_sequence(self.dims, transpose_order),
            units=self.units,
            origin=_transpose_sequence(self.origin, transpose_order),
            extent=_transpose_sequence(self.extent, transpose_order),
            allow_mismatch_float_precision=allow_mismatch_float_precision,
            backend=self.backend,
        )
        transposed._attrs = self._attrs
        return transposed

    def plot_k_level(self, k_index: int = 0) -> None:
        field = self.data
        plt.xlabel("I")
        plt.ylabel("J")

        im = plt.imshow(field[:, :, k_index].transpose(), origin="lower")

        plt.colorbar(im)
        plt.title("Plot at K = " + str(k_index))
        plt.show()


def _transpose_sequence(sequence, order):  # type: ignore[no-untyped-def]
    return sequence.__class__(sequence[i] for i in order)


def _collapse_dims(
    target_dims: Sequence[str | Iterable[str]], dims: tuple[str, ...]
) -> list[str]:
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


def _validate_quantity_property_lengths(shape, dims, origin, extent):  # type: ignore[no-untyped-def]
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


def _ensure_int_tuple(arg: Sequence, arg_name: str) -> tuple:
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


def _resolve_backend(data: xr.DataArray, backend: str | None) -> str:
    if backend is not None:
        # Forced backend name takes precedence
        return backend

    # If backend name was serialized with data, take this one
    if "backend" in data.attrs:
        return data.attrs["backend"]

    # else, fall back to assume python-based layout.
    return "debug"
