from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from gt4py import storage as gt_storage

from ndsl.constants import SPATIAL_DIMS
from ndsl.dsl.typing import Float
from ndsl.initialization import GridSizer
from ndsl.logging import ndsl_log
from ndsl.quantity import Quantity, QuantityHaloSpec


class _Allocator:
    def __init__(self, backend: str) -> None:
        """
        Initialize an object that provides gt4py storage objects for zeros(), ones(), and empty().

        Args:
            backend: GT4Py backend name used for performance-optimized allocation.
        """
        self.backend = backend

    def empty(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return gt_storage.empty(*args, backend=self.backend, **kwargs)

    def ones(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return gt_storage.ones(*args, backend=self.backend, **kwargs)

    def zeros(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return gt_storage.zeros(*args, backend=self.backend, **kwargs)


class QuantityFactory:
    def __init__(self, sizer: GridSizer, *, backend: str) -> None:
        """
        Initialize a QuantityFactory from a GridSizer and a GT4Py backend name.

        Args:
            sizer: GridSizer object that determines the array sizes.
            backend: GT4Py backend name used for performance-optimized allocation.
        """
        self.sizer = sizer
        self.backend = backend

        self._allocator = _Allocator(self.backend)

    @classmethod
    def from_backend(cls, sizer: GridSizer, backend: str) -> QuantityFactory:
        """Initialize a QuantityFactory to use a specific GT4Py backend.

        Note: This method is deprecated. Please change your code to use the
        constructor instead.

        Args:
            sizer: GridSizer object that determines the array sizes.
            backend: GT4Py backend name used for performance-optimized allocation.
        """
        ndsl_log.warning(
            "QuantityFactory.from_backend(sizer, backend) is deprecated. Use QuantityFactory(sizer, backend=backend) instead."
        )
        return cls(sizer, backend=backend)

    def set_extra_dim_lengths(self, **kwargs: Any) -> None:
        """
        Set the length of extra (non-x/y/z) dimensions.
        """
        warnings.warn(
            "`QuantityFactory.set_extra_dim_lengths` is deprecated. "
            "Use `add_data_dimensions` or `update_data_dimensions`.",
            DeprecationWarning,
            2,
        )
        self.sizer.data_dimensions.update(kwargs)

    def update_data_dimensions(
        self,
        data_dimension_descriptions: dict[str, int],
    ) -> None:
        """
        Update the length of data (non-x/y/z) dimensions, unknown data dimensions
        will be added, existing ones updated.

        Args:
            data_dimension_descriptions: Dict of name/length pairs
        """
        self.sizer.data_dimensions.update(data_dimension_descriptions)

    def add_data_dimensions(
        self,
        data_dimension_descriptions: dict[str, int],
    ) -> None:
        """
        Add new data (non-x/y/z) dimensions via a key-length pair. If the dimension
        already exists, it will error out.

        Args:
            data_dimension_descriptions: Dict of name/length pairs
        """
        for name in data_dimension_descriptions.keys():
            if name in self.sizer.data_dimensions.keys():
                raise ValueError(
                    f"[NDSL] Data dimension {name} already exists! "
                    "Use `update_data_dimensions` if you meant to update the length."
                )

        self.sizer.data_dimensions.update(data_dimension_descriptions)

    def empty(
        self,
        dims: Sequence[str],
        units: str,
        dtype: type = Float,
        *,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """Allocate a Quantity and fill it with uninitialized (undefined) values.
        
        Equivalent to `numpy.empty`"""
        return self._allocate(
            self._allocator.empty, dims, units, dtype, allow_mismatch_float_precision
        )

    def zeros(
        self,
        dims: Sequence[str],
        units: str,
        dtype: type = Float,
        *,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """Allocate a Quantity and fill it with the value 0.

        Equivalent to `numpy.zeros`"""
        return self._allocate(
            self._allocator.zeros, dims, units, dtype, allow_mismatch_float_precision
        )

    def ones(
        self,
        dims: Sequence[str],
        units: str,
        dtype: type = Float,
        *,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """Allocate a Quantity and fill it with the value 1.

        Equivalent to `numpy.ones`"""
        return self._allocate(
            self._allocator.ones, dims, units, dtype, allow_mismatch_float_precision
        )

    def full(
        self,
        dims: Sequence[str],
        units: str,
        value: Any,  # no type hint because it would be a TypeVar = type[dtype] and mypy says no
        dtype: type = Float,
        *,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """Allocate a Quantity and fill it with the given value.

        Equivalent to `numpy.full`"""
        quantity = self._allocate(
            self._numpy.empty,
            dims,
            units,
            dtype,
            allow_mismatch_float_precision,
        )
        quantity.data[:] = value
        return quantity

    def from_array(
        self,
        data: np.ndarray,
        dims: Sequence[str],
        units: str,
        *,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """
        Create a Quantity from values in the `data` array.

        This copies the values of `data` into the resulting Quantity. The data
        array thus must correspond to the correct shape and extent for the given dims.
        """
        base = self.zeros(
            dims=dims,
            units=units,
            dtype=data.dtype,
            allow_mismatch_float_precision=allow_mismatch_float_precision,
        )
        base.data[:] = base.np.asarray(data)
        return base

    def from_compute_array(
        self,
        data: np.ndarray,
        dims: Sequence[str],
        units: str,
        *,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """
        Create a Quantity from values of the compute domain.

        This function will allocate the full Quantity (including potential
        halo points) to zero. The values of `data` are then copied into
        the compute domain. That numpy array must correspond to the correct
        shape and extent of the compute domain for the given dims.
        """
        base = self.zeros(
            dims=dims,
            units=units,
            dtype=data.dtype,
            allow_mismatch_float_precision=allow_mismatch_float_precision,
        )
        base.view[:] = base.np.asarray(data)
        return base

    def _allocate(
        self,
        allocator: Callable,
        dims: Sequence[str],
        units: str,
        dtype: type = Float,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        origin = self.sizer.get_origin(dims)
        extent = self.sizer.get_extent(dims)
        shape = self.sizer.get_shape(dims)
        dimensions = [
            (
                axis
                if any(dim in axis_dims for axis_dims in SPATIAL_DIMS)
                else str(shape[index])
            )
            for index, (dim, axis) in enumerate(
                zip(dims, ("I", "J", "K", *([None] * (len(dims) - 3))))
            )
        ]
        try:
            data = allocator(
                shape, dtype=dtype, aligned_index=origin, dimensions=dimensions
            )
        except TypeError:
            data = allocator(shape, dtype=dtype)
        return Quantity(
            data,
            dims=dims,
            units=units,
            origin=origin,
            extent=extent,
            gt4py_backend=self.backend,
            allow_mismatch_float_precision=allow_mismatch_float_precision,
            number_of_halo_points=self.sizer.n_halo,
        )

    def get_quantity_halo_spec(
        self,
        dims: Sequence[str],
        n_halo: int | None = None,
        dtype: type = Float,
    ) -> QuantityHaloSpec:
        """Build memory specifications for the halo update.

        Args:
            dims: dimensionality of the data
            n_halo: number of halo points to update, defaults to self.n_halo
            dtype: data type of the data
            backend: gt4py backend to use
        """

        # TEMPORARY: we do a nasty temporary allocation here to read in the hardware
        # memory layout. Further work in GT4PY will allow for deferred allocation
        # which will give access to those information while making sure
        # we don't allocate
        # Refactor is filed in ticket DSL-820

        temp_quantity = self.zeros(dims=dims, units="", dtype=dtype)

        if n_halo is None:
            n_halo = self.sizer.n_halo

        return temp_quantity.halo_spec(n_halo)
