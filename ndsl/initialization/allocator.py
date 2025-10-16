from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
from gt4py import storage as gt_storage

from ndsl.constants import SPATIAL_DIMS
from ndsl.dsl.typing import Float
from ndsl.initialization import GridSizer
from ndsl.quantity import Quantity, QuantityHaloSpec


class StorageNumpy:
    def __init__(self, backend: str) -> None:
        """Initialize an object which behaves like the numpy module, but uses
        gt4py storage objects for zeros, ones, and empty.

        Args:
            backend: gt4py backend
        """
        self.backend = backend

    def empty(self, *args, **kwargs) -> np.ndarray:
        return gt_storage.empty(*args, backend=self.backend, **kwargs)

    def ones(self, *args, **kwargs) -> np.ndarray:
        return gt_storage.ones(*args, backend=self.backend, **kwargs)

    def zeros(self, *args, **kwargs) -> np.ndarray:
        return gt_storage.zeros(*args, backend=self.backend, **kwargs)


class QuantityFactory:
    def __init__(
        self, sizer: GridSizer, numpy, *, silence_deprecation_warning: bool = False
    ) -> None:
        if not silence_deprecation_warning:
            warnings.warn(
                "Usage of QuantityFactory(sizer, numpy) is discouraged and will change "
                "in the next release. Use QuantityFactory.from_backend(sizer, backend) "
                "instead for a stable experience across the release.",
                DeprecationWarning,
                2,
            )
        self.sizer: GridSizer = sizer
        self._numpy = numpy

    def set_extra_dim_lengths(self, **kwargs) -> None:
        """
        Set the length of extra (non-x/y/z) dimensions.
        """
        self.sizer.extra_dim_lengths.update(kwargs)

    @classmethod
    def from_backend(cls, sizer: GridSizer, backend: str) -> QuantityFactory:
        """Initialize a QuantityFactory to use a specific gt4py backend.

        Args:
            sizer: object which determines array sizes
            backend: gt4py backend
        """
        numpy = StorageNumpy(backend)
        # Don't print the deprecation warning in this case
        return cls(sizer, numpy, silence_deprecation_warning=True)

    def _backend(self) -> str | None:
        if isinstance(self._numpy, StorageNumpy):
            return self._numpy.backend

        return None

    def empty(
        self,
        dims: Sequence[str],
        units: str,
        dtype: type = Float,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """Allocate a Quantity - values are random.

        Equivalent to `numpy.empty`"""
        return self._allocate(
            self._numpy.empty, dims, units, dtype, allow_mismatch_float_precision
        )

    def zeros(
        self,
        dims: Sequence[str],
        units: str,
        dtype: type = Float,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """Allocate a Quantity and fill it with the value 0.

        Equivalent to `numpy.zeros`"""
        return self._allocate(
            self._numpy.zeros, dims, units, dtype, allow_mismatch_float_precision
        )

    def ones(
        self,
        dims: Sequence[str],
        units: str,
        dtype: type = Float,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """Allocate a Quantity and fill it with the value 1.

        Equivalent to `numpy.ones`"""
        return self._allocate(
            self._numpy.ones, dims, units, dtype, allow_mismatch_float_precision
        )

    def full(
        self,
        dims: Sequence[str],
        units: str,
        value,  # no type hint because it would be a TypeVar = Type[dtype] and mypy says no
        dtype: type = Float,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """Allocate a Quantity and fill it with the value.

        Equivalent to `numpy.full`"""
        quantity = self._allocate(
            self._numpy.empty, dims, units, dtype, allow_mismatch_float_precision
        )
        quantity.data[:] = value
        return quantity

    def from_array(
        self,
        data: np.ndarray,
        dims: Sequence[str],
        units: str,
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """
        Create a Quantity from a numpy array.

        That numpy array must correspond to the correct shape and extent
        for the given dims.
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
        allow_mismatch_float_precision: bool = False,
    ) -> Quantity:
        """
        Create a Quantity from a numpy array.

        That numpy array must correspond to the correct shape and extent
        of the compute domain for the given dims.
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
            gt4py_backend=self._backend(),
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
