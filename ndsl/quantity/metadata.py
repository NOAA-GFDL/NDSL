from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ndsl.optional_imports import cupy
from ndsl.types import NumpyModule


if cupy is None:
    import numpy as cupy


@dataclasses.dataclass
class QuantityMetadata:
    origin: tuple[int, ...]
    "The start of the computational domain."
    extent: tuple[int, ...]
    "The shape of the computational domain."
    n_halo: int
    "Number of halo-points used in the horizontal."
    dims: tuple[str, ...]
    "Names of each dimension."
    units: str
    "Units of the quantity."
    data_type: type
    "ndarray-like type used to store the data."
    dtype: type
    "dtype of the data in the ndarray-like object."
    backend: str
    "GT4Py backend name. Used for performance optimal data allocation."

    @property
    def dim_lengths(self) -> dict[str, int]:
        """Mapping of dimension names to their lengths."""
        return dict(zip(self.dims, self.extent))

    @property
    def np(self) -> NumpyModule:
        """numpy-like module used to interact with the data."""
        if issubclass(self.data_type, cupy.ndarray):
            return cupy

        if issubclass(self.data_type, np.ndarray):
            return np

        raise TypeError(
            f"Quantity underlying data is of unexpected type {self.data_type}"
        )

    def duplicate_metadata(self, metadata_copy: QuantityMetadata) -> None:
        metadata_copy.origin = self.origin
        metadata_copy.extent = self.extent
        metadata_copy.dims = self.dims
        metadata_copy.units = self.units
        metadata_copy.data_type = self.data_type
        metadata_copy.dtype = self.dtype
        metadata_copy.backend = self.backend


@dataclasses.dataclass
class QuantityHaloSpec:
    """Describe the memory to be exchanged, including size of the halo."""

    n_points: int
    strides: tuple[int]
    itemsize: int
    shape: tuple[int]
    origin: tuple[int, ...]
    extent: tuple[int, ...]
    dims: tuple[str, ...]
    numpy_module: NumpyModule
    dtype: Any
