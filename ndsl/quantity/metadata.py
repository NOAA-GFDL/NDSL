import dataclasses
from typing import Any, Dict, Tuple, Union

import numpy as np

from ndsl.optional_imports import cupy
from ndsl.types import NumpyModule


if cupy is None:
    import numpy as cupy


@dataclasses.dataclass
class QuantityMetadata:
    origin: Tuple[int, ...]
    "the start of the computational domain"
    extent: Tuple[int, ...]
    "the shape of the computational domain"
    dims: Tuple[str, ...]
    "names of each dimension"
    units: str
    "units of the quantity"
    data_type: type
    "ndarray-like type used to store the data"
    dtype: type
    "dtype of the data in the ndarray-like object"
    gt4py_backend: Union[str, None] = None
    "backend to use for gt4py storages"

    @property
    def dim_lengths(self) -> Dict[str, int]:
        """mapping of dimension names to their lengths"""
        return dict(zip(self.dims, self.extent))

    @property
    def np(self) -> NumpyModule:
        """numpy-like module used to interact with the data"""
        if issubclass(self.data_type, cupy.ndarray):
            return cupy
        elif issubclass(self.data_type, np.ndarray):
            return np
        else:
            raise TypeError(
                f"quantity underlying data is of unexpected type {self.data_type}"
            )

    def duplicate_metadata(self, metadata_copy):
        metadata_copy.origin = self.origin
        metadata_copy.extent = self.extent
        metadata_copy.dims = self.dims
        metadata_copy.units = self.units
        metadata_copy.data_type = self.data_type
        metadata_copy.dtype = self.dtype
        metadata_copy.gt4py_backend = self.gt4py_backend


@dataclasses.dataclass
class QuantityHaloSpec:
    """Describe the memory to be exchanged, including size of the halo."""

    n_points: int
    strides: Tuple[int]
    itemsize: int
    shape: Tuple[int]
    origin: Tuple[int, ...]
    extent: Tuple[int, ...]
    dims: Tuple[str, ...]
    numpy_module: NumpyModule
    dtype: Any
