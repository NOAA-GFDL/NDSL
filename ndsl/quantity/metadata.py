import dataclasses
from typing import Any, Dict, Tuple, Union

import numpy as np

from ndsl.optional_imports import cupy
from ndsl.types import NumpyModule


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
