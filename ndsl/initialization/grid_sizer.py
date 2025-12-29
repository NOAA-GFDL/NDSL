from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class GridSizer(ABC):
    nx: int
    """Length of the x compute dimension for produced arrays."""
    ny: int
    """Length of the y compute dimension for produced arrays."""
    nz: int
    """Length of the z compute dimension for produced arrays."""
    n_halo: int
    """Number of horizontal halo points for produced arrays."""
    data_dimensions: dict[str, int]
    """Name/Lengths pair of any non-x/y/z dimensions, such as land or radiation dimensions."""
    internal_pad_non_interface_dimensions: bool = False
    """Enforce interface and non-interface dimensions true memory to be the same"""

    @abstractmethod
    def get_origin(self, dims: Sequence[str]) -> tuple[int, ...]: ...

    @abstractmethod
    def get_extent(self, dims: Sequence[str]) -> tuple[int, ...]: ...

    @abstractmethod
    def get_shape(self, dims: Sequence[str]) -> tuple[int, ...]: ...
