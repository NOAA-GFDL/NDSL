from dataclasses import dataclass
from typing import Sequence


@dataclass
class GridSizer:
    nx: int
    """Length of the x compute dimension for produced arrays."""
    ny: int
    """Length of the y compute dimension for produced arrays."""
    nz: int
    """Length of the z compute dimension for produced arrays."""
    n_halo: int
    """Number of horizontal halo points for produced arrays."""
    extra_dim_lengths: dict[str, int]
    """Lengths of any non-x/y/z dimensions, such as land or radiation dimensions."""

    def get_origin(self, dims: Sequence[str]) -> tuple[int, ...]:
        raise NotImplementedError()

    def get_extent(self, dims: Sequence[str]) -> tuple[int, ...]:
        raise NotImplementedError()

    def get_shape(self, dims: Sequence[str]) -> tuple[int, ...]:
        raise NotImplementedError()
