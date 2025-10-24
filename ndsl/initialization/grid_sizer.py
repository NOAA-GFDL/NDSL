import warnings
from collections.abc import Sequence
from dataclasses import dataclass


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
    data_dimensions: dict[str, int]
    """Name/Lengths pair of any non-x/y/z dimensions, such as land or radiation dimensions."""

    @property
    def extra_dim_lengths(self) -> dict[str, int]:
        warnings.warn(
            "`GridSizer.extra_dim_lengths` is a deprecated API, use `GridSizer.data_dimensions`.",
            DeprecationWarning,
            2,
        )
        return self.data_dimensions

    def get_origin(self, dims: Sequence[str]) -> tuple[int, ...]:
        raise NotImplementedError()

    def get_extent(self, dims: Sequence[str]) -> tuple[int, ...]:
        raise NotImplementedError()

    def get_shape(self, dims: Sequence[str]) -> tuple[int, ...]:
        raise NotImplementedError()
