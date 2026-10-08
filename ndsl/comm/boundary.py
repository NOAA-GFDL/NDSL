import dataclasses
from enum import Enum, auto
from typing import Any

import ndsl.constants as constants
from ndsl.comm._boundary_utils import get_boundary_slice
from ndsl.quantity import Quantity, QuantityHaloSpec


class CommType(Enum):
    """Communication direction represented by a boundary."""

    SYMMETRIC = auto()
    SEND_ONLY = auto()
    RECV_ONLY = auto()


@dataclasses.dataclass
class Boundary:
    """Maps part of a subtile domain to another rank which shares halo points."""

    from_rank: int
    to_rank: int
    n_clockwise_rotations: int
    """
    number of clockwise rotations data undergoes if it moves from the from_rank
    to the to_rank. The same as the number of clockwise rotations to get from the
    orientation of the axes in from_rank to the orientation of the axes in to_rank.
    """
    comm_type: CommType = dataclasses.field(
        default=CommType.SYMMETRIC,
        kw_only=True,
    )

    def does_send(self) -> bool:
        return self.comm_type in (
            CommType.SYMMETRIC,
            CommType.SEND_ONLY,
        )

    def does_recv(self) -> bool:
        return self.comm_type in (
            CommType.SYMMETRIC,
            CommType.RECV_ONLY,
        )

    def send_view(self, quantity: Quantity, n_points: int) -> Any:
        """Return a sliced view of points which should be sent at this boundary.

        Args:
            quantity: quantity for which to return a slice
            n_points: the width of boundary to include
        """
        return self._view(quantity, n_points, interior=True)

    def recv_view(self, quantity: Quantity, n_points: int) -> Any:
        """Return a sliced view of points which should be received at this boundary.

        Args:
            quantity: quantity for which to return a slice
            n_points: the width of boundary to include
        """
        return self._view(quantity, n_points, interior=False)

    def send_slice(self, specification: QuantityHaloSpec) -> tuple[slice, ...]:
        """Return the index slices which should be sent at this boundary.

        Args:
            specification: data specifications for the halo. Including shape
                and number of halo points.

        Returns:
            A tuple of slices (one per dimensions)
        """
        return self._slice(specification, interior=True)

    def recv_slice(self, specification: QuantityHaloSpec) -> tuple[slice, ...]:
        """Return the index slices which should be received at this boundary.

        Args:
            specification: data specifications for the halo. Including shape
                and number of halo points.

        Returns:
            A tuple of slices (one per dimensions)
        """
        return self._slice(specification, interior=False)

    def _slice(
        self, specification: QuantityHaloSpec, interior: bool
    ) -> tuple[slice, ...]:
        """Returns a tuple of slices (one per dimensions) indexing the data to be exchange.

        Args:
            specification: memory information on this halo, including halo size

        Return:
            A tuple of slices (one per dimensions)
        """
        raise NotImplementedError("Boundary._slice()")

    def _view(self, quantity: Quantity, n_points: int, interior: bool) -> Any:
        """Return a sliced view of points in the given quantity at this boundary.

        Args:
            quantity: quantity for which to return a slice
            n_points: the width of boundary to include
            interior: if True, give points inside the computational domain (default),
                otherwise give points in the halo
        """
        raise NotImplementedError("Boundary._view()")


@dataclasses.dataclass
class SimpleBoundary(Boundary):
    """A boundary representing an edge or corner of a subtile."""

    boundary_type: int

    def _view(self, quantity: Quantity, n_points: int, interior: bool) -> Any:
        boundary_slice = get_boundary_slice(
            quantity.dims,
            quantity.origin,
            quantity.extent,
            quantity.shape,
            self.boundary_type,
            n_points,
            interior,
        )
        return quantity[tuple(boundary_slice)]

    def _slice(
        self, specification: QuantityHaloSpec, interior: bool
    ) -> tuple[slice, ...]:
        return get_boundary_slice(
            specification.dims,
            specification.origin,
            specification.extent,
            specification.shape,
            self.boundary_type,
            specification.n_points,
            interior,
        )


@dataclasses.dataclass
class NestedBoundary(Boundary):
    """A boundary defined by an explicit horizontal data window.

    window_start is relative to the Quantity compute-domain origin,
    so negative offsets may address allocated halo storage.
    window_extent gives the size of the exchanged horizontal region.

    The explicit window replaces the edge/corner geometry used by
    SimpleBoundary. Non-horizontal dimensions span the Quantity's
    compute domain.
    """

    window_start: tuple[int, int]
    window_extent: tuple[int, int]

    def _view(
        self,
        quantity: Quantity,
        n_points: int,
        interior: bool,
    ) -> Any:
        # NestedBoundary windows fully define the exchanged region.
        boundary_slice = self._slice_from_fields(
            quantity.dims,
            quantity.origin,
            quantity.extent,
            quantity.shape,
        )
        return quantity[boundary_slice]

    def _slice(
        self,
        specification: QuantityHaloSpec,
        interior: bool,
    ) -> tuple[slice, ...]:
        return self._slice_from_fields(
            specification.dims,
            specification.origin,
            specification.extent,
            specification.shape,
        )

    @staticmethod
    def _horizontal_dimension_indices(
        dims: tuple[str, ...],
    ) -> tuple[int, int]:
        i_indices = [index for index, dim in enumerate(dims) if dim in constants.I_DIMS]
        j_indices = [index for index, dim in enumerate(dims) if dim in constants.J_DIMS]

        if len(i_indices) != 1 or len(j_indices) != 1:
            raise ValueError(
                "NestedBoundary requires exactly one I dimension and one J dimension, "
                f"got dims={dims}"
            )

        return i_indices[0], j_indices[0]

    def _slice_from_fields(
        self,
        dims: tuple[str, ...],
        origin: tuple[int, ...],
        extent: tuple[int, ...],
        shape: tuple[int, ...],
    ) -> tuple[slice, ...]:
        i_dim, j_dim = self._horizontal_dimension_indices(dims)

        result = [slice(start, start + size) for start, size in zip(origin, extent)]

        i_start = origin[i_dim] + self.window_start[0]
        j_start = origin[j_dim] + self.window_start[1]
        i_stop = i_start + self.window_extent[0]
        j_stop = j_start + self.window_extent[1]

        if i_start < 0 or j_start < 0 or i_stop > shape[i_dim] or j_stop > shape[j_dim]:
            raise ValueError(
                "NestedBoundary window lies outside the allocated Quantity."
            )

        result[i_dim] = slice(i_start, i_stop)
        result[j_dim] = slice(j_start, j_stop)

        return tuple(result)
