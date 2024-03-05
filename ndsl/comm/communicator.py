from typing import List, Optional, Sequence, Tuple, Union, cast

import ndsl.constants as constants
from ndsl.comm.partitioner import CubedSpherePartitioner, TilePartitioner
from ndsl.halo.updater import HaloUpdater, HaloUpdateRequest
from ndsl.performance.timer import Timer
from ndsl.quantity import Quantity, QuantityMetadata
from ndsl.typing import Communicator


try:
    import cupy
except ImportError:
    cupy = None


def bcast_metadata_list(comm, quantity_list):
    is_root = comm.Get_rank() == constants.ROOT_RANK
    if is_root:
        metadata_list = []
        for quantity in quantity_list:
            metadata_list.append(quantity.metadata)
    else:
        metadata_list = None
    return comm.bcast(metadata_list, root=constants.ROOT_RANK)


def bcast_metadata(comm, array):
    return bcast_metadata_list(comm, [array])[0]


class TileCommunicator(Communicator):
    """Performs communications within a single tile or region of a tile"""

    def __init__(
        self,
        comm,
        partitioner: TilePartitioner,
        force_cpu: bool = False,
        timer: Optional[Timer] = None,
    ):
        """Initialize a TileCommunicator.

        Args:
            comm: communication object behaving like mpi4py.Comm
            partitioner: tile partitioner
            force_cpu: force all communication to go through central memory
            timer: Time communication operations.
        """
        super(TileCommunicator, self).__init__(
            comm, partitioner, force_cpu=force_cpu, timer=timer
        )
        self.partitioner: TilePartitioner = partitioner

    @classmethod
    def from_layout(
        cls,
        comm,
        layout: Tuple[int, int],
        force_cpu: bool = False,
        timer: Optional[Timer] = None,
    ) -> "TileCommunicator":
        partitioner = TilePartitioner(layout=layout)
        return cls(comm=comm, partitioner=partitioner, force_cpu=force_cpu, timer=timer)

    @property
    def tile(self):
        return self

    def start_halo_update(
        self, quantity: Union[Quantity, List[Quantity]], n_points: int
    ) -> HaloUpdater:
        """Start an asynchronous halo update on a quantity.

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if self.partitioner.layout[0] < 3 or self.partitioner.layout[1] < 3:
            raise NotImplementedError(
                "implementing halo updates on smaller layouts requires "
                "refactoring our code to remove the assumption that any pair "
                "of ranks only share one boundary"
            )
        else:
            return super().start_halo_update(quantity, n_points)

    def start_vector_halo_update(
        self,
        x_quantity: Union[Quantity, List[Quantity]],
        y_quantity: Union[Quantity, List[Quantity]],
        n_points: int,
    ) -> HaloUpdater:
        """Start an asynchronous halo update of a horizontal vector quantity.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if self.partitioner.layout[0] < 3 or self.partitioner.layout[1] < 3:
            raise NotImplementedError(
                "implementing halo updates on smaller layouts requires "
                "refactoring our code to remove the assumption that any pair "
                "of ranks only share one boundary"
            )
        else:
            return super().start_vector_halo_update(x_quantity, y_quantity, n_points)

    def start_synchronize_vector_interfaces(
        self, x_quantity: Quantity, y_quantity: Quantity
    ) -> HaloUpdateRequest:
        """
        Synchronize shared points at the edges of a vector interface variable.

        Sends the values on the south and west edges to overwrite the values on adjacent
        subtiles. Vector must be defined on the Arakawa C grid.

        For interface variables, the edges of the tile are computed on both ranks
        bordering that edge. This routine copies values across those shared edges
        so that both ranks have the same value for that edge. It also handles any
        rotation of vector quantities needed to move data across the edge.

        Args:
            x_quantity: the x-component quantity to be synchronized
            y_quantity: the y-component quantity to be synchronized

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if self.partitioner.layout[0] < 3 or self.partitioner.layout[1] < 3:
            raise NotImplementedError(
                "implementing halo updates on smaller layouts requires "
                "refactoring our code to remove the assumption that any pair "
                "of ranks only share one boundary"
            )
        else:
            return super().start_synchronize_vector_interfaces(x_quantity, y_quantity)


class CubedSphereCommunicator(Communicator):
    """Performs communications within a cubed sphere"""

    timer: Timer
    partitioner: CubedSpherePartitioner

    def __init__(
        self,
        comm,
        partitioner: CubedSpherePartitioner,
        force_cpu: bool = False,
        timer: Optional[Timer] = None,
    ):
        """Initialize a CubedSphereCommunicator.

        Args:
            comm: mpi4py.Comm object
            partitioner: cubed sphere partitioner
            force_cpu: Force all communication to go through central memory.
            timer: Time communication operations.
        """
        if comm.Get_size() != partitioner.total_ranks:
            raise ValueError(
                f"was given a partitioner for {partitioner.total_ranks} ranks but a "
                f"comm object with only {comm.Get_size()} ranks, are we running "
                "with mpi and the correct number of ranks?"
            )
        self._tile_communicator: Optional[TileCommunicator] = None
        self._force_cpu = force_cpu
        super(CubedSphereCommunicator, self).__init__(
            comm, partitioner, force_cpu, timer
        )
        self.partitioner: CubedSpherePartitioner = partitioner

    @classmethod
    def from_layout(
        cls,
        comm,
        layout: Tuple[int, int],
        force_cpu: bool = False,
        timer: Optional[Timer] = None,
    ) -> "CubedSphereCommunicator":
        partitioner = CubedSpherePartitioner(tile=TilePartitioner(layout=layout))
        return cls(comm=comm, partitioner=partitioner, force_cpu=force_cpu, timer=timer)

    @property
    def tile(self) -> TileCommunicator:
        """communicator for within a tile"""
        if self._tile_communicator is None:
            self._initialize_tile_communicator()
        return cast(TileCommunicator, self._tile_communicator)

    def _initialize_tile_communicator(self):
        tile_comm = self.comm.Split(
            color=self.partitioner.tile_index(self.rank), key=self.rank
        )
        self._tile_communicator = TileCommunicator(tile_comm, self.partitioner.tile)

    def _get_gather_recv_quantity(
        self, global_extent: Sequence[int], metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving global data during gather

        Args:
            shape: ndarray shape, numpy-style
            metadata: metadata to the created Quantity
        """
        # needs to change the quantity dimensions since we add a "tile" dimension,
        # unlike for tile scatter/gather which retains the same dimensions
        recv_quantity = Quantity(
            metadata.np.zeros(global_extent, dtype=metadata.dtype),
            dims=(constants.TILE_DIM,) + metadata.dims,
            units=metadata.units,
            origin=(0,) + tuple([0 for dim in metadata.dims]),
            extent=global_extent,
            gt4py_backend=metadata.gt4py_backend,
            allow_mismatch_float_precision=True,
        )
        return recv_quantity

    def _get_scatter_recv_quantity(
        self, shape: Sequence[int], metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving subtile data during scatter

        Args:
            shape: ndarray shape, numpy-style
            metadata: metadata to the created Quantity
        """
        # needs to change the quantity dimensions since we remove a "tile" dimension,
        # unlike for tile scatter/gather which retains the same dimensions
        recv_quantity = Quantity(
            metadata.np.zeros(shape, dtype=metadata.dtype),
            dims=metadata.dims[1:],
            units=metadata.units,
            gt4py_backend=metadata.gt4py_backend,
            allow_mismatch_float_precision=True,
        )
        return recv_quantity
