from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any, Generic, Self, TypeVar, cast

import numpy as np

import ndsl.constants as constants
from ndsl.buffer import array_buffer, device_synchronize, recv_buffer, send_buffer
from ndsl.comm.boundary import Boundary
from ndsl.comm.comm_abc import Comm as CommABC
from ndsl.comm.comm_abc import ReductionOperator
from ndsl.comm.partitioner import (
    CubedSpherePartitioner,
    NestedPartitioner,
    Partitioner,
    TilePartitioner,
)
from ndsl.halo.updater import HaloUpdater, HaloUpdateRequest, VectorInterfaceHaloUpdater
from ndsl.optional_imports import cupy
from ndsl.performance.timer import NullTimer, Timer
from ndsl.quantity import Quantity, QuantityHaloSpec, QuantityMetadata


def to_numpy(array, dtype=None) -> np.ndarray:  # type: ignore[no-untyped-def]
    """
    Input array can be a numpy array or a cupy array. Returns numpy array.
    """
    try:
        output = np.asarray(array)
    except ValueError as err:
        if err.args[0] == "object __array__ method not producing an array":
            output = cupy.asnumpy(array)
        else:
            raise err
    except TypeError as err:
        if err.args[0].startswith(
            "Implicit conversion to a NumPy array is not allowed."
        ):
            output = cupy.asnumpy(array)
        else:
            raise err
    if dtype:
        output = output.astype(dtype=dtype)
    return output


P = TypeVar("P", bound=Partitioner)


class Communicator(abc.ABC, Generic[P]):
    def __init__(
        self,
        comm: CommABC,
        partitioner: P,
        force_cpu: bool = False,
        timer: Timer | None = None,
    ):
        self.comm = comm
        self.partitioner: P = partitioner
        self._force_cpu = force_cpu
        self._boundaries: Mapping[int, Boundary] | None = None
        self._last_halo_tag = 0
        self.timer: Timer = timer if timer is not None else NullTimer()

    @property
    @abc.abstractmethod
    def tile(self) -> TileCommunicator:
        pass

    @classmethod
    @abc.abstractmethod
    def from_layout(
        cls,
        comm: CommABC,
        layout: tuple[int, int],
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> Self:
        pass

    @property
    def rank(self) -> int:
        """Rank of the current process within this communicator."""
        return self.comm.Get_rank()

    @property
    def size(self) -> int:
        """Total number of ranks in this communicator."""
        return self.comm.Get_size()

    def _maybe_force_cpu(self, module: ModuleType) -> ModuleType:
        """
        Get a numpy-like module depending on configuration and
        Quantity original allocator.
        """
        if self._force_cpu:
            return np
        return module

    @staticmethod
    def _device_synchronize() -> None:
        """Wait for all work that could be in-flight to finish."""
        # this is a method so we can profile it separately from other device syncs
        device_synchronize()

    def _create_all_reduce_quantity(
        self, input_metadata: QuantityMetadata, input_data: Any
    ) -> Quantity:
        """Create a Quantity for all_reduce data and metadata."""
        all_reduce_quantity = Quantity(
            input_data,
            dims=input_metadata.dims,
            units=input_metadata.units,
            origin=input_metadata.origin,
            extent=input_metadata.extent,
            backend=input_metadata.backend,
            allow_mismatch_float_precision=False,
        )
        return all_reduce_quantity

    def all_reduce(
        self,
        input_quantity: Quantity,
        op: ReductionOperator,
        output_quantity: Quantity | None = None,
    ) -> Quantity:
        reduced_quantity_data = self.comm.allreduce(input_quantity._data, op)
        if output_quantity is None:
            return self._create_all_reduce_quantity(
                input_quantity.metadata, reduced_quantity_data
            )

        if output_quantity.shape != input_quantity.shape:
            raise TypeError("Shapes not matching")

        input_quantity.metadata.duplicate_metadata(output_quantity.metadata)

        output_quantity[:] = reduced_quantity_data[:]
        return output_quantity

    def all_reduce_per_element(
        self,
        input_quantity: Quantity,
        output_quantity: Quantity,
        op: ReductionOperator,
    ) -> None:
        self.comm.Allreduce(input_quantity._data, output_quantity._data, op)

    def all_reduce_per_element_in_place(
        self, quantity: Quantity, op: ReductionOperator
    ) -> None:
        # Note that device_synchronization is Cupy/Cuda specific
        # at the moment.
        device_synchronize()
        self.comm.Allreduce_inplace(quantity._data, op)

    def _Scatter(self, numpy_module, sendbuf, recvbuf, **kwargs):  # type: ignore[no-untyped-def]
        with send_buffer(numpy_module.zeros, sendbuf) as send:
            with recv_buffer(numpy_module.zeros, recvbuf) as recv:
                self.comm.Scatter(send, recv, **kwargs)

    def _Gather(self, numpy_module, sendbuf, recvbuf, **kwargs):  # type: ignore[no-untyped-def]
        with send_buffer(numpy_module.zeros, sendbuf) as send:
            with recv_buffer(numpy_module.zeros, recvbuf) as recv:
                self.comm.Gather(send, recv, **kwargs)

    def scatter(
        self,
        send_quantity: Quantity | None = None,
        recv_quantity: Quantity | None = None,
    ) -> Quantity:
        """Transfer subtile regions of a full-tile quantity
        from the tile root rank to all subtiles.

        Args:
            send_quantity: quantity to send, only required/used on the tile root rank
            recv_quantity: if provided, assign received data into this Quantity.
        Returns:
            recv_quantity
        """
        if self.rank == constants.ROOT_RANK and send_quantity is None:
            raise TypeError("send_quantity is a required argument on the root rank")
        if self.rank == constants.ROOT_RANK:
            send_quantity = cast(Quantity, send_quantity)
            metadata: QuantityMetadata = self.comm.bcast(
                send_quantity.metadata, root=constants.ROOT_RANK
            )  # type: ignore[assignment]
        else:
            metadata = self.comm.bcast(None, root=constants.ROOT_RANK)  # type: ignore[assignment]
        shape = self.partitioner.subtile_extent(metadata, self.rank)
        if recv_quantity is None:
            recv_quantity = self._get_scatter_recv_quantity(shape, metadata)
        if self.rank == constants.ROOT_RANK:
            send_quantity = cast(Quantity, send_quantity)
            with array_buffer(
                self._maybe_force_cpu(metadata.np).zeros,
                (self.partitioner.total_ranks,) + shape,
                dtype=metadata.dtype,
            ) as sendbuf:
                for rank in range(0, self.partitioner.total_ranks):
                    subtile_slice = self.partitioner.subtile_slice(
                        rank=rank,
                        global_dims=metadata.dims,
                        global_extent=metadata.extent,
                        overlap=True,
                    )
                    sendbuf.assign_from(
                        send_quantity.view[subtile_slice],
                        buffer_slice=np.index_exp[rank, :],
                    )
                self._Scatter(
                    metadata.np,
                    sendbuf.array,
                    recv_quantity.view[:],
                    root=constants.ROOT_RANK,
                )
        else:
            self._Scatter(
                metadata.np,
                None,
                recv_quantity.view[:],
                root=constants.ROOT_RANK,
            )
        return recv_quantity

    def _get_gather_recv_quantity(
        self, global_extent: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving global data during gather"""
        recv_quantity = Quantity(
            send_metadata.np.zeros(global_extent, dtype=send_metadata.dtype),
            dims=send_metadata.dims,
            units=send_metadata.units,
            origin=tuple([0 for dim in send_metadata.dims]),
            extent=global_extent,
            backend=send_metadata.backend,
            allow_mismatch_float_precision=True,
        )
        return recv_quantity

    def _get_scatter_recv_quantity(
        self, shape: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving subtile data during scatter"""
        recv_quantity = Quantity(
            send_metadata.np.zeros(shape, dtype=send_metadata.dtype),
            dims=send_metadata.dims,
            units=send_metadata.units,
            backend=send_metadata.backend,
            allow_mismatch_float_precision=True,
        )
        return recv_quantity

    def gather(
        self, send_quantity: Quantity, recv_quantity: Quantity | None = None
    ) -> Quantity | None:
        """Transfer subtile regions of a full-tile quantity
        from each rank to the tile root rank.

        Args:
            send_quantity: quantity to send
            recv_quantity: if provided, assign received data into this Quantity (only
                used on the tile root rank)
        Returns:
            recv_quantity: quantity if on root rank, otherwise None
        """
        result: Quantity | None
        if self.rank == constants.ROOT_RANK:
            with array_buffer(
                send_quantity.np.zeros,
                (self.partitioner.total_ranks,) + tuple(send_quantity.extent),
                dtype=send_quantity.dtype,
            ) as recvbuf:
                self._Gather(
                    send_quantity.np,
                    send_quantity.view[:],
                    recvbuf.array,
                    root=constants.ROOT_RANK,
                )
                if recv_quantity is None:
                    global_extent = self.partitioner.global_extent(
                        send_quantity.metadata
                    )
                    recv_quantity = self._get_gather_recv_quantity(
                        global_extent, send_quantity.metadata
                    )
                for rank in range(self.partitioner.total_ranks):
                    to_slice = self.partitioner.subtile_slice(
                        rank=rank,
                        global_dims=recv_quantity.dims,
                        global_extent=recv_quantity.extent,
                        overlap=True,
                    )
                    recvbuf.assign_to(
                        recv_quantity.view[to_slice], buffer_slice=np.index_exp[rank, :]
                    )
                result = recv_quantity
        else:
            self._Gather(
                send_quantity.np,
                send_quantity.view[:],
                None,
                root=constants.ROOT_RANK,
            )
            result = None
        return result

    def gather_state(self, send_state=None, recv_state=None, transfer_type=None):  # type: ignore[no-untyped-def]
        """Transfer a state dictionary from subtile ranks to the tile root rank.

        'time' is assumed to be the same on all ranks, and its value will be set
        to the value from the root rank.

        Args:
            send_state: the model state to be sent containing the subtile data
            recv_state: the pre-allocated state in which to receive the full tile
                state. Only variables which are scattered will be written to.
        Returns:
            recv_state: on the root rank, the state containing the entire tile
        """
        if self.rank == constants.ROOT_RANK and recv_state is None:
            recv_state = {}
        for name, quantity in send_state.items():
            if name == "time":
                if self.rank == constants.ROOT_RANK:
                    recv_state["time"] = send_state["time"]
            else:
                gather_value = to_numpy(quantity.view[:], dtype=transfer_type)
                gather_quantity = Quantity(
                    data=gather_value,
                    dims=quantity.dims,
                    units=quantity.units,
                    allow_mismatch_float_precision=True,
                    backend=quantity.backend,
                )
                if recv_state is not None and name in recv_state:
                    tile_quantity = self.gather(
                        gather_quantity, recv_quantity=recv_state[name]
                    )
                else:
                    tile_quantity = self.gather(gather_quantity)
                if self.rank == constants.ROOT_RANK:
                    recv_state[name] = tile_quantity
                del gather_quantity
        return recv_state

    def scatter_state(self, send_state=None, recv_state=None):  # type: ignore[no-untyped-def]
        """Transfer a state dictionary from the tile root rank to all subtiles.

        Args:
            send_state: the model state to be sent containing the entire tile,
                required only from the root rank
            recv_state: the pre-allocated state in which to receive the scattered
                state. Only variables which are scattered will be written to.
        Returns:
            rank_state: the state corresponding to this rank's subdomain
        """

        def scatter_root() -> None:
            if send_state is None:
                raise TypeError("send_state is a required argument on the root rank")
            name_list = list(send_state.keys())
            while "time" in name_list:
                name_list.remove("time")
            name_list = self.comm.bcast(name_list, root=constants.ROOT_RANK)  # type: ignore[assignment]
            array_list = [send_state[name] for name in name_list]
            for name, array in zip(name_list, array_list):
                if name in recv_state:
                    self.scatter(send_quantity=array, recv_quantity=recv_state[name])
                else:
                    recv_state[name] = self.scatter(send_quantity=array)
            recv_state["time"] = self.comm.bcast(
                send_state.get("time", None), root=constants.ROOT_RANK
            )

        def scatter_client() -> None:
            name_list = self.comm.bcast(None, root=constants.ROOT_RANK)
            for name in name_list:  # type: ignore
                if name in recv_state:
                    self.scatter(recv_quantity=recv_state[name])
                else:
                    recv_state[name] = self.scatter()
            recv_state["time"] = self.comm.bcast(None, root=constants.ROOT_RANK)

        if recv_state is None:
            recv_state = {}
        if self.rank == constants.ROOT_RANK:
            scatter_root()
        else:
            scatter_client()
        if recv_state["time"] is None:
            recv_state.pop("time")
        return recv_state

    def halo_update(self, quantity: Quantity | list[Quantity], n_points: int) -> None:
        """Perform a halo update on a quantity or quantities

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior
        """
        if isinstance(quantity, Quantity):
            quantities = [quantity]
        else:
            quantities = quantity

        halo_updater = self.start_halo_update(quantities, n_points)
        halo_updater.wait()

    def start_halo_update(
        self, quantity: Quantity | list[Quantity], n_points: int
    ) -> HaloUpdater:
        """Start an asynchronous halo update on a quantity.

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if isinstance(quantity, Quantity):
            quantities = [quantity]
        else:
            quantities = quantity

        specifications = []
        for quantity in quantities:
            specification = QuantityHaloSpec(
                n_points=n_points,
                shape=quantity.shape,
                strides=quantity._data.strides,
                itemsize=quantity._data.itemsize,
                origin=quantity.origin,
                extent=quantity.extent,
                dims=quantity.dims,
                numpy_module=self._maybe_force_cpu(quantity.np),
                dtype=quantity.metadata.dtype,
            )
            specifications.append(specification)

        halo_updater = self.get_scalar_halo_updater(specifications)
        halo_updater.force_finalize_on_wait()
        halo_updater.start(quantities)
        return halo_updater

    def vector_halo_update(
        self,
        x_quantity: Quantity | list[Quantity],
        y_quantity: Quantity | list[Quantity],
        n_points: int,
    ) -> None:
        """Perform a halo update of a horizontal vector quantity or quantities.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior
        """
        if isinstance(x_quantity, Quantity):
            x_quantities = [x_quantity]
        else:
            x_quantities = x_quantity
        if isinstance(y_quantity, Quantity):
            y_quantities = [y_quantity]
        else:
            y_quantities = y_quantity

        halo_updater = self.start_vector_halo_update(
            x_quantities, y_quantities, n_points
        )
        halo_updater.wait()

    def start_vector_halo_update(
        self,
        x_quantity: Quantity | list[Quantity],
        y_quantity: Quantity | list[Quantity],
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
        if isinstance(x_quantity, Quantity):
            x_quantities = [x_quantity]
        else:
            x_quantities = x_quantity
        if isinstance(y_quantity, Quantity):
            y_quantities = [y_quantity]
        else:
            y_quantities = y_quantity

        x_specifications = []
        y_specifications = []
        for x_quantity, y_quantity in zip(x_quantities, y_quantities):
            x_specification = QuantityHaloSpec(
                n_points=n_points,
                shape=x_quantity.shape,
                strides=x_quantity._data.strides,
                itemsize=x_quantity._data.itemsize,
                origin=x_quantity.metadata.origin,
                extent=x_quantity.metadata.extent,
                dims=x_quantity.metadata.dims,
                numpy_module=self._maybe_force_cpu(x_quantity.np),
                dtype=x_quantity.metadata.dtype,
            )
            x_specifications.append(x_specification)
            y_specification = QuantityHaloSpec(
                n_points=n_points,
                shape=y_quantity.shape,
                strides=y_quantity._data.strides,
                itemsize=y_quantity._data.itemsize,
                origin=y_quantity.metadata.origin,
                extent=y_quantity.metadata.extent,
                dims=y_quantity.metadata.dims,
                numpy_module=self._maybe_force_cpu(y_quantity.np),
                dtype=y_quantity.metadata.dtype,
            )
            y_specifications.append(y_specification)

        halo_updater = self.get_vector_halo_updater(x_specifications, y_specifications)
        halo_updater.force_finalize_on_wait()
        halo_updater.start(x_quantities, y_quantities)
        return halo_updater

    def synchronize_vector_interfaces(
        self, x_quantity: Quantity, y_quantity: Quantity
    ) -> None:
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
        """
        req = self.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        req.wait()

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
        halo_updater = VectorInterfaceHaloUpdater(
            comm=self.comm,
            boundaries=self.boundaries,
            force_cpu=self._force_cpu,
            timer=self.timer,
        )
        req = halo_updater.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        return req

    def get_scalar_halo_updater(
        self, specifications: list[QuantityHaloSpec]
    ) -> HaloUpdater:
        if len(specifications) == 0:
            raise RuntimeError("Cannot create updater with specifications list")
        if specifications[0].n_points == 0:
            raise ValueError("cannot perform a halo update on zero halo points")
        return HaloUpdater.from_scalar_specifications(
            self,
            self._maybe_force_cpu(specifications[0].numpy_module),
            specifications,
            self.boundaries.values(),
            self._get_halo_tag(),
            self.timer,
        )

    def get_vector_halo_updater(
        self,
        specifications_x: list[QuantityHaloSpec],
        specifications_y: list[QuantityHaloSpec],
    ) -> HaloUpdater:
        if len(specifications_x) == 0 and len(specifications_y) == 0:
            raise RuntimeError("Cannot create updater with empty specifications list")
        if specifications_x[0].n_points == 0 and specifications_y[0].n_points == 0:
            raise ValueError("Cannot perform a halo update on zero halo points")
        return HaloUpdater.from_vector_specifications(
            self,
            self._maybe_force_cpu(specifications_x[0].numpy_module),
            specifications_x,
            specifications_y,
            self.boundaries.values(),
            self._get_halo_tag(),
            self.timer,
        )

    def _get_halo_tag(self) -> int:
        self._last_halo_tag += 1
        return self._last_halo_tag

    @property
    def boundaries(self) -> Mapping[int, Boundary]:
        """boundaries of this tile with neighboring tiles"""
        if self._boundaries is None:
            self._boundaries = {}
            for boundary_type in constants.BOUNDARY_TYPES:
                boundary = self.partitioner.boundary(boundary_type, self.rank)
                if boundary is not None:
                    self._boundaries[boundary_type] = boundary
        return self._boundaries


def bcast_metadata_list(comm: CommABC, quantity_list: list[Quantity]):  # type: ignore[no-untyped-def]
    is_root = comm.Get_rank() == constants.ROOT_RANK
    if is_root:
        metadata_list = []
        for quantity in quantity_list:
            metadata_list.append(quantity.metadata)
    else:
        metadata_list = None
    return comm.bcast(metadata_list, root=constants.ROOT_RANK)


def bcast_metadata(comm: CommABC, array: Quantity):  # type: ignore[no-untyped-def]
    return bcast_metadata_list(comm, [array])[0]


class TileCommunicator(Communicator[TilePartitioner]):
    """Performs communications within a single tile or region of a tile."""

    def __init__(
        self,
        comm: CommABC,
        partitioner: TilePartitioner,
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> None:
        """Initialize a TileCommunicator.

        Args:
            comm: communication object behaving like mpi4py.Comm
            partitioner: tile partitioner
            force_cpu: force all communication to go through central memory
            timer: Time communication operations.
        """
        super().__init__(comm, partitioner, force_cpu, timer)

    @classmethod
    def from_layout(
        cls,
        comm: CommABC,
        layout: tuple[int, int],
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> TileCommunicator:
        return cls(comm, TilePartitioner(layout=layout), force_cpu, timer)

    @property
    def tile(self) -> TileCommunicator:
        return self

    def start_halo_update(
        self, quantity: Quantity | list[Quantity], n_points: int
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

        return super().start_halo_update(quantity, n_points)

    def start_vector_halo_update(
        self,
        x_quantity: Quantity | list[Quantity],
        y_quantity: Quantity | list[Quantity],
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

        return super().start_synchronize_vector_interfaces(x_quantity, y_quantity)


class CubedSphereCommunicator(Communicator[CubedSpherePartitioner]):
    """Performs communications within a cubed sphere."""

    _tile_communicator: TileCommunicator | None

    def __init__(
        self,
        comm: CommABC,
        partitioner: CubedSpherePartitioner,
        force_cpu: bool = False,
        timer: Timer | None = None,
    ):
        """Initialize a CubedSphereCommunicator.

        Args:
            comm: mpi4py.Comm object
            partitioner: cubed sphere partitioner
            force_cpu: Force all communication to go through central memory.
            timer: Time communication operations.
        """
        if not issubclass(type(comm), CommABC):
            raise TypeError(
                "Communicator needs to be instantiated with communication subsystem"
                f" derived from `comm_abc.Comm`, got {type(comm)}."
            )
        if comm.Get_size() < partitioner.total_ranks:
            raise ValueError(
                f"was given a partitioner for {partitioner.total_ranks} ranks but a "
                f"comm object with only {comm.Get_size()} ranks, are we running "
                "with mpi and the correct number of ranks?"
            )

        super().__init__(comm, partitioner, force_cpu, timer)
        self._tile_communicator = None

    @classmethod
    def from_layout(
        cls,
        comm: CommABC,
        layout: tuple[int, int],
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> CubedSphereCommunicator:
        partitioner = CubedSpherePartitioner(tile=TilePartitioner(layout=layout))
        return cls(comm=comm, partitioner=partitioner, force_cpu=force_cpu, timer=timer)

    @property
    def tile(self) -> TileCommunicator:
        """Communicator for within a tile."""
        if self._tile_communicator is None:
            tile_comm = self.comm.Split(
                color=self.partitioner.tile_index(self.rank), key=self.rank
            )
            self._tile_communicator = TileCommunicator(tile_comm, self.partitioner.tile)

        return self._tile_communicator

    def _get_gather_recv_quantity(
        self, global_extent: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving global data during gather.

        Args:
            shape: ndarray shape, numpy-style
            send_metadata: metadata to the created Quantity
        """
        # needs to change the quantity dimensions since we add a "tile" dimension,
        # unlike for tile scatter/gather which retains the same dimensions
        recv_quantity = Quantity(
            send_metadata.np.zeros(global_extent, dtype=send_metadata.dtype),
            dims=(constants.TILE_DIM,) + send_metadata.dims,
            units=send_metadata.units,
            origin=(0,) + tuple([0 for dim in send_metadata.dims]),
            extent=global_extent,
            backend=send_metadata.backend,
            allow_mismatch_float_precision=True,
        )
        return recv_quantity

    def _get_scatter_recv_quantity(
        self, shape: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving subtile data during scatter.

        Args:
            shape: ndarray shape, numpy-style
            send_metadata: metadata to the created Quantity
        """
        # needs to change the quantity dimensions since we remove a "tile" dimension,
        # unlike for tile scatter/gather which retains the same dimensions
        recv_quantity = Quantity(
            send_metadata.np.zeros(shape, dtype=send_metadata.dtype),
            dims=send_metadata.dims[1:],
            units=send_metadata.units,
            backend=send_metadata.backend,
            allow_mismatch_float_precision=True,
        )
        return recv_quantity


class NestTileCommunicator(TileCommunicator):
    """Communicate within one non-periodic nested grid region.

    Nested halo updates use the generic Communicator implementation rather
    than the specialized TileCommunicator path. NestedPartitioner supports
    non-periodic layouts that do not necessarily satisfy the layout
    assumptions enforced by TileCommunicator.
    """

    @classmethod
    def from_layout(
        cls,
        comm: CommABC,
        layout: tuple[int, int],
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> NestTileCommunicator:
        raise NotImplementedError(
            "NestTileCommunicator requires an existing NestedPartitioner"
        )

    @property
    def tile(self) -> NestTileCommunicator:
        return self

    def start_halo_update(
        self,
        quantity: Quantity | list[Quantity],
        n_points: int,
    ) -> HaloUpdater:
        return Communicator.start_halo_update(
            self,
            quantity,
            n_points,
        )

    def start_vector_halo_update(
        self,
        x_quantity: Quantity | list[Quantity],
        y_quantity: Quantity | list[Quantity],
        n_points: int,
    ) -> HaloUpdater:
        return Communicator.start_vector_halo_update(
            self,
            x_quantity,
            y_quantity,
            n_points,
        )


class NestedCommunicator:
    """Coordinate communication among a parent domain and nested domains.

    NestedCommunicator is an orchestration layer rather than the communicator
    for a single partitioned domain. It delegates same-resolution halo updates
    to the nested-domain communicators and coordinates parent-to-nested
    transport over the world communicator.

    The cross-domain orchestration in this class supports the current nested-grid
    prototype. Some of these responsibilities may ultimately belong to a
    higher-level component once ownership of nested-grid orchestration is
    established.

    Numerical coarse-to-fine interpolation is owned by the caller.
    """

    def __init__(
        self,
        comm: CommABC,
        parent_partitioner: Partitioner,
        nested_partitioners: Mapping[int, NestedPartitioner],
        nested_world_ranks: Mapping[int, Sequence[int]],
        parent_comm: Communicator | None = None,
        nested_comms: Mapping[int, NestTileCommunicator] | None = None,
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> None:
        self.comm = comm
        self.parent_comm = parent_comm
        self.nested_comms = dict(nested_comms) if nested_comms is not None else {}

        self.parent_partitioner = parent_partitioner
        self.nested_partitioners = dict(nested_partitioners)
        self.nested_world_ranks = {
            nest_id: tuple(world_ranks)
            for nest_id, world_ranks in nested_world_ranks.items()
        }

        self._force_cpu = force_cpu
        self.timer = timer if timer is not None else NullTimer()
        self._last_halo_tag = 0

        self.world_rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        self.parent_size = self.parent_partitioner.total_ranks

        self._validate_configuration()

    # -------------------------------------------------------------------------
    # Domain and rank bookkeeping
    #
    # These methods describe participation in the parent and nested domains and
    # translate between nested-domain ranks and world ranks.
    # -------------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        """Validate nested-domain rank mappings against configured partitioners."""
        if set(self.nested_partitioners) != set(self.nested_world_ranks):
            raise ValueError(
                "nested_partitioners and nested_world_ranks must "
                "contain the same nest ID's"
            )

        for nest_id, partitioner in self.nested_partitioners.items():
            world_ranks = self.nested_world_ranks[nest_id]

            if len(world_ranks) != partitioner.total_ranks:
                raise ValueError(
                    f"nested domain {nest_id} requires "
                    f"{partitioner.total_ranks} ranks, "
                    f"got {len(world_ranks)} world ranks"
                )

            for world_rank in world_ranks:
                if world_rank < 0 or world_rank >= self.world_size:
                    raise ValueError(
                        f"nested domain {nest_id} contains invalid world rank "
                        f"{world_rank} for world size {self.world_size}"
                    )

    @property
    def rank(self) -> int:
        return self.comm.Get_rank()

    @property
    def size(self) -> int:
        return self.comm.Get_size()

    @property
    def is_parent_rank(self) -> bool:
        return self.parent_comm is not None

    @property
    def is_nested_rank(self) -> bool:
        return bool(self.nested_comms)

    @property
    def parent_rank(self) -> int | None:
        if self.parent_comm is None:
            return None

        return self.parent_comm.rank

    def nested_rank(self, nest_id: int) -> int | None:
        nested_comm = self.nested_comms.get(nest_id)

        if nested_comm is None:
            return None

        return nested_comm.rank

    def nested_size(self, nest_id: int) -> int:
        return self.nested_partitioners[nest_id].total_ranks

    def _nested_partitioner(self, nest_id: int) -> NestedPartitioner:
        try:
            return self.nested_partitioners[nest_id]
        except KeyError as err:
            raise ValueError(f"unknown nested domain {nest_id}") from err

    def _nested_world_rank(self, nest_id: int, nested_rank: int) -> int:
        try:
            world_ranks = self.nested_world_ranks[nest_id]
        except KeyError as err:
            raise ValueError(f"unknown nested domain {nest_id}") from err

        if nested_rank < 0 or nested_rank >= len(world_ranks):
            raise ValueError(
                f"nested rank {nested_rank} is outside nested "
                f"domain {nest_id} of size {len(world_ranks)}"
            )

        return world_ranks[nested_rank]

    # -------------------------------------------------------------------------
    # Same-domain nested communication
    #
    # Fine-grid halo exchange remains ordinary communication within a single
    # nested domain and is delegated to that domain's NestTileCommunicator.
    # -------------------------------------------------------------------------

    def update_nested_halo(
        self,
        nest_id: int,
        fine_quantity: Quantity | None,
        n_points: int,
    ) -> None:
        """Perform same-resolution halo communication within one nest."""
        nested_comm = self.nested_comms.get(nest_id)

        if nested_comm is None:
            return

        if fine_quantity is None:
            raise ValueError(
                "fine_quantity is required on ranks participating "
                f"in nested domain {nest_id}"
            )

        nested_comm.halo_update(
            fine_quantity,
            n_points=n_points,
        )

    # -------------------------------------------------------------------------
    # Prototype cross-domain orchestration
    #
    # The methods below coordinate parent-to-nested transport. They establish
    # common quantity geometry, enforce current prototype restrictions, ask the
    # NestedPartitioner for cross-domain boundary geometry, and determine the
    # local side of each exchange.
    #
    # This orchestration may ultimately move to a higher-level component once
    # the ownership of nested-grid coupling is established.
    # -------------------------------------------------------------------------

    def _parent_quantity_geometry(
        self,
        parent_quantity: Quantity | None,
        anchor_parent_rank: int,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Broadcast parent quantity dimensions and parent-tile extent."""
        parent_info = None

        if self.world_rank == anchor_parent_rank:
            if self.parent_comm is None:
                raise RuntimeError(
                    "NestMapping anchor rank does not participate "
                    "in the parent communicator"
                )

            if parent_quantity is None:
                raise ValueError(
                    "parent_quantity must be supplied on the "
                    "NestMapping anchor parent rank"
                )

            parent_tile = self.parent_comm.tile
            parent_info = (
                tuple(parent_quantity.dims),
                tuple(parent_tile.partitioner.global_extent(parent_quantity.metadata)),
            )

        parent_info = self.comm.bcast(
            parent_info,
            root=anchor_parent_rank,
        )

        if parent_info is None:
            raise RuntimeError("Failed to broadcast parent quantity geometry")

        dims = tuple(parent_info[0])
        extent = tuple(int(value) for value in parent_info[1])

        return dims, extent

    def _nested_quantity_geometry(
        self,
        nest_id: int,
        nested_quantity: Quantity | None,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Broadcast nested quantity dimensions and nest-global extent."""
        nested_partitioner = self._nested_partitioner(nest_id)
        nested_anchor_world_rank = self._nested_world_rank(nest_id, 0)
        nested_info = None

        if self.world_rank == nested_anchor_world_rank:
            nested_comm = self.nested_comms.get(nest_id)

            if nested_comm is None:
                raise RuntimeError(
                    f"rank zero of nested domain {nest_id} "
                    "does not have its NestTileCommunicator"
                )

            if nested_quantity is None:
                raise ValueError(
                    "nested_quantity must be supplied on nested rank zero "
                    f"of nested domain {nest_id}"
                )

            nested_info = (
                tuple(nested_quantity.dims),
                tuple(nested_partitioner.global_extent(nested_quantity.metadata)),
            )

        nested_info = self.comm.bcast(
            nested_info,
            root=nested_anchor_world_rank,
        )

        if nested_info is None:
            raise RuntimeError(
                "Failed to broadcast nested quantity geometry "
                f"for nested domain {nest_id}"
            )

        dims = tuple(nested_info[0])
        extent = tuple(int(value) for value in nested_info[1])

        return dims, extent

    def _validate_parent_anchor(self, parent_rank: int) -> None:
        """Ensure a nest's parent anchor belongs to the parent communicator."""
        if parent_rank >= self.parent_size:
            raise ValueError(
                f"NestMapping parent_rank={parent_rank} is outside "
                f"parent communicator of size {self.parent_size}"
            )

    def _validate_nested_data_domain_within_parent_tile(
        self,
        nest_id: int,
        parent_tile_extent: tuple[int, ...],
        nested_global_extent: tuple[int, ...],
        dims: tuple[str, ...],
        coarse_n_points: int,
    ) -> None:
        """Ensure the requested nested coarse data domain stays on one parent tile.

        Crossing parent-rank boundaries is supported. Crossing a parent-tile
        boundary is not yet implemented.
        """
        nested_partitioner = self._nested_partitioner(nest_id)
        mapping = nested_partitioner.mapping

        try:
            i_index = next(
                index for index, dim in enumerate(dims) if dim in constants.I_DIMS
            )
            j_index = next(
                index for index, dim in enumerate(dims) if dim in constants.J_DIMS
            )
        except StopIteration as err:
            raise ValueError(
                f"Nested exchange requires horizontal i/j dimensions, got {dims}"
            ) from err

        parent_i0, parent_j0 = mapping.parent_start

        data_i0 = parent_i0 - coarse_n_points
        data_j0 = parent_j0 - coarse_n_points

        data_i1 = parent_i0 + nested_global_extent[i_index] + coarse_n_points
        data_j1 = parent_j0 + nested_global_extent[j_index] + coarse_n_points

        tile_i_extent = parent_tile_extent[i_index]
        tile_j_extent = parent_tile_extent[j_index]

        if (
            data_i0 < 0
            or data_j0 < 0
            or data_i1 > tile_i_extent
            or data_j1 > tile_j_extent
        ):
            raise NotImplementedError(
                "Nested data domains crossing parent tile boundaries "
                "are not yet implemented. "
                f"nest_id={nest_id}, "
                f"data_domain=(({data_i0}, {data_i1}), "
                f"({data_j0}, {data_j1})), "
                f"parent_tile_extent=({tile_i_extent}, {tile_j_extent})"
            )

    def _collect_parent_to_nested_boundaries(
        self,
        nest_id: int,
        parent_tile_extent: tuple[int, ...],
        nested_global_extent: tuple[int, ...],
        dims: tuple[str, ...],
        coarse_n_points: int,
    ) -> list[Boundary]:
        """Collect the parent-to-nested boundaries relevant to this world rank."""
        boundaries: list[Boundary] = []
        nested_partitioner = self._nested_partitioner(nest_id)

        if self.is_parent_rank:
            nested_ranks_to_process: Sequence[int] = range(
                nested_partitioner.total_ranks
            )
        else:
            nested_rank = self.nested_rank(nest_id)

            if nested_rank is None:
                return boundaries

            nested_ranks_to_process = (nested_rank,)

        for nested_rank in nested_ranks_to_process:
            for boundary_type in nested_partitioner.external_boundary_types(
                nested_rank
            ):
                exchanges = nested_partitioner.parent_to_nested_boundaries(
                    parent_partitioner=self.parent_partitioner,
                    parent_tile_extent=parent_tile_extent,
                    nested_global_extent=nested_global_extent,
                    dims=dims,
                    boundary_type=boundary_type,
                    nested_rank=nested_rank,
                    nested_world_rank=self._nested_world_rank(
                        nest_id,
                        nested_rank,
                    ),
                    coarse_n_points=coarse_n_points,
                )

                for parent_boundary, nested_boundary in exchanges:
                    if self.is_parent_rank:
                        if self.world_rank != parent_boundary.from_rank:
                            continue

                        boundaries.append(parent_boundary)
                    else:
                        boundaries.append(nested_boundary)

        return boundaries

    def _local_parent_to_nested_quantity(
        self,
        nest_id: int,
        parent_quantity: Quantity | None,
        nested_quantity: Quantity | None,
        dims: tuple[str, ...],
    ) -> Quantity:
        """Return the local quantity participating in an inter-domain exchange."""
        if self.is_parent_rank:
            if parent_quantity is None:
                raise ValueError(
                    "parent_quantity is required on a parent rank "
                    "participating in a nested exchange"
                )

            quantity = parent_quantity
        else:
            if nested_quantity is None:
                raise ValueError(
                    "nested_quantity is required on ranks "
                    f"participating in nested domain {nest_id}"
                )

            quantity = nested_quantity

        if tuple(quantity.dims) != dims:
            raise ValueError(
                "local Quantity dimensions do not match the planned exchange: "
                f"expected {dims}, got {quantity.dims}"
            )

        return quantity

    def _exchange_parent_to_nested(
        self,
        quantity: Quantity,
        boundaries: Sequence[Boundary],
        n_points: int,
        tag: int,
    ) -> None:
        """Execute a planned parent-to-nested exchange."""
        specification = self._quantity_halo_spec(
            quantity,
            n_points=n_points,
        )

        updater = HaloUpdater.from_scalar_specifications(
            comm=cast(Communicator[Any], self),
            numpy_like_module=self._maybe_force_cpu(quantity.np),
            specifications=[specification],
            boundaries=boundaries,
            tag=tag,
            optional_timer=self.timer,
        )

        updater.force_finalize_on_wait()
        updater.update([quantity])

    def parent_to_nested(
        self,
        nest_id: int,
        parent_quantity: Quantity | None,
        nested_quantity: Quantity | None,
        coarse_n_points: int,
    ) -> None:
        """Transport parent-resolution boundary data to a nested domain."""
        if coarse_n_points <= 0:
            raise ValueError("coarse_n_points must be positive")

        nested_partitioner = self._nested_partitioner(nest_id)
        anchor_parent_rank = nested_partitioner.mapping.parent_rank

        self._validate_parent_anchor(anchor_parent_rank)

        # Establish the geometry of the parent and nested transport quantities.
        parent_dims, parent_tile_extent = self._parent_quantity_geometry(
            parent_quantity,
            anchor_parent_rank,
        )
        nested_dims, nested_global_extent = self._nested_quantity_geometry(
            nest_id,
            nested_quantity,
        )

        if parent_dims != nested_dims:
            raise ValueError(
                "parent and nested quantities must have matching dimensions: "
                f"parent={parent_dims}, nested={nested_dims}"
            )

        dims = parent_dims

        # Validate the currently supported nesting geometry and determine which
        # parent/nested boundaries participate on this world rank.
        self._validate_nested_data_domain_within_parent_tile(
            nest_id=nest_id,
            parent_tile_extent=parent_tile_extent,
            nested_global_extent=nested_global_extent,
            dims=dims,
            coarse_n_points=coarse_n_points,
        )

        boundaries = self._collect_parent_to_nested_boundaries(
            nest_id=nest_id,
            parent_tile_extent=parent_tile_extent,
            nested_global_extent=nested_global_extent,
            dims=dims,
            coarse_n_points=coarse_n_points,
        )

        # Advance the tag on every rank involved in this orchestration so that
        # subsequent exchanges remain synchronized even on ranks with no local
        # parent-to-nested boundary.
        tag = self._get_halo_tag()

        if not boundaries:
            return

        quantity = self._local_parent_to_nested_quantity(
            nest_id=nest_id,
            parent_quantity=parent_quantity,
            nested_quantity=nested_quantity,
            dims=dims,
        )

        self._exchange_parent_to_nested(
            quantity=quantity,
            boundaries=boundaries,
            n_points=coarse_n_points,
            tag=tag,
        )

    def update_nested_boundaries(
        self,
        nest_id: int,
        parent_quantity: Quantity | None,
        nested_coarse_quantity: Quantity | None,
        fine_quantity: Quantity | None,
        fine_n_points: int,
        coarse_n_points: int,
    ) -> None:
        """Update internal nested halos and transport parent boundary data.

        Internal nested halos are exchanged at the fine-grid resolution.
        Parent boundary data is transported into a parent-resolution quantity
        on the nested ranks for subsequent coarse-to-fine interpolation.
        """
        self.update_nested_halo(
            nest_id=nest_id,
            fine_quantity=fine_quantity,
            n_points=fine_n_points,
        )

        self.parent_to_nested(
            nest_id=nest_id,
            parent_quantity=parent_quantity,
            nested_quantity=nested_coarse_quantity,
            coarse_n_points=coarse_n_points,
        )

    # -------------------------------------------------------------------------
    # HaloUpdater compatibility for cross-domain transport
    #
    # HaloUpdater currently expects a Communicator-shaped object. These methods
    # provide the minimal interface needed to reuse that machinery for
    # inter-domain exchanges. They are an implementation bridge for the current
    # prototype rather than part of the coordinator's conceptual API.
    # -------------------------------------------------------------------------

    def _get_halo_tag(self) -> int:
        self._last_halo_tag += 1
        return self._last_halo_tag

    def _device_synchronize(self) -> None:
        Communicator._device_synchronize()

    def _maybe_force_cpu(self, module: ModuleType) -> ModuleType:
        if self._force_cpu:
            return np

        return module

    def _quantity_halo_spec(
        self,
        quantity: Quantity,
        n_points: int,
    ) -> QuantityHaloSpec:
        data = quantity[:]

        return QuantityHaloSpec(
            n_points=n_points,
            shape=quantity.shape,
            strides=data.strides,
            itemsize=data.itemsize,
            origin=quantity.origin,
            extent=quantity.extent,
            dims=quantity.dims,
            numpy_module=self._maybe_force_cpu(quantity.np),
            dtype=quantity.metadata.dtype,
        )
