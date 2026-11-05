"""Eight-rank parent/nest scalar communication, without physical grid metrics.

Run with ``mpiexec -np 8 python -m mpi4py -m pytest -x -m parallel
 tests/mpi/test_nested_grid.py``. Layout (2, 1) means two ranks along j.
"""

import numpy as np
import pytest

from ndsl import (
    Backend,
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    MPIComm,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from ndsl.comm.communicator import NestedCommunicator, NestTileCommunicator
from ndsl.comm.partitioner import NestedPartitioner, NestMapping
from ndsl.constants import (
    BOUNDARY_TYPES,
    I_DIM,
    I_INTERFACE_DIM,
    J_DIM,
    J_INTERFACE_DIM,
    NORTH,
    SOUTH,
)

pytestmark = pytest.mark.parallel

# The parent patch [2:4, 2:6] and one coarse halo fit inside an 8x8 face.
PARENT_CELLS = (8, 8)
COARSE_CELLS = (2, 4)
FINE_CELLS = (4, 8)
STORAGE_HALO = 2
EXCHANGE_HALO = 1
SENTINEL = -999999.0


def _quantity(comm, cells, dims, halo=STORAGE_HALO):
    backend = Backend("st:numpy:cpu:IJK")
    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=cells[0],
        ny_tile=cells[1],
        nz=1,
        n_halo=halo,
        layout=comm.partitioner.layout,
        tile_partitioner=comm.partitioner,
        tile_rank=comm.rank,
        backend=backend,
    )
    quantity = QuantityFactory(sizer, backend=backend).zeros(dims, units="1")
    quantity[:] = SENTINEL
    return quantity


def _coordinates(quantity, j_start=0):
    """Storage coordinates, derived from the fixed test decomposition."""
    i = np.arange(quantity.shape[0])[:, None] - quantity.origin[0]
    j = np.arange(quantity.shape[1])[None, :] - quantity.origin[1] + j_start
    return i, j


def _parent_values(i, j, epoch, face=0):
    return 1000.0 + 10000 * epoch + 1000000 * face + 100 * i + j


def _fine_values(i, j, epoch):
    return 20000.0 + 1000 * epoch + 100 * i + j


def _check_all(world, checks):
    """Finish exchanges before reporting numerical failures on every rank."""
    failures = []
    for label, actual, expected in checks:
        try:
            np.testing.assert_array_equal(actual, expected)
        except AssertionError as error:
            failures.append(f"rank {world.Get_rank()}, {label}: {error}")
    failures_by_rank = world.allgather(failures)
    message = "\n".join(
        failure for rank_failures in failures_by_rank for failure in rank_failures
    )
    assert not message, message


@pytest.fixture
def domains():
    world = MPIComm()
    if world.Get_size() != 8:
        pytest.skip("requires six parent and two nested MPI ranks")
    parent = world.Get_rank() < 6
    raw_role = world.Split(color=0 if parent else 1, key=world.Get_rank())
    tile = None
    try:
        role = MPIComm()
        role._comm = raw_role  # MPIComm.Split returns a raw mpi4py communicator.
        parent_partitioner = CubedSpherePartitioner(TilePartitioner((1, 1)))
        nested_partitioner = NestedPartitioner(
            (2, 1), NestMapping(0, (2, 2), COARSE_CELLS, 2, parent_region=0)
        )
        parent_comm = (
            CubedSphereCommunicator(role, parent_partitioner) if parent else None
        )
        nested_comm = None if parent else NestTileCommunicator(role, nested_partitioner)
        # CubedSphereCommunicator.tile lazily splits the parent communicator;
        # materialize it on ALL parent ranks before world-level transport starts.
        tile = parent_comm.tile if parent else None
        coupling = NestedCommunicator(
            world,
            parent_partitioner,
            {0: nested_partitioner},
            {0: (6, 7)},
            parent_comm=parent_comm,
            nested_comms={} if parent else {0: nested_comm},
        )
        actual = world.allgather(
            (
                coupling.is_parent_rank,
                coupling.is_nested_rank,
                coupling.parent_rank,
                coupling.nested_rank(0),
                role.Get_rank(),
                role.Get_size(),
                coupling.parent_size,
                coupling.nested_size(0),
                None if parent else nested_partitioner.subtile_index(nested_comm.rank),
                (
                    None
                    if parent
                    else tuple(
                        (
                            None
                            if nested_partitioner.boundary(b, nested_comm.rank) is None
                            else nested_partitioner.boundary(
                                b, nested_comm.rank
                            ).to_rank
                        )
                        for b in BOUNDARY_TYPES
                    )
                ),
                (
                    (
                        tile.rank,
                        tile.size,
                        parent_partitioner.tile_index(parent_comm.rank),
                    )
                    if parent
                    else None
                ),
            )
        )
        expected = []
        for rank in range(8):
            is_parent = rank < 6
            neighbor = NORTH if rank == 6 else SOUTH
            expected.append(
                (
                    is_parent,
                    not is_parent,
                    rank if is_parent else None,
                    None if is_parent else rank - 6,
                    rank if is_parent else rank - 6,
                    6 if is_parent else 2,
                    6,
                    2,
                    None if is_parent else (rank - 6, 0),
                    (
                        None
                        if is_parent
                        else tuple(
                            7 - rank if boundary == neighbor else None
                            for boundary in BOUNDARY_TYPES
                        )
                    ),
                    (0, 1, rank) if is_parent else None,
                )
            )
        assert actual == expected
        yield world, coupling, tile if parent else nested_comm
    finally:
        try:
            if tile is not None:
                tile.comm.Free()
        finally:
            raw_role.Free()


@pytest.mark.parametrize(
    "dims",
    [(I_DIM, J_DIM), (I_INTERFACE_DIM, J_DIM), (I_DIM, J_INTERFACE_DIM)],
    ids=["center", "i_interface", "j_interface"],
)
def test_nested_grid(domains, dims):
    world, coupling, comm = domains
    parent = coupling.is_parent_rank
    quantity = _quantity(comm, PARENT_CELLS if parent else FINE_CELLS, dims)
    coarse = None if parent else _quantity(comm, COARSE_CELLS, dims, halo=EXCHANGE_HALO)
    for epoch in range(2):
        quantity[:] = SENTINEL
        i, j = _coordinates(quantity, 0 if parent else comm.rank * 4)
        values = (
            _parent_values(i, j, epoch, world.Get_rank())
            if parent
            else _fine_values(i, j, epoch)
        )
        origin, extent = quantity.origin, quantity.extent
        compute = tuple(slice(o, o + e) for o, e in zip(origin, extent))
        quantity.view[:] = values[compute]
        expected_fine = np.array(quantity[:], copy=True)
        if not parent:
            coarse[:] = SENTINEL
            # Only one internal boundary exists: north for rank 0, south for 1.
            receive_j = origin[1] + extent[1] if comm.rank == 0 else origin[1] - 1
            expected_fine[origin[0] : origin[0] + extent[0], receive_j] = _fine_values(
                i, j, epoch
            )[origin[0] : origin[0] + extent[0], receive_j]
        coupling.update_nested_boundaries(
            0,
            quantity if parent else None,
            coarse,
            None if parent else quantity,
            EXCHANGE_HALO,
            EXCHANGE_HALO,
        )
        checks = [
            (
                "parent source" if parent else "fine internal halo",
                quantity[:],
                expected_fine,
            )
        ]
        if not parent:
            # Independent rectangles for this fixed two-rank j decomposition.
            # Interface quantities include physical exterior endpoints.
            ci, cj = _coordinates(coarse)
            ni, nj = coarse.extent
            logical = (ci >= -1) & (ci < ni + 1) & (cj >= -1) & (cj < nj + 1)
            external = (ci < int(dims[0] == I_INTERFACE_DIM)) | (ci >= 2)
            external = external | (
                (cj < int(dims[1] == J_INTERFACE_DIM)) if comm.rank == 0 else (cj >= 2)
            )
            requested = logical & external
            expected_coarse = np.full(coarse.shape, SENTINEL)
            donors = _parent_values(2 + ci, 2 + comm.rank * 2 + cj, epoch)
            expected_coarse[requested] = donors[requested]
            checks.append(
                ("parent-to-nest coarse boundary", coarse[:], expected_coarse)
            )
        _check_all(world, checks)
