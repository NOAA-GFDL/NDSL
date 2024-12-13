import pytest

from tests.mpi.mpi_comm import MPI
# from ndsl.typing import Communicator

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    Quantity,
    TilePartitioner,
)

from ndsl.comm.partitioner import Partitioner

@pytest.fixture
def layout():
    if MPI is not None:
        size = MPI.COMM_WORLD.Get_size()
        ranks_per_tile = size // 6
        ranks_per_edge = int(ranks_per_tile ** 0.5)
        return (ranks_per_edge, ranks_per_edge)
    else:
        return (1, 1)

@pytest.fixture(params=[0.1, 1.0])
def edge_interior_ratio(request):
    return request.param

@pytest.fixture
def tile_partitioner(layout, edge_interior_ratio: float):
    return TilePartitioner(layout, edge_interior_ratio=edge_interior_ratio)

@pytest.fixture
def cube_partitioner(tile_partitioner):
    return CubedSpherePartitioner(tile_partitioner)

@pytest.fixture()
def communicator(cube_partitioner):
    return CubedSphereCommunicator(
        comm=MPI.COMM_WORLD,
        partitioner=cube_partitioner,
    )

def test_all_reduce_sum(
    communicator,      
):
    print("Communicator rank = ", communicator.rank)
    print("Communicator size = ", communicator.size)
    assert True