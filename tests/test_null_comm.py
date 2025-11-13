import pytest

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    NullComm,
    TilePartitioner,
)


def test_can_create_cube_communicator():
    with pytest.deprecated_call(match="NullComm is deprecated"):
        mpi_comm = NullComm(rank=2, total_ranks=24)
        partitioner = CubedSpherePartitioner(TilePartitioner(layout=(2, 2)))
        communicator = CubedSphereCommunicator(mpi_comm, partitioner)
        assert communicator.tile.partitioner
