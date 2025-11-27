import numpy as np
import pytest

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    Quantity,
    TilePartitioner,
)
from ndsl.comm.comm_abc import ReductionOperator
from ndsl.comm.mpi import MPIComm
from ndsl.dsl.typing import Float
from tests.mpi import MPI


@pytest.fixture
def layout():
    if MPI is None:
        return (1, 1)

    size = MPI.COMM_WORLD.Get_size()
    ranks_per_tile = size // 6
    ranks_per_edge = int(ranks_per_tile**0.5)
    return (ranks_per_edge, ranks_per_edge)


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
        comm=MPIComm(),
        partitioner=cube_partitioner,
    )


@pytest.mark.skipif(MPI is None, reason="pytest is not run in parallel")
def test_all_reduce(communicator):
    backends = ["dace:cpu", "gt:cpu_kfirst", "numpy"]

    for backend in backends:
        base_array = np.array([i for i in range(5)], dtype=Float)

        testQuantity_1D = Quantity(
            data=base_array,
            dims=["K"],
            units="Some 1D unit",
            backend=backend,
        )

        base_array = np.array([i for i in range(5 * 5)], dtype=Float)
        base_array = base_array.reshape(5, 5)

        testQuantity_2D = Quantity(
            data=base_array,
            dims=["I", "J"],
            units="Some 2D unit",
            backend=backend,
        )

        base_array = np.array([i for i in range(5 * 5 * 5)], dtype=Float)
        base_array = base_array.reshape(5, 5, 5)

        testQuantity_3D = Quantity(
            data=base_array,
            dims=["I", "J", "K"],
            units="Some 3D unit",
            backend=backend,
        )

        global_sum_q = communicator.all_reduce(testQuantity_1D, ReductionOperator.SUM)
        assert global_sum_q.metadata == testQuantity_1D.metadata
        assert (global_sum_q.data == (testQuantity_1D.data * communicator.size)).all()

        global_sum_q = communicator.all_reduce(testQuantity_2D, ReductionOperator.SUM)
        assert global_sum_q.metadata == testQuantity_2D.metadata
        assert (global_sum_q.data == (testQuantity_2D.data * communicator.size)).all()

        global_sum_q = communicator.all_reduce(testQuantity_3D, ReductionOperator.SUM)
        assert global_sum_q.metadata == testQuantity_3D.metadata
        assert (global_sum_q.data == (testQuantity_3D.data * communicator.size)).all()

        base_array = np.array([i for i in range(5)], dtype=Float)
        testQuantity_1D_out = Quantity(
            data=base_array,
            dims=["K"],
            units="New 1D unit",
            backend=backend,
            origin=(8,),
            extent=(7,),
        )

        base_array = np.array([i for i in range(5 * 5)], dtype=Float)
        base_array = base_array.reshape(5, 5)

        testQuantity_2D_out = Quantity(
            data=base_array,
            dims=["I", "J"],
            units="Some 2D unit",
            backend=backend,
        )

        base_array = np.array([i for i in range(5 * 5 * 5)], dtype=Float)
        base_array = base_array.reshape(5, 5, 5)

        testQuantity_3D_out = Quantity(
            data=base_array,
            dims=["I", "J", "K"],
            units="Some 3D unit",
            backend=backend,
        )
        communicator.all_reduce(
            testQuantity_1D, ReductionOperator.SUM, testQuantity_1D_out
        )
        assert testQuantity_1D_out.metadata == testQuantity_1D.metadata
        assert (
            testQuantity_1D_out.data == (testQuantity_1D.data * communicator.size)
        ).all()

        communicator.all_reduce(
            testQuantity_2D, ReductionOperator.SUM, testQuantity_2D_out
        )
        assert testQuantity_2D_out.metadata == testQuantity_2D.metadata
        assert (
            testQuantity_2D_out.data == (testQuantity_2D.data * communicator.size)
        ).all()

        communicator.all_reduce(
            testQuantity_3D, ReductionOperator.SUM, testQuantity_3D_out
        )
        assert testQuantity_3D_out.metadata == testQuantity_3D.metadata
        assert (
            testQuantity_3D_out.data == (testQuantity_3D.data * communicator.size)
        ).all()
