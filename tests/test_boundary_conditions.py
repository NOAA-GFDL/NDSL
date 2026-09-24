from dataclasses import dataclass
from pathlib import Path

import pytest

from ndsl import (
    Backend,
    Quantity,
    QuantityFactory,
    SubtileGridSizer,
    TileCommunicator,
    TilePartitioner,
)
from ndsl.boundary_condition import BoundaryCondition
from ndsl.comm.mpi import MPIComm
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.grid import MetricTerms


@pytest.mark.boundary_condition
def test_boundary_condition():

    backend = Backend.python()

    mpi_comm = MPIComm()
    size = mpi_comm.Get_size()
    npes_per_tile = size
    npes_per_edge = int(npes_per_tile**0.5)
    layout = (npes_per_edge, npes_per_edge)
    pe = mpi_comm.Get_rank()

    ny = 24
    nx = ny
    nz = 5
    nhalo = 3

    partitioner = TilePartitioner(layout)

    communicator = TileCommunicator(mpi_comm, partitioner)

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=nz,
        n_halo=nhalo,
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
        backend=backend,
    )

    quantity_factory = QuantityFactory(sizer=sizer, backend=backend)

    metric_terms = MetricTerms(
        quantity_factory=quantity_factory, communicator=communicator
    )

    var2D = quantity_factory.zeros(dims=(I_DIM, J_DIM), units="m/s", dtype="float")
    var3D = quantity_factory.zeros(
        dims=(I_DIM, J_DIM, K_DIM), units="m/s", dtype="float"
    )

    var2D[:-1, :-1] = 200 + pe
    var2D[3:-4, 3:-4] = pe

    var3D[:-1, :-1, :-1] = 300 + pe
    var3D[3:-4, 3:-4, :] = pe

    @dataclass
    class Tester:
        var2D: Quantity
        var3D: Quantity

    tester_dict = {"var2D": var2D, "var3D": var3D}
    tester_instance = Tester(**tester_dict)

    reg_bc = BoundaryCondition(comm=communicator, layout=layout)
    reg_bc.write_out_bcs(state=tester_instance, bc_file_name="test_bc.nc")

    var2D[:-1, :-1] = 0
    var2D[3:-4, 3:-4] = pe

    reg_bc2 = BoundaryCondition(comm=communicator, layout=layout, file="test_bc.nc")
    reg_bc2.scatter_bcs(state=tester_instance, timestep=0)
    top_list = []
    bottom_list = []
    right_list = []
    left_list = []

    for n in range(layout[1]):
        top_list.append(layout[1] * (layout[1] - 1) + n)
        bottom_list.append(n)
        right_list.append(layout[0] * (n + 1) - 1)
        left_list.append(layout[0] * n)

    if pe in top_list:
        assert (var2D[:3, :-1] == 200 + pe).all()
    if pe in bottom_list:
        assert (var2D[-4:-1, :-1] == 200 + pe).all()
    if pe in right_list:
        assert (var2D[:-1, -4:-1] == 200 + pe).all()
    if pe in left_list:
        assert (var2D[:-1, :3] == 200 + pe).all()

    file_path = Path("test_bc.nc")
    file_path.unlink(missing_ok=True)
