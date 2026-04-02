from datetime import timedelta

import cftime
import numpy as np
import zarr
from mpi4py import MPI

from ndsl import (
    CubedSpherePartitioner,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
    ZarrMonitor,
)
from ndsl.config import backend_python
from ndsl.constants import I_DIM, J_DIM, K_DIM


OUTPUT_PATH = "output/zarr_monitor.zarr"


def get_example_state(time):
    sizer = SubtileGridSizer(
        nx=48, ny=48, nz=70, n_halo=3, data_dimensions={}, backend=backend_python
    )
    allocator = QuantityFactory(sizer, backend=backend_python)
    air_temperature = allocator.zeros([I_DIM, J_DIM, K_DIM], units="degK")
    air_temperature.view[:] = np.random.randn(*air_temperature.extent)
    return {"time": time, "air_temperature": air_temperature}


if __name__ == "__main__":
    size = MPI.COMM_WORLD.Get_size()
    # assume square tile faces
    ranks_per_edge = int((size // 6) ** 0.5)
    layout = (ranks_per_edge, ranks_per_edge)

    store = zarr.storage.DirectoryStore(OUTPUT_PATH)
    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    monitor = ZarrMonitor(store, partitioner, mpi_comm=MPI.COMM_WORLD)

    time = cftime.DatetimeJulian(2020, 1, 1)
    timestep = timedelta(hours=1)

    for _i in range(10):
        state = get_example_state(time)
        monitor.store(state)
        time += timestep
