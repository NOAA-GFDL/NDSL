import numpy as np
from cartopy import crs as ccrs
from matplotlib import pyplot as plt

from ndsl import Quantity, ndsl_log
from ndsl.comm.communicator import Communicator
from ndsl.grid import GridData
from ndsl.viz.fv3 import pcolormesh_cube


def plot_cube_sphere(
    quantity: Quantity,
    k_level: int,
    comm: Communicator,
    grid_data: GridData,
    save_to_path: str,
):
    if len(quantity.shape) < 2 or len(quantity.shape) > 3:
        ndsl_log.error(
            f"[Plot Cube] Can't plot quantity with shape == {quantity.shape}"
        )
        return

    data = comm.gather(quantity)
    lat = comm.gather(grid_data.lat)
    lon = comm.gather(grid_data.lon)

    if comm.rank == 0:
        # We are on the root rank so comm.gather() did gather. This is just to make mypy happy.
        assert data is not None and lat is not None and lon is not None

        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.Robinson()})
        pcolormesh_cube(
            lat.view[:] * 180.0 / np.pi,
            lon.view[:] * 180.0 / np.pi,
            data.view[:] if len(data.shape) == 3 else data.view[:, :, :, k_level],
            ax=ax,
        )
        fig.savefig(save_to_path)
