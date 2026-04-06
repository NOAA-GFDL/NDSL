"""Tests the diag_manager_monitor class can output ndsl quantity data.
This test case uses a cubic (6 tile) mosaic, and outputs a file for each tile.
"""

from datetime import datetime, timedelta
from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr
import yaml

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    DiagManagerMonitor,
    MPIComm,
    QuantityFactory,
    TilePartitioner,
)
from ndsl.config import Backend
from ndsl.initialization import SubtileGridSizer


pyfms = pytest.importorskip("pyfms")


# init fms mpi and set up a simple domain
def fms_mpp_init():
    pyfms.fms.init(localcomm=MPIComm()._comm.py2f(), calendar_type=pyfms.fms.NOLEAP)
    x = 8
    y = 8
    layout = [1, 1]
    io_layout = [1, 1]
    halo = 1
    tiles = 6
    domain_id = pyfms.mpp_domains.define_cubic_mosaic(
        ni=[x for i in range(6)],
        nj=[y for i in range(6)],
        global_indices=[0, x - 1, 0, y - 1],
        layout=layout,
        ntiles=tiles,
        halo=halo,
        use_memsize=False,
    )
    pyfms.mpp_domains.define_io_domain(
        domain_id=domain_id,
        io_layout=io_layout,
    )
    pyfms.mpp_domains.set_current_domain(domain_id)
    return domain_id


def _create_input(reduction: str = "none"):
    diag_config = {
        "title": "ndsl_diag_manager_test",
        "base_date": "1 1 1 0 0 0",
        "diag_files": [
            {
                "file_name": "diag_manager_cubed_sphere",
                "freq": "15 seconds",
                "time_units": "seconds",
                "unlimdim": "time",
                "varlist": [
                    {
                        "module": "atm_mod",
                        "var_name": "var1",
                        "long_name": "variable_number_one",
                        "reduction": reduction,
                        "kind": "r8",
                    },
                    {
                        "module": "atm_mod",
                        "var_name": "var2",
                        "long_name": "variable_number_one",
                        "reduction": reduction,
                        "kind": "r8",
                    },
                ],
            }
        ],
    }
    with open("diag_table.yaml", "w") as f:
        yaml.dump(diag_config, f, default_flow_style=False, sort_keys=False)
    text_content = "&diag_manager_nml\nuse_modern_diag=.true.\n/"
    with open("input.nml", "w", encoding="utf-8") as f:
        f.write(text_content)


# Simple test, uses a lat/lon grid and (1, npes) layout
def test_dm_monitor():

    npes = MPIComm()._comm.Get_size()
    if npes % 6 != 0:
        raise RuntimeError("this test requires npes to be a multiple of 6 to run")

    _create_input()

    nx = 8
    ny = 8
    nz = 2
    nhalo = 0
    layout = (1, 1)  # 1 pe per tile
    backend = Backend.python()
    ntimesteps = 3

    domain_id = fms_mpp_init()
    partitioner = CubedSpherePartitioner(TilePartitioner((1, 1)))
    communicator = CubedSphereCommunicator(MPIComm(), partitioner)
    communicator.tile

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
    quantity_factory = QuantityFactory(sizer, backend=backend)

    # pace will set up model start/end times and register axis info
    monitor = DiagManagerMonitor(
        domain_id=domain_id,
    )
    start = datetime(1, 1, 1, 0, 0, second=0)
    end = datetime(1, 1, 1, 0, 0, second=45)
    step = timedelta(seconds=15)
    monitor.set_timestep(step)
    monitor.set_end_time(end)

    monitor.register_axis(
        name="x",
        axis_data=np.arange(nx, dtype=np.float64),
        cart_name="x",
        long_name="x coordinate",
        units="m",
        not_xy=False,
        domain_id=domain_id,
    )
    monitor.register_axis(
        name="y",
        axis_data=np.arange(ny, dtype=np.float64),
        cart_name="y",
        long_name="y coordinate",
        units="m",
        not_xy=False,
        domain_id=domain_id,
    )
    monitor.register_axis(
        name="z",
        axis_data=np.arange(nz, dtype=np.float64),
        cart_name="z",
        long_name="z coordinate",
        units="m",
        not_xy=True,
    )

    # fields will be registered in the component they are defined in (either pyFV3 or pySHiELD)
    monitor.register_field(
        module_name="atm_mod",
        field_name="var1",
        dims=["x", "y"],
        units="m",
        long_name="variable one",
        init_time=start,
        missing_value=-999.0,
        dtype="float64",
    )
    monitor.register_field(
        module_name="atm_mod",
        field_name="var2",
        dims=["x", "y", "z"],
        units="m",
        long_name="variable two",
        init_time=start,
        missing_value=-999.0,
        dtype="float64",
    )
    assert "x" in monitor.axes
    assert "y" in monitor.axes
    assert "var1" in monitor.fields

    # pace driver will call store for each timestep to send the data
    for t in range(1, ntimesteps + 1):
        current_time = start + t * step
        field_q1 = quantity_factory.full(
            dims=("i", "j"), units="m", value=t, dtype=np.float64
        )
        field_q2 = quantity_factory.full(
            dims=("i", "j", "k"), units="m", value=t * 2, dtype=np.float64
        )
        state = {"time": current_time, "var1": field_q1, "var2": field_q2}
        monitor.store(state)

    # cleanup writes and closes the file
    monitor.cleanup()

    pe = MPIComm()._comm.Get_rank() + 1
    filename = "diag_manager_cubed_sphere.tile" + str(pe) + ".nc"
    assert Path(filename).exists()
    ds = xr.open_mfdataset(filename, decode_times=True)
    assert "var1" in ds
    np.testing.assert_array_equal(ds["var1"].shape, (ntimesteps, ny, nx))
    assert "var2" in ds
    np.testing.assert_array_equal(ds["var2"].shape, (ntimesteps, nz, ny, nx))
    assert ds["var1"].dims == ("time", "y", "x")
    assert ds["var2"].dims == ("time", "z", "y", "x")
    assert ds["time"].shape == (ntimesteps,)
    assert ds["time"].dims == ("time",)
    assert ds["time"].values[0] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 15)
    assert ds["time"].values[1] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 30)
    assert ds["time"].values[2] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, 45)
    # data is just the timestep number
    np.testing.assert_array_equal(ds["var1"].values[0, :, :], 1)
    np.testing.assert_array_equal(ds["var1"].values[1, :, :], 2)
    np.testing.assert_array_equal(ds["var1"].values[2, :, :], 3)
    np.testing.assert_array_equal(ds["var2"].values[0, :, :, :], 2)
    np.testing.assert_array_equal(ds["var2"].values[1, :, :, :], 4)
    np.testing.assert_array_equal(ds["var2"].values[2, :, :, :], 6)

    pyfms.fms.end()
