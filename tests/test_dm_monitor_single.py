"""
    Test for DiagManagerMonitor using a single tile partitioner and communicator.
    This is a simple test that registers some axes and fields, sends data to the monitor, and then checks the output file.
"""


import logging
from datetime import timedelta, datetime
from typing import List

import cftime
import numpy as np
import pytest
import xarray as xr
import pdb

from ndsl import (
    LocalComm,
    MPIComm,
    TileCommunicator,
    TilePartitioner,
    DiagManagerMonitor,
)

from ndsl.initialization import SubtileGridSizer
from ndsl import QuantityFactory
from pyfms import mpp_domains, fms, diag_manager

from pathlib import Path

import yaml

import pdb

logger = logging.getLogger(__name__)


# mpi info
npes = MPIComm()._comm.Get_size()
pe = MPIComm()._comm.Get_rank()
# tile parameters for quantities/domains 
nx = 8
ny = 8
nz = 2 
nhalo = 0
backend = "debug"
ntimesteps = 3
layout_fms = [1, npes]
io_layout = [1, 1]
layout_ndsl = (npes, 1) # flipped to match fms domain decomposition
global_indices = [0, nx-1, 0, ny-1]

def _create_input(reduction: str = "none"):
    diag_config = {
        "title": "ndsl_diag_manager_test",
        "base_date": "2 1 1 1 1 1",
        "diag_files": [
            {
                "file_name": "ndsl_diag_test",
                "freq": "1 hours",
                "time_units": "hours",
                "unlimdim": "time",
                "varlist": [
                    {
                        "module": "atm_mod",
                        "var_name": "var_2d",
                        "long_name": "variable_too_dee",
                        "reduction": "none",
                        "kind": "r8",
                        "output_name": "var_2avg"
                    },
                    {
                        "module": "atm_mod",
                        "var_name": "var_3d",
                        "long_name": "variable_three_dee",
                        "reduction": "none",
                        "kind": "r8",
                        "output_name" : "var3"
                    }
                ]
            }
        ]
    }
    with open("diag_table.yaml", "w") as f:
        yaml.dump(diag_config, f, default_flow_style=False, sort_keys=False)
    text_content = "&diag_manager_nml\nuse_modern_diag=.true.\n/"
    with open("input.nml", "w", encoding="utf-8") as f:
        f.write(text_content)


# Simple test, uses single tile partitioner and communicator and then stores
# a faux state dict via DiagManagerMonitor 
def test_dm_monitor_single_tile():

    _create_input()

    fms.init(localcomm=MPIComm()._comm.py2f(), calendar_type=fms.NOLEAP)

    domain = mpp_domains.define_domains(
        global_indices=global_indices,
        layout=layout_fms,
    )
    mpp_domains.set_current_domain(domain_id=domain.domain_id)
    domain_id = domain.domain_id
    mpp_domains.define_io_domain(
        domain_id=domain_id,
        io_layout=io_layout,
    )

    # handles mpi/serial set up steps
    if npes > 1: 
        rank = MPIComm()._comm.Get_rank()
        print(f"intializing partitioner/communicator rank {rank} of {npes}")
        # this only works with 1 pe
        partitioner = TilePartitioner(layout=layout_ndsl)
        communicator = TileCommunicator(MPIComm(), partitioner)
        communicator.tile
    else:
        buffer = {}
        partitioner = TilePartitioner((1,1)) 
        communicator = TileCommunicator(
                comm = LocalComm(rank=0, total_ranks=npes, buffer_dict=buffer),
                partitioner=partitioner,
        )
        communicator.tile

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=nz,
        n_halo=nhalo,
        layout=layout_ndsl,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    quantity_factory = QuantityFactory(sizer, backend=backend)

    # pace will set up model start/end times and register axis info
    monitor = DiagManagerMonitor()
    start = datetime(2, 1, 1, 1, 1, 1)
    step = timedelta(seconds=3600)
    end = start + ntimesteps * step

    monitor.set_timestep(step)
    monitor.set_end_time(end)

    monitor.register_axis(
        name="x",
        axis_data=np.arange(nx, dtype=np.float64),
        units="point_E",
        cart_name='x',
        domain_id=domain_id,
        long_name="point_E",
        set_name="atm",
        not_xy=False,
    )
    monitor.register_axis(
        name="y",
        axis_data=np.arange(ny, dtype=np.float64),
        units="point_N",
        cart_name='y',
        domain_id=domain_id,
        long_name="point_N",
        set_name="atm",
        not_xy=False,
    )
    monitor.register_axis(
        name="z",
        axis_data=np.arange(nz, dtype=np.float64),
        units="point_Z",
        cart_name='z',
        long_name="point_Z",        set_name="atm",
        not_xy=True,
    )

    # fields will be registered in the component they are defined in (either pyFV3 or pySHiELD)
    monitor.register_field(
        module_name="atm_mod",
        field_name="var_2d",
        dims=["x","y"],
        units="muntin",
        init_time=start,
        dtype="float64",
        missing_value=-99.99,
    )
    monitor.register_field(
        module_name="atm_mod",
        field_name="var_3d",
        dims=["x","y","z"],
        units="muntin",
        init_time=start,
        dtype="float64",
        missing_value=-99.99,
    )
    assert "x" in monitor.axes
    assert "y" in monitor.axes
    assert "z" in monitor.axes
    assert "var_2d" in monitor.fields
    assert "var_3d" in monitor.fields
    
    # set up data to send for diagnostics
    var2_global = np.empty(shape=(nx, ny), dtype=np.float64)
    var3_global = np.empty(shape=(nx, ny, nz), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            var2_global[i][j] = i * 10.0 + j
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                var3_global[i][j][k] = i * 100 + j * 10 + k
    var2 = var2_global[domain.isc: domain.iec + 1, domain.jsc: domain.jec + 1]
    var3 = var3_global[domain.isc: domain.iec + 1, domain.jsc: domain.jec + 1, :]

    # pad arrays for quantity factory
    var2 = np.pad(var2, (0,1))
    var3 = np.pad(var3, (0,1))
    field_q1 = quantity_factory.from_array(var2, dims=("x", "y"), units="m")
    field_q2 = quantity_factory.from_array(var3, dims=("x","y","z"), units="m")

    MPIComm()._comm.Barrier()

    # pace driver will call store for each timestep to send the data
    current_time = start
    for t in range(ntimesteps):
        current_time = current_time + step
        state = {
            "time": current_time,
            "var_2d": field_q1,
            "var_3d": field_q2,
        }
        monitor.store(state)

    #if pe == 0:
    MPIComm()._comm.Barrier()
    # cleanup writes and closes the file
    monitor.cleanup()
    fms.end()
    return
    ## check output!
    assert Path("ndsl_diag_test.nc").exists()
    ds = xr.open_mfdataset("ndsl_diag_test.nc", decode_times=True)
    assert "var1" in ds
    np.testing.assert_array_equal(
        ds["var1"].shape, (ntimesteps, nx, ny)
    )
    assert ds["var1"].dims == ("time", "y", "x")
    assert ds["var1"].attrs["units"] == "m"
    assert ds["var2"].dims == ("time", "z", "y", "x")
    assert ds["var2"].attrs["units"] == "m"
    assert ds["time"].shape == (ntimesteps,)
    assert ds["time"].dims == ("time",)
    assert ds["time"].values[0] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, second=15)
    assert ds["time"].values[1] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, second=30) 
    assert ds["time"].values[2] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, second=45) 
    np.testing.assert_array_equal(ds["var1"].values[0,:,:], var2_global.transpose())
    np.testing.assert_array_equal(ds["var1"].values[1,:,:], var2_global.transpose())
    np.testing.assert_array_equal(ds["var1"].values[2,:,:], var2_global.transpose())
    # data is transposed when passed into fortran
    np.testing.assert_array_equal(ds["var2"].values[0,:,:,:], var3_global.transpose())
    np.testing.assert_array_equal(ds["var2"].values[1,:,:,:], var3_global.transpose())
    np.testing.assert_array_equal(ds["var2"].values[2,:,:,:], var3_global.transpose())

    fms.end()

if __name__ == "__main__":
    test_dm_monitor_single_tile()