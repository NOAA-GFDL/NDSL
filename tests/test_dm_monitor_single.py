""" Tests the diag_manager_monitor class can output ndsl quantity data.
    This test case uses a single tile domain decomposition, and outputs a file from the root pe
    with data gathered from any other processors.
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
                "file_name": "diag_manager_single_tile",
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
                    },
                    {
                        "module": "atm_mod",
                        "var_name": "var_3d",
                        "long_name": "variable_three_dee",
                        "reduction": "none",
                        "kind": "r8",
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

    if npes > 1: 
        rank = MPIComm()._comm.Get_rank()
        print(f"intializing partitioner/communicator rank {rank} of {npes}")
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

    # set up for diag manager for before the main loop, need to set timestep + end_time and register all axes and fields
    monitor = DiagManagerMonitor(domain_id=domain_id)
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
        long_name="point_Z",
        set_name="atm",
        not_xy=True,
    )

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

    current_time = start
    for t in range(ntimesteps):
        current_time = current_time + step
        state = {
            "time": current_time,
            "var_2d": field_q1,
            "var_3d": field_q2,
        }
        monitor.store(state)

    # cleanup writes and closes the file
    monitor.cleanup()

    ## check output!
    assert Path("diag_manager_single_tile.nc").exists()
    ds = xr.open_mfdataset("diag_manager_single_tile.nc", decode_times=True)
    assert "var_2d" in ds
    np.testing.assert_array_equal(
        ds["var_2d"].shape, (ntimesteps, nx, ny)
    )
    assert ds["var_2d"].dims == ("time", "y", "x")
    assert ds["var_2d"].attrs["units"] == "muntin"
    assert ds["var_3d"].dims == ("time", "z", "y", "x")
    assert ds["var_3d"].attrs["units"] == "muntin"
    assert ds["time"].shape == (ntimesteps,)
    assert ds["time"].dims == ("time",)
    assert ds["time"].values[0] == cftime.DatetimeNoLeap(2, 1, 1, 2, 1, 1)
    assert ds["time"].values[1] == cftime.DatetimeNoLeap(2, 1, 1, 3, 1, 1) 
    assert ds["time"].values[2] == cftime.DatetimeNoLeap(2, 1, 1, 4, 1, 1) 
    np.testing.assert_array_equal(ds["var_2d"].values[0,:,:], var2_global.transpose())
    np.testing.assert_array_equal(ds["var_2d"].values[1,:,:], var2_global.transpose())
    np.testing.assert_array_equal(ds["var_2d"].values[2,:,:], var2_global.transpose())
    # data is transposed when passed into fortran
    np.testing.assert_array_equal(ds["var_3d"].values[0,:,:,:], var3_global.transpose())
    np.testing.assert_array_equal(ds["var_3d"].values[1,:,:,:], var3_global.transpose())
    np.testing.assert_array_equal(ds["var_3d"].values[2,:,:,:], var3_global.transpose())

    fms.end()

if __name__ == "__main__":
    test_dm_monitor_single_tile()