import logging
from datetime import timedelta, datetime
from typing import List

import cftime
import numpy as np
import pytest
import xarray as xr
import pdb

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    LocalComm,
    MPIComm,
    Quantity,
    TileCommunicator,
    TilePartitioner,
    DiagManagerMonitor,
)

from ndsl.initialization import SubtileGridSizer
from ndsl import QuantityFactory
from pyfms import mpp_domains, fms

from pathlib import Path

from mpi4py import MPI

import yaml


logger = logging.getLogger(__name__)

# shared parameters for the test functions
nx = 16
ny = 16
nz = 10
nhalo = 0
backend = "debug"
ntimesteps = 3


# init fms mpi and set up a simple domain
def fms_mpp_init(pes: int = 1):

    fms.init(localcomm=MPIComm()._comm.py2f(), calendar_type=fms.NOLEAP)

    layout = [1, pes]
    io_layout = [1, 1]
    global_indices = [0, nx-1, 0, ny-1]
    halo = nhalo 

    domain = mpp_domains.define_domains(
        global_indices=global_indices,
        layout=layout,
        whalo=halo,
        ehalo=halo,
        shalo=halo,
        nhalo=halo,
        xflags=mpp_domains.CYCLIC_GLOBAL_DOMAIN,
        yflags=mpp_domains.CYCLIC_GLOBAL_DOMAIN,
    )

    id_num = domain.domain_id
    mpp_domains.define_io_domain(
        domain_id=id_num,
        io_layout=io_layout,
    )
    return id_num

def _create_input(reduction: str = "none"):
    diag_config = {
        "title": "ndsl_diag_manager_test",
        "base_date": "1 1 1 0 0 0",
        "diag_files": [
            {
                "file_name": "ndsl_diag_test",
                "freq": "15 seconds",
                "time_units": "seconds",
                "unlimdim": "time",
                "varlist": [
                    {
                        "module": "atm_mod",
                        "var_name": "var1",
                        "long_name": "variable_number_one",
                        "reduction": reduction,
                        "kind": "r8"
                    },
                    {
                        "module": "atm_mod",
                        "var_name": "var2",
                        "long_name": "variable_number_one",
                        "reduction": reduction,
                        "kind": "r8"
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

    npes = MPIComm()._comm.Get_size()
    pe = MPIComm()._comm.Get_rank()
    if(pe == 0 or pe is None):
        _create_input()

    buffer = {}
    layout = (1, npes)

    # handles mpi/serial set up steps
    if npes > 1: 
        domain_id = fms_mpp_init(pes=npes)
        rank = MPIComm()._comm.Get_rank()
        print(f"intializing partitioner/communicator rank {rank} of {npes}")
        partitioner = TilePartitioner(layout=layout)
        communicator = TileCommunicator(MPIComm(), partitioner)
        communicator.tile
    else:
        domain_id = fms_mpp_init()
        npes = 1
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
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    quantity_factory = QuantityFactory(sizer, backend=backend)

    # pace will set up model start/end times and register axis info
    monitor = DiagManagerMonitor(
        path="./",
        communicator=communicator,
        domain_id=domain_id,
        time_chunk_size=1,
        precision=np.float32,
    )
    start = datetime(1, 1, 1, 0, 0, second=0)
    end = datetime(1, 1, 1, 0, 0, second=45)
    step = timedelta(seconds=15)

    monitor.set_timestep(step)
    monitor.set_end_time(end)

    monitor.register_axis(
        name="x",
        axis_data=np.arange(nx, dtype=np.float64),
        cart_name='x',
        long_name="x coordinate",
        units="m",
        not_xy=False,
        domain_id=domain_id,
    )
    monitor.register_axis(
        name="y",
        axis_data=np.arange(ny, dtype=np.float64),
        cart_name='y',
        long_name="y coordinate",
        units="m",
        not_xy=False,
        domain_id=domain_id,
    )
    monitor.register_axis(
        name="z",
        axis_data=np.arange(nz, dtype=np.float64),
        cart_name='z',
        long_name="vertical level",
        units="m",
        not_xy=False,
    )

    # fields will be registered in the component they are defined in (either pyFV3 or pySHiELD)
    monitor.register_field(
        module_name="atm_mod",
        field_name="var1",
        dims=["y","x"],
        units="m",
        init_time=start,
    )
    monitor.register_field(
        module_name="atm_mod",
        field_name="var2",
        dims=["z","y","x"],
        units="m",
        init_time=start,
    )
    assert "x" in monitor.axes
    assert "y" in monitor.axes
    assert "z" in monitor.axes
    assert "var1" in monitor.fields
    assert "var2" in monitor.fields

    # pace driver will call store for each timestep to send the data
    current_time = start
    for t in range(ntimesteps):
        current_time = current_time + step
        field_q1 = quantity_factory.full( dims=("y","x"), units="m", value=t, dtype=np.float64 )
        field_q2 = quantity_factory.zeros( dims=("z","y","x"), units="m", dtype=np.float64 )
        np_tmp = np.zeros( (nz,ny,nx))
        for (z,y,x), val in np.ndenumerate(np_tmp): # theres probably much better ways to do this..
            np_tmp.data[z,y,x] = x * 1000 + y + z*0.001
        field_q2.data = np_tmp
        state = {
            "time": current_time,
            "var1": field_q1,
            "var2": field_q2,
        }
        monitor.store(state)

    # cleanup writes and closes the file
    monitor.cleanup()
    
    ## check output!
    assert Path("ndsl_diag_test.nc").exists()
    ds = xr.open_mfdataset("ndsl_diag_test.nc", decode_times=True)
    assert "var1" in ds
    np.testing.assert_array_equal(
        ds["var1"].shape, (ntimesteps, nx, ny)
    )
    assert ds["var1"].dims == ("time", "x", "y")
    assert ds["var1"].attrs["units"] == "m"
    assert ds["var2"].dims == ("time", "x", "y", "z")
    assert ds["var2"].attrs["units"] == "m"
    assert ds["time"].shape == (ntimesteps,)
    assert ds["time"].dims == ("time",)
    assert ds["time"].values[0] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, second=15)
    assert ds["time"].values[1] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, second=30) 
    assert ds["time"].values[2] == cftime.DatetimeNoLeap(1, 1, 1, 0, 0, second=45) 
    np.testing.assert_array_equal(ds["var1"].values[0,:,:], 0.0)
    np.testing.assert_array_equal(ds["var1"].values[1,:,:], 1.0)
    np.testing.assert_array_equal(ds["var1"].values[2,:,:], 2.0)
    # data needs to get transposed when passed into fortran
    np.testing.assert_array_equal(ds["var2"].values[0,:,:,:], np_tmp.transpose())
    np.testing.assert_array_equal(ds["var2"].values[1,:,:,:], np_tmp.transpose())
    np.testing.assert_array_equal(ds["var2"].values[2,:,:,:], np_tmp.transpose())
