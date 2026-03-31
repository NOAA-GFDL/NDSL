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

# init fms mpi and set up a simple domain
def fms_mpp_init(pes: int = 1):
    fms.init(localcomm=MPIComm()._comm.py2f(), calendar_type=fms.NOLEAP)
    x = 8 
    y = 8
    layout = [1, 1] 
    io_layout = [1, 1]
    halo = 1
    tiles = 6
    domain_id = mpp_domains.define_cubic_mosaic(
        ni= [x for i in range(6)],
        nj= [y for i in range(6)],
        global_indices= [0, x-1, 0, y-1],
        layout=layout,
        ntiles=tiles,
        halo=halo,
        use_memsize=False,
    )
    mpp_domains.define_io_domain(
        domain_id=domain_id,
        io_layout=io_layout,
    )
    mpp_domains.set_current_domain(domain_id)
    return domain_id

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


# Simple test, uses a lat/lon grid and (1, npes) layout 
def test_dm_monitor(reduction):

    npes = MPIComm()._comm.Get_size()

    if npes % 6 != 0:
        raise RuntimeError("this test requires npes to be a multiple of 6 to run")

    _create_input(reduction)

    nx = 8
    ny = 8
    nz = 2
    nhalo = 0
    layout = (1, 1) # 1 pe per tile
    backend = "debug"
    ntimesteps = 3

    domain_id = fms_mpp_init(pes=npes)
    rank = MPIComm()._comm.Get_rank()
    print(f"intializing partitioner/communicator rank {rank} of {npes}")
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
        long_name="z coordinate",
        units="m",
        not_xy=True,
    )

    # fields will be registered in the component they are defined in (either pyFV3 or pySHiELD)
    monitor.register_field(
        module_name="atm_mod",
        field_name="var1",
        dims=["x","y"],
        units="m",
        long_name="variable one",
        init_time=start,
    )
    monitor.register_field(
        module_name="atm_mod",
        field_name="var2",
        dims=["x","y", "z"],
        units="m",
        long_name="variable two",
        init_time=start,
    )
    assert "x" in monitor.axes
    assert "y" in monitor.axes
    assert "var1" in monitor.fields

    # pace driver will call store for each timestep to send the data
    # store also currently advances the time in diag_manager and calls diag_send_complete
    for t in range(ntimesteps):
        current_time = start + t * step
        field_q1 = quantity_factory.full( dims=("x","y"), units="m", value=t, dtype=np.float64 )
        field_q2 = quantity_factory.full( dims=("x","y","z"), units="m", value=t*2, dtype=np.float64 )
        state = {
            "time": current_time,
            "var1": field_q1,
            "var2": field_q2
        }
        monitor.store(state)

    # cleanup writes and closes the file
    monitor.cleanup()

    fms.end()