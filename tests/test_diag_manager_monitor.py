import logging
from datetime import timedelta, datetime
from typing import List

import cftime
import numpy as np
import pytest
import xarray as xr
import mpi4py.MPI as MPI

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    LocalComm,
    Quantity,
    TilePartitioner,
    DiagManagerMonitor,
)

from pyfms import mpp_domains, mpp, fms

from pathlib import Path

import yaml

diag_config = {
    "title": "pace_diag_manager_test",
    "base_date": "1 1 1 0 0 0",
    "diag_files": [
        {
            "file_name": "pace_diagnostics",
            "freq": "15 seconds",
            "time_units": "seconds",
            "unlimdim": "time",
            "varlist": [
                {
                    "module": "atm_mod",
                    "var_name": "var1",
                    "long_name": "variable_number_one",
                    "reduction": "none",
                    "kind": "r8"
                }
            ]
        }
    ]
}


logger = logging.getLogger(__name__)

# init fms mpi and set up a simple domain
def fms_mpp_init(cubed_sphere:bool=False):
    nhalo = 1
    x = 16
    y = 16

    fms.init(localcomm=MPI.COMM_WORLD.py2f(), calendar_type=fms.NOLEAP)

    if not cubed_sphere:
        layout = [1, 1]
        io_layout = [1, 1]
        global_indices = [0, x-1, 0, y-1]
        halo = 1

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
    else:
        raise NotImplementedError("Cubed sphere FMS MPP init not implemented yet")

# initializes domain to use for ALL tests
domain_id = fms_mpp_init(cubed_sphere=False)


def _create_input():
    with open("diag_table.yaml", "w") as f:
        yaml.dump(diag_config, f, default_flow_style=False, sort_keys=False)
    text_content = "&diag_manager_nml\nuse_modern_diag=.true.\n/"
    with open("input.nml", "w", encoding="utf-8") as f:
        f.write(text_content)


# Simple test to ensure DiagManagerMonitor can injest the diag yaml and register fields and axes
#@pytest.mark.parametrize("do_send_data", [True, False])
def test_dm_monitor():

    _create_input()

    nx = 16
    ny = 16
    ntimesteps = 3

    buffer = {}

    communicator = CubedSphereCommunicator(
            comm = LocalComm(rank=0, total_ranks=1, buffer_dict=buffer),
            partitioner=TilePartitioner((1,1)),
    )
    communicator.tile

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
    monitor.set_model_times(init_time=start, end_time=end)

    monitor.register_axis(
        name="x",
        axis_data=np.arange(nx, dtype=np.float64),
        cart_name='x',
        long_name="x coordinate",
        units="m",
        not_xy=False,
    )
    monitor.register_axis(
        name="y",
        axis_data=np.arange(ny, dtype=np.float64),
        cart_name='y',
        long_name="y coordinate",
        units="m",
        not_xy=False,
    )

    monitor.register_field(
        module_name="atm_mod",
        field_name="var1",
        dims=["x","y"],
        units="m",
        long_name="variable one",
        timestep=step,
    )
    assert "x" in monitor.axes
    assert "y" in monitor.axes
    assert "var1" in monitor.fields

    for t in range(ntimesteps):
        # increment time and create the expected state dict with quantity data
        current_time = start + t * step
        field_q = Quantity(
            data=t * np.ones( (nx,ny), dtype=np.float64),
            dims=("x","y"),
            units="m",
            backend="debug",
        )
        state = {
            "time": current_time,
            "var1": field_q,
        }
        # send the state to the monitor and send it off to diag_manager
        monitor.store(state)

    monitor.cleanup()
    assert Path("pace_diagnostics.nc").exists()



# TODO fix the parametrizations once the test works
#@pytest.mark.parametrize("layout", [(1, 1), (1, 2), (4, 4)])
#@pytest.mark.parametrize(
#    "nt, time_chunk_size",
#    [pytest.param(1, 1, id="single_time"), pytest.param(5, 2, id="chunked_time")],
#)
#@pytest.mark.parametrize(
#    "shape, ny_rank_add, nx_rank_add, dims",
#    [
#        pytest.param((5, 4, 4), 0, 0, ("z", "y", "x"), id="cell_center"),
#        pytest.param(
#            (5, 4, 4), 1, 1, ("z", "y_interface", "x_interface"), id="cell_corner"
#        ),
#        pytest.param((5, 4, 4), 0, 1, ("z", "y", "x_interface"), id="cell_edge"),
#    ],
#)
#def test_monitor_store_multi_rank_state(
#    layout, nt, time_chunk_size, tmpdir, shape, ny_rank_add, nx_rank_add, dims, numpy
#):

# taken from netcdfmonitor test to make sure the generic monitor routines function as expected
@pytest.mark.skip("Skipping original test for now")
def test_diag_monitor_store_multi_rank_state():
    units = "m"
    backend = "debug"
    nz, ny, nx = (16, 16, 1)
    nt = 3
    layout = (1, 1)
    ny_rank = int(ny / layout[0]) # + ny_rank_add)
    nx_rank = int(nx / layout[1])  # + nx_rank_add)
    tile = TilePartitioner(layout)
    time = cftime.DatetimeJulian(2010, 6, 20, 6, 0, 0)
    timestep = timedelta(hours=1)
    total_ranks = 6 * layout[0] * layout[1]
    partitioner = CubedSpherePartitioner(tile)
    shared_buffer = {}
    monitor_list: List[DiagManagerMonitor] = []
    tmpdir = "./diag_manager_monitor_test/"
    dims = ("z", "y", "x")

    _create_input()

    for rank in range(total_ranks):
        communicator = CubedSphereCommunicator(
            partitioner=partitioner,
            comm=LocalComm(
                rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
            ),
        )
        # must eagerly initialize the tile object so that their ranks are
        # created in ascending order
        communicator.tile
        monitor_list.append(
            DiagManagerMonitor(
                path=tmpdir,
                communicator=communicator,
                time_chunk_size=1,
                domain_id=domain_id,
            )
        )
    # TODO constant storage
    for rank in range(total_ranks - 1, -1, -1):
        state = {
            "var_const1": Quantity(
                np.ones([nz, ny_rank, nx_rank]),
                dims=dims,
                units=units,
                backend=backend,
            ),
        }
    #    monitor_list[rank].store_constant(state)

    tile_gathered = []
    for i_t in range(nt):
        for rank in range(total_ranks - 1, -1, -1):
            state = {
                "time": time + i_t * timestep,
                "var1": Quantity(
                    np.ones([nz, ny_rank, nx_rank]),
                    dims=dims,
                    units=units,
                    backend=backend,
                ),
            }
            monitor_list[rank].store(state)
            tile_gathered.append(
                monitor_list[rank]._communicator.tile.gather_state(state)
            )

    for rank in range(total_ranks - 1, -1, -1):
        state = {
            "var_const2": Quantity(
                np.ones([nz, ny_rank, nx_rank]),
                dims=dims,
                units=units,
                backend=backend,
            ),
        }
    #    monitor_list[rank].store_constant(state)

    for monitor in monitor_list:
        monitor.cleanup()

    # TODO output checking
    #ds = xr.open_mfdataset(str(tmpdir / "state_*_tile*.nc"), decode_times=True)
    #assert "var1" in ds
    #np.testing.assert_array_equal(
        #ds["var1"].shape, (nt, 6, nz, ny + ny_rank_add, nx + nx_rank_add)
    #)
    #assert ds["var1"].dims == ("time", "tile") + dims
    #assert ds["var1"].attrs["units"] == units
    #assert ds["time"].shape == (nt,)
    #assert ds["time"].dims == ("time",)
    #assert ds["time"].values[0] == time
    #np.testing.assert_array_equal(ds["var1"].values, 1.0)

    #ds_const = xr.open_dataset(str(tmpdir / "constants_var_const1.nc"))
    #assert "var_const1" in ds_const
    #np.testing.assert_array_equal(
        #ds_const["var_const1"].shape, (6, nz, ny + ny_rank_add, nx + nx_rank_add)
    #)
    #assert ds_const["var_const1"].dims == ("tile",) + dims
    #assert ds_const["var_const1"].attrs["units"] == units
    #np.testing.assert_array_equal(ds_const["var_const1"].values, 1.0)
    #ds_const2 = xr.open_dataset(str(tmpdir / "constants_var_const2.nc"))
    #assert "var_const2" in ds_const2
    #np.testing.assert_array_equal(
        #ds_const2["var_const2"].shape, (6, nz, ny + ny_rank_add, nx + nx_rank_add)
    #)
    #assert ds_const2["var_const2"].dims == ("tile",) + dims
    #assert ds_const2["var_const2"].attrs["units"] == units
    #np.testing.assert_array_equal(ds_const2["var_const2"].values, 1.0)


