from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    LocalComm,
    MPIComm,
    Quantity,
    QuantityFactory,
    TileCommunicator,
    TilePartitioner,
    DiagManagerMonitor,
    MPIComm,
    SubtileGridSizer
)

from pyfms import diag_manager, fms, mpp, mpp_domains

import yaml
from datetime import datetime, timedelta
import numpy as np

diag_config = {
    "title": "ndsl_diag_manager_test",
    "base_date": "2 1 1 1 1 1",
    "diag_files": [
        {
            "file_name": "ndsl_diag_script",
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
                    "output_name" : "var2_avg"
                },
                {
                    "module": "atm_mod",
                    "var_name": "var_3d",
                    "long_name": "variable_three_dee",
                    "reduction": "none",
                    "kind": "r8",
                    "output_name" : "var3_avg"
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

# tile parameters for quantities/domains 
nx = 8
ny = 8
nz = 2 
nhalo = 0
backend = "debug"
ntimesteps = 3
npes = MPIComm()._comm.Get_size() 
layout_fms = [1, npes] 
layout_ndsl = (npes, 1)
io_layout = [1, 2]
global_indices = [0, nx-1, 0, ny-1]

# set up fms domain 
fms.init(localcomm=MPIComm()._comm.py2f(), calendar_type=fms.NOLEAP)
domain = mpp_domains.define_domains(
    global_indices=global_indices,
    layout=layout_fms,
)
mpp_domains.set_current_domain(domain_id=domain.domain_id)
id_num = domain.domain_id
mpp_domains.define_io_domain(
    domain_id=id_num,
    io_layout=io_layout,
)
diag_manager.init(diag_model_subset=diag_manager.DIAG_ALL)

print(f"pe={mpp.pe()} domain obj={domain}")

# set up ndsl communicator, partitioner and quantity factory
rank = MPIComm()._comm.Get_rank()
print(f"intializing partitioner/communicator rank {rank} of {npes}")
partitioner = TilePartitioner(layout=layout_ndsl)
communicator = TileCommunicator(MPIComm(), partitioner)
communicator.tile

halo = nhalo 
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

# diagnostic time parameters
start = datetime(2, 1, 1, 1, 1, 1)
step = timedelta(seconds=3600)
end = start + ntimesteps * step

# register axes/field with pyfms diag_manager
x = np.arange(nx, dtype=np.float64)
id_x = diag_manager.axis_init(
    name="x",
    axis_data=x,
    units="point_E",
    cart_name="x",
    domain_id=domain.domain_id,
    long_name="point_E",
    set_name="atm",
)
y = np.arange(ny, dtype=np.float64)
id_y = diag_manager.axis_init(
    name="y",
    axis_data=y,
    units="point_N",
    cart_name="y",
    domain_id=domain.domain_id,
    long_name="point_N",
    set_name="atm",
)
z = np.arange(nz, dtype=np.float64)
id_z = diag_manager.axis_init(
    name="z",
    axis_data=z,
    units="point_Z",
    cart_name="z",
    long_name="point_Z",
    set_name="atm",
    not_xy=True,
)
print("axes are registered, ids=", id_x, id_y, id_z)
# TODO; issues registering
id_var3 = diag_manager.register_field_array(
    module_name="atm_mod",
    field_name="var_3d",
    dtype="float64",
    axes=[id_x, id_y, id_z],
    long_name="Var in a lon/lat domain",
    units="muntin",
    missing_value=-99.99,
    range_data=np.array([-1000.0, 1000.0], dtype=np.float64),
    init_time=start,
)
assert id_var3 != -1
id_var2 = diag_manager.register_field_array(
    module_name="atm_mod",
    field_name="var_2d",
    dtype="float64",
    axes=[id_x, id_y],
    long_name="Var in a lon/lat domain",
    units="muntin",
    missing_value=-99.99,
    range_data=np.array([-1000.0, 1000.0], dtype=np.float64),
    init_time=start,
)
assert id_var2 != -1

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

# send the data from each quantity
current_time = start
diag_manager.set_time_end(end)
for t in range(ntimesteps):
    current_time = current_time + step
    success = diag_manager.send_data(
        diag_field_id=id_var3,
        field=field_q2.field,
        time=current_time,
        convert_cf_order= True,
    )
    assert success
    success = diag_manager.send_data(
        diag_field_id=id_var2,
        field=field_q1.field,
        time=current_time,
        convert_cf_order= True,
    )
    assert success
    diag_manager.send_complete(step)

diag_manager.end(end)
fms.end()