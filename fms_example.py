import ctypes as ct
import numpy as np
import pyFMS

lib_fms =  ct.cdll.LoadLibrary('./pyFMS/libs/libpyFMS.so')

pyFMS.fms_init(lib_fms=lib_fms, local_comm=-999, alt_input_nml_path='None')


isd = ct.c_int(5)
jsd = ct.c_int(5)
isc = ct.c_int(0)
iec = ct.c_int(5)
jsc = ct.c_int(0)
jec = ct.c_int(5)
in_array = np.array(np.random.rand(iec,jec,5), dtype=ct.c_double)
tmask = np.array(np.random.rand(iec,jec,5), dtype=ct.c_double)
nk = ct.c_int(0)
g_min = ct.c_double()
g_max = ct.c_double()
geo_x = ct.c_double()
geo_y = ct.c_double()
geo_z = ct.c_double()
xgmin = ct.c_double()
ygmin = ct.c_double()
zgmin = ct.c_double()
xgmax = ct.c_double()
ygmax = ct.c_double()
zgmax = ct.c_double()

pyFMS.mpp_array_global_min_max(
    lib_fms,
    in_array,
    tmask,
    isd,
    jsd,
    isc,
    iec,
    jsc,
    jec,
    nk,
    g_min,
    g_max,
    geo_x,
    geo_y,
    geo_z,
    xgmin,
    ygmin,
    zgmin,
    xgmax,
    ygmax,
    zgmax,
)




