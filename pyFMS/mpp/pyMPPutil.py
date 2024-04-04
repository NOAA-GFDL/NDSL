import ctypes as ct


def mpp_array_global_min_max(
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
):
    """
    Compute and return the global min and max of an array
    and the corresponding lat-lon-depth locations.
    This algorithm works only for an input array that has a unique global
    max and min location. This is assured by introducing a factor that distinguishes
    the values of extrema at each processor.

    Arguments:
        lib_fms: shared object library containing FMS c-binded modules, routines, and functions
        in_array: 
        tmask: 
        isd:
        jsd:
        isc:
        iec:
        jsc:
        jec:
        nk:
        g_min:
        g_max:
        geo_x:
        geo_y:
        geo_z:
        xgmin:
        ygmin:
        zgmin:
        xgmax:
        ygmax:
        zgmax:

    Vectorized using maxloc() and minloc() intrinsic functions by
    Russell.Fiedler@csiro.au (May 2005).

    Modified by Zhi.Liang@noaa.gov (July 2005)

    Modified by Niki.Zadeh@noaa.gov (Feb. 2009)
    """
    lib_fms.mpp_array_global_min_max(
        ct.byref(in_array),
        ct.byref(tmask),
        ct.byref(isd),
        ct.byref(jsd),
        ct.byref(isc),
        ct.byref(iec),
        ct.byref(jsc),
        ct.byref(jec),
        ct.byref(nk),
        ct.byref(g_min),
        ct.byref(g_max),
        ct.byref(geo_x),
        ct.byref(geo_y),
        ct.byref(geo_z),
        ct.byref(xgmin),
        ct.byref(ygmin),
        ct.byref(zgmin),
        ct.byref(xgmax),
        ct.byref(ygmax),
        ct.byref(zgmax),
    )