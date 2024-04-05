import ctypes as ct


# TODO: Structural non-functional code


def define_blocks(
    lib_fms,
    component,
    Block,
    isc: ct.c_int,
    iec: ct.c_int,
    jsc: ct.c_int,
    jec: ct.c_int,
    kpts: ct.c_int,
    nx_block: ct.c_int,
    ny_block: ct.c_int,
    message: ct.c_bool,
):
    """
    Sets up "blocks" used for OpenMP threading of column-based
    calculations using rad_n[x/y]xblock from coupler_nml
    """

    lib_fms.define_blocks(
        component,
        Block,
        ct.byref(isc),
        ct.byref(iec),
        ct.byref(jsc),
        ct.byref(jec),
        ct.byref(kpts),
        ct.byref(nx_block),
        ct.byref(ny_block),
        message,
    )


def define_blocks_packed(
    lib_fms,
    component,
    Block,
    isc: ct.c_int,
    iec: ct.c_int,
    jsc: ct.c_int,
    jec: ct.c_int,
    kpts: ct.c_int,
    blksz: ct.c_int,
    message: ct.c_bool,
):
    """
    Creates and populates a data type which is used for defining the
    sub-blocks of the MPI-domain to enhance OpenMP and memory performance.
    Uses a packed concept.
    """

    lib_fms.define_blocks_packed(
        component,
        Block,
        ct.byref(isc),
        ct.byref(iec),
        ct.byref(jsc),
        ct.byref(jec),
        ct.byref(kpts),
        ct.byref(blksz),
        message,
    )
