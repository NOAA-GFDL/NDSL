import ctypes as ct


# TODO: Structural non-functional code


class ix_type(ct.Structure):

    _fields_ = ["ix", (ct.POINTER(ct.c_int) * 32) * 2]


class pk_type(ct.Structure):

    _fields_ = ["ii", ct.POINTER(ct.c_int) * 32, "jj", ct.POINTER(ct.c_int) * 32]


class block_control_type(ct.Structure):

    _fields_ = [
        "nx_block",
        ct.c_int,
        "ny_block",
        ct.c_int,
        "nbliks",
        ct.c_int,
        "isc",
        ct.c_int,
        "iec",
        ct.c_int,
        "jsc",
        ct.c_int,
        "jec",
        ct.c_int,
        "npz",
        ct.c_int,
        "ibs",
        ct.POINTER(ct.c_int) * 32,
        "ibe",
        ct.POINTER(ct.c_int) * 32,
        "jbs",
        ct.POINTER(ct.c_int) * 32,
        "jbe",
        ct.POINTER(ct.c_int) * 32,
        "ix",
        ix_type,
        "blksz",
        ct.POINTER(ct.c_int) * 32,
        "blkno",
        (ct.POINTER(ct.c_int) * 32) * 2,
        "ixp",
        (ct.POINTER(ct.c_int) * 32) * 2,
        "index",
        pk_type,
    ]


def define_blocks(
    lib_fms,
    component: str,
    Block: block_control_type,
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
        ct.byref(Block),
        ct.byref(isc),
        ct.byref(iec),
        ct.byref(jsc),
        ct.byref(jec),
        ct.byref(kpts),
        ct.byref(nx_block),
        ct.byref(ny_block),
        ct.byref(message),
    )


def define_blocks_packed(
    lib_fms,
    component: str,
    Block: block_control_type,
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
        ct.byref(Block),
        ct.byref(isc),
        ct.byref(iec),
        ct.byref(jsc),
        ct.byref(jec),
        ct.byref(kpts),
        ct.byref(blksz),
        ct.byref(message),
    )
