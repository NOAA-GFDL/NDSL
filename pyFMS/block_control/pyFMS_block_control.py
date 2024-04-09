import ctypes as ct


# TODO: Structural non-functional code


class ix_type(ct.Structure):
    _fields_ = []


class pk_type(ct.Structure):
    _fields_ = []


class block_control_type(ct.Structure):

    _fields_ = []


def define_blocks(
    lib_fms,
    component,
    Block,
    isc,
    iec,
    jsc,
    jec,
    kpts,
    nx_block,
    ny_block,
    message,
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
    component,
    Block,
    isc,
    iec,
    jsc,
    jec,
    kpts,
    blksz,
    message,
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
