# import ctypes as ct


# TODO: Structural non-functional code


def mpp_pset_init(lib_fms):
    lib_fms.mpp_pset_init()


def mpp_pset_create(lib_fms, npset, pset, stacksize, pelist, comID):
    lib_fms.mpp_pset_create(npset, pset, stacksize, pelist, comID)


def mpp_pset_delete(lib_fms, pset):
    lib_fms.mpp_pset_delete(pset)


def mpp_send_ptr_scalar(lib_fms, ptr, pe):
    lib_fms.mpp_send_ptr_scalar(ptr, pe)


def mpp_send_ptr_array(lib_fms, ptr, pe):
    lib_fms.mpp_send_ptr_array(ptr, pe)


def mpp_recv_ptr_scalar(lib_fms, ptr, pe):
    lib_fms.mpp_recv_ptr_scalar(ptr, pe)


def mpp_recv_ptr_array(lib_fms, ptr, pe):
    lib_fms.mpp_recv_ptr_array(ptr, pe)


def mpp_translate_remote_ptr(lib_fms, ptr, pe):
    lib_fms.mpp_translate_remote_ptr(ptr, pe)


def mpp_pset_sync(lib_fms, pset):
    lib_fms.mpp_pset_sync(pset)


def mpp_pset_broadcast(lib_fms, pset, a):
    lib_fms.mpp_pset_broadcast(pset, a)


def mpp_pset_broadcast_ptr_scalar(lib_fms, pset, ptr):
    lib_fms.mpp_pset_broadcast_ptr_scalar(pset, ptr)


def mpp_pset_broadcast_ptr_array(lib_fms, pset, ptr):
    lib_fms.mpp_pset_broadcast_ptr_array(pset, ptr)


def mpp_pset_check_ptr(lib_fms, pset, ptr):
    lib_fms.mpp_pset_check_ptr(pset, ptr)


def mpp_pset_segment_array(lib_fms, pset, ls, le, lsp, lep):
    lib_fms.mpp_pset_segment_array(pset, ls, le, lsp, lep)


def mpp_pset_stack_push(lib_fms, pset, ptr, len):
    lib_fms.mpp_pset_stack_push(pset, ptr, len)


def mpp_pset_stack_reset(lib_fms, pset):
    lib_fms.mpp_pset_stack_reset(pset)


def mpp_pset_print_chksum_1D(lib_fms, pset, caller, array):
    lib_fms.mpp_pset_print_chksum_1D(pset, caller, array)


def mpp_pset_print_chksum_2D(lib_fms, pset, caller, array):
    lib_fms.mpp_pset_print_chksum_2D(pset, caller, array)


def mpp_pset_print_chksum_3D(lib_fms, pset, caller, array):
    lib_fms.mpp_pset_print_chksum_3D(pset, caller, array)


def mpp_pset_print_chksum_4D(lib_fms, pset, caller, array):
    lib_fms.mpp_pset_print_chksum_4D(pset, caller, array)


def mpp_pset_print_stack_chksum(lib_fms, pset, caller):
    lib_fms.mpp_pset_print_stack_chksum(pset, caller)


def mpp_pset_get_root_pelist(lib_fms, pset, pelist, commID):
    lib_fms.mpp_pset_get_root_pelist(pset, pelist, commID)


def mpp_pset_root(lib_fms, pset):
    lib_fms.mpp_pset_root(pset)


def mpp_pset_numroots(lib_fms, pset):
    lib_fms.mpp_pset_numroots(pset)
