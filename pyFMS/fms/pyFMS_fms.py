import ctypes


def fms_init(lib_fms=None, local_comm=-999, alt_input_nml_path="None"):
    alt_input_nml_path = ctypes.c_char_p(alt_input_nml_path.encode("ascii"))
    local_comm = ctypes.c_int(local_comm)
    lib_fms.fms_init_c(ctypes.byref(local_comm), ctypes.byref(alt_input_nml_path))
