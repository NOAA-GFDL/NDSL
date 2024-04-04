import ctypes as ct

# TODO: Structural non-functional code

def mpp_memuse_begin(lib_fms):
    lib_fms.mpp_memuse_begin()

def mpp_memuse_end(lib_fms, text, unit):
    lib_fms.mpp_memuse_end(text, unit)

def mpp_print_memuse_stats(lib_fms, text, unit):
    lib_fms.mpp_print_memuse_stats(text, unit)

def mpp_mem_dump(lib_fms, memuse):
    lib_fms.mpp_mem_dump(memuse)