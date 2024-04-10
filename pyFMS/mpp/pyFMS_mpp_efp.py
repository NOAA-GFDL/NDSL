# import ctypes as ct


# TODO: Structural non-functional code


def increment_ints(lib_fms, int_sum, int2, prec_error):
    lib_fms.increment_ints(int_sum, int2, prec_error)


def increment_ints_faster(lib_fms, int_sum, r, max_mag_term):
    lib_fms.increment_ints_faster(int_sum, r, max_mag_term)


def carry_overflow(lib_fms, int_sum, prec_error):
    lib_fms.carry_overflow(int_sum, prec_error)


def regularize_ints(lib_fms, int_sum):
    lib_fms.regularize_ints(int_sum)


def mpp_query_efp_overflow_error(lib_fms):
    lib_fms.mpp_query_efp_overflow_error()


def mpp_reset_efp_overflow_error(lib_fms):
    lib_fms.mpp_reset_efp_overflow_error()


def mpp_efp_assign(lib_fms, EFP1, EFP2):
    lib_fms.mpp_efp_assign(EFP1, EFP2)


def mpp_efp_list_sum_across_PEs(lib_fms, EFPs, nval, errors):
    lib_fms.mpp_efp_list_sum_across_PEs(EFPs, nval, errors)


def mpp_reproducing_sum_r8_2d(
    lib_fms, array, isr, ier, jsr, jer, EFP_sum, reproducing, overflow_check, err
):
    lib_fms.mpp_reproducing_sum_r8_2d(
        array, isr, ier, jsr, jer, EFP_sum, reproducing, overflow_check, err
    )


def mpp_reproducing_sum_r4_2d(
    lib_fms, array, isr, ier, jsr, jer, EFP_sum, reproducing, overflow_check, err
):
    lib_fms.mpp_reproducing_sum_r4_2d(
        array, isr, ier, jsr, jer, EFP_sum, reproducing, overflow_check, err
    )


def mpp_reproducing_sum_r8_3d(lib_fms, array, isr, ier, jsr, jer, sums, EFP_sum, err):
    lib_fms.mpp_reproducing_sum_r8_3d(array, isr, ier, jsr, jer, sums, EFP_sum, err)


def real_to_ints(lib_fms, r, prec_error, overflow):
    lib_fms.real_to_ints(r, prec_error, overflow)


def ints_to_real(lib_fms, ints):
    lib_fms.ints_to_real(ints)


def mpp_efp_plus(lib_fms, EFP1, EFP2):
    lib_fms.mpp_efp_plus(EFP1, EFP2)


def mpp_efp_minus(lib_fms, EFP1, EFP2):
    lib_fms.mpp_efp_minus(EFP1, EFP2)


def mpp_efp_to_real(lib_fms, EFP1):
    lib_fms.mpp_efp_to_real(EFP1)


def mpp_efp_real_diff(lib_fms, EFP1, EFP2):
    lib_fms.mpp_efp_real_diff(EFP1, EFP2)


def mpp_real_to_efp(lib_fms, val, overflow):
    lib_fms.mpp_real_to_efp(val, overflow)
