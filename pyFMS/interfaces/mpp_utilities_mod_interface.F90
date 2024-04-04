module mpp_utilities_mod_interface

    use iso_c_binding

    implicit none

    public :: mpp_array_global_min_max_c_r4, mpp_array_global_min_max_c_r8

contains

subroutine mpp_array_global_min_max_c_r4(in_array,tmask,isd,jsd,isc,iec,jsc,jec,nk, g_min, g_max, &
    geo_x,geo_y,geo_z, xgmin, ygmin, zgmin, xgmax, ygmax, zgmax) bind(c, name='mpp_array_global_min_max_r4')

    integer(c_int),                             intent(in) :: isd,jsd,isc,iec,jsc,jec,nk
    real(c_double), dimension(isd:,jsd:,:),     intent(in) :: in_array
    real(c_double), dimension(isd:,jsd:,:),     intent(in) :: tmask
    real(c_double),                             intent(out):: g_min, g_max
    real(c_double), dimension(isd:,jsd:),       intent(in) :: geo_x,geo_y
    real(c_double), dimension(:),               intent(in) :: geo_z
    real(c_double),                             intent(out):: xgmin, ygmin, zgmin, xgmax, ygmax, zgmax

    call mpp_array_global_min_max(in_array,tmask,isd,jsd,isc,iec,jsc,jec,nk, g_min, g_max, &
    geo_x,geo_y,geo_z, xgmin, ygmin, zgmin, xgmax, ygmax, zgmax)

end subroutine mpp_array_global_min_max_c_r4

subroutine mpp_array_global_min_max_c_r8(in_array,tmask,isd,jsd,isc,iec,jsc,jec,nk, g_min, g_max, &
    geo_x,geo_y,geo_z, xgmin, ygmin, zgmin, xgmax, ygmax, zgmax) bind(c, name='mpp_array_global_min_max_r8')

    integer(c_int),                             intent(in) :: isd,jsd,isc,iec,jsc,jec,nk
    real(c_double), dimension(isd:,jsd:,:),     intent(in) :: in_array
    real(c_double), dimension(isd:,jsd:,:),     intent(in) :: tmask
    real(c_double),                             intent(out):: g_min, g_max
    real(c_double), dimension(isd:,jsd:),       intent(in) :: geo_x,geo_y
    real(c_double), dimension(:),               intent(in) :: geo_z
    real(c_double),                             intent(out):: xgmin, ygmin, zgmin, xgmax, ygmax, zgmax

    call mpp_array_global_min_max(in_array,tmask,isd,jsd,isc,iec,jsc,jec,nk, g_min, g_max, &
    geo_x,geo_y,geo_z, xgmin, ygmin, zgmin, xgmax, ygmax, zgmax)

end subroutine mpp_array_global_min_max_c_r8

end module mpp_utilities_mod_interface