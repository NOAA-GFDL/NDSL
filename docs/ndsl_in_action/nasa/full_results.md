# GEOS Integration Results

Four components of GEOS has been ported to NDSL:
- Finite Volume Cubed (FV3) Dynamical Core
- Geophysical Fluid Dynamics Lab Single-Moment Microphysics
- Grell-Freitas Convection Parameterization
- University of Washington Shallow Convection And Moist Turbulence Scheme

These ports maintain numerical accuracy, and strive for numerical equivalence. Throughout the process, redundant code was consolidated, unused code was removed, and
poorly organized code was restructured. These improvements often depart from numerical equivalence; however, it has been verified that - at a given timestep - a field
has errors of no more than six orders of magnitude less than the observed value or 10^-11 (whichever is greater), and if such a condition is met it can be said
that numerical accuracy is maintained.

Ported code is tested for numerical accuracy by running "translate tests" - a numerical comparison of the ported code and the original Fortran code on a single timestep
in isolation from the rest of the model. Validation has been performed using Single Column Model (SCM) runs and General Cirtulation Model (GCM) runs. For SCM experiments,
the ported code is verified independently on each of the first 20 timesteps. For GCM runs (at C24 horizontal resolution), only the first ten timesteps are verified.
A successful, or "passed" test signals that numerical accuracy has been maintained in the ported code.

Next, scientific validity is assesed by comparing the output of a full model runs (SCM or GCM) using either the Fortran and NDSL component. This analysis has been
performed with three SCM experiments ("bomex", "armtwp-ice", and "armtwp-july97") and a GCM run initialized on 14 April 2000. SCM experiments ran for their entire
durations, while GCM runs were limited to 10 days.


More information on the benchmarking process coming soon...

---

## Geophysical Fluid Dynamics Lab Single-Moment Microphysics

### Scientific Validation

=== "SCM"

    === "`armtwp_ice` 6 days, 72 lev"
        ![Timeseries](../../img/scm_gfdl1m_armtwp_ice_timeseries_gpu_72.png)
        ![Hovmoller](../../img/scm_gfdl1m_armtwp_ice_hovmoller_gpu_72.png)

    === "`armtwp_ice` 6 days, 181 lev"
        ![Timeseries](../../img/scm_gfdl1m_armtwp_ice_timeseries_gpu_181.png)
        ![Hovmoller](../../img/scm_gfdl1m_armtwp_ice_hovmoller_gpu_181.png)

=== "GCM"

    === "C180 T"
        ![Temp](../../img/gcm_T_gfdl1m_c180_l72_7days.png)

    === "C180 QV"
        ![QV](../../img/gcm_QV_gfdl1m_c180_l72_7days.png)

    === "C180 U"
        ![U](../../img/gcm_U_gfdl1m_c180_l72_7days.png)

    === "C180 V"
        ![V](../../img/gcm_V_gfdl1m_c180_l72_7days.png)

### Performance Benchmarks
Coming soon...
<!-- | Resolution | Layout | Fortran | NDSL GPU (st:dace:gpu) | NDSL CPU (st:dace:cpu:KJI) | Speedup (GPU) | Speedup (CPU) |
|---|---|---|---|---|---|---|
| C180 (~51 km) | 4×4 | 0.23 s | 0.04 s | 0.37 s | 6.29× | -1.62× |
| C360 (~xx km) | ?×? | x.xx s | x.xx s | x.xx s | x.xx× | -x.xx× |
| C720 (~xx km) | ?×? | x.xx s | x.xx s | x.xx s | x.xx× | -x.xx× |

--- -->

## University of Washington Shallow Convection And Moist Turbulence Scheme

### Scientific Validation

=== "SCM"

    === "`bomex` 6 hr, 72 lev"
        ![Hovmoller](../../img/scm_uw_bomex_hovmoller_gpu_72.png)

    === "`bomex` 6 hr, 181 lev"
        ![Hovmoller](../../img/scm_uw_bomex_hovmoller_gpu_181.png)

    === "`arm_97jun` 14 hr, 72 lev"
        ![Hovmoller](../../img/scm_uw_arm_97jun_hovmoller_gpu_72.png)

    === "`arm_97jun` 14 hr, 181 lev"
        ![Hovmoller](../../img/scm_uw_arm_97jun_hovmoller_gpu_181.png)

=== "GCM"

    === "C180 T"
        ![Temp](../../img/gcm_T_uw_c180_l72_7days.png)

    === "C180 QV"
        ![QV](../../img/gcm_QV_uw_c180_l72_7days.png)

    === "C180 U"
        ![U](../../img/gcm_U_uw_c180_l72_7days.png)

    === "C180 V"
        ![V](../../img/gcm_V_uw_c180_l72_7days.png)

### Performance Benchmarks
Coming soon...
<!-- | Resolution | Layout | Fortran | NDSL GPU (dace:gpu) | NDSL CPU (gt:cpu_kfirst) | Speedup (GPU) | Speedup (CPU) |
|---|---|---|---|---|---|---|
| C180 (~51 km) | 4×4 | 0.23 s | 0.04 s | 0.37 s | 6.29× | -1.62× |

--- -->

## Grell-Freitas Convection Parameterization

### Scientific Validation

=== "SCM"

    === "`armtwp_ice` 6 days, 72 lev"
        ![Timeseries](../../img/scm_gf_armtwp_ice_timeseries_gpu_72.png)
        ![Hovmoller](../../img/scm_gf_armtwp_ice_hovmoller_gpu_72.png)

    === "`armtwp_ice` 6 days, 181 lev"
        ![Timeseries](../../img/scm_gf_armtwp_ice_timeseries_gpu_181.png)
        ![Hovmoller](../../img/scm_gf_armtwp_ice_hovmoller_gpu_181.png)

=== "GCM"

    === "C48 T"
        ![Temp](../../img/gcm_T_gf2020_c48_l72_7days.png)

    === "C48 QV"
        ![QV](../../img/gcm_QV_gf2020_c48_l72_7days.png)

    === "C48 U"
        ![U](../../img/gcm_U_gf2020_c48_l72_7days.png)

    === "C48 V"
        ![V](../../img/gcm_V_gf2020_c48_l72_7days.png)

### Performance Benchmarks
Coming soon...
<!-- | Resolution | Layout | Fortran | NDSL GPU (st:dace:gpu) | NDSL CPU (st:dace:cpu:KJI) | Speedup (GPU) | Speedup (CPU) |
|---|---|---|---|---|---|---|
| C180 (~51 km) | 4×4 | 0.23 s | 0.04 s | 0.37 s | 6.29× | -1.62× |
| C360 (~xx km) | ?×? | x.xx s | x.xx s | x.xx s | x.xx× | -x.xx× |
| C720 (~xx km) | ?×? | x.xx s | x.xx s | x.xx s | x.xx× | -x.xx× | -->

## Moist Physics

Work concluded June 2026.

### Validation

=== "SCM"

    === "`bomex` 6 hr, 72 lev"
        ![Hovmoller](../../img/scm_moist_bomex_hovmoller_gpu_72.png)

    === "`bomex` 6 hr, 181 lev"
        ![Hovmoller](../../img/scm_moist_bomex_hovmoller_gpu_181.png)

    === "`arm_97jun` 14 hr, 72 lev"
        ![Hovmoller](../../img/scm_moist_arm_97jun_hovmoller_gpu_72.png)

    === "`arm_97jun` 14 hr, 181 lev"
        ![Hovmoller](../../img/scm_moist_arm_97jun_hovmoller_gpu_181.png)

    === "`armtwp_ice` 6 days, 72 lev"
        ![Timeseries](../../img/scm_moist_armtwp_ice_timeseries_gpu_72.png)
        ![Hovmoller](../../img/scm_moist_armtwp_ice_hovmoller_gpu_72.png)

### Benchmarks
Coming soon...
<!-- | Resolution | Layout | Fortran | NDSL GPU (st:dace:gpu) | NDSL CPU (st:dace:cpu:KJI) | Speedup (GPU) | Speedup (CPU) |
|---|---|---|---|---|---|---|
| C180 (~51 km) | 4×4 | 0.23 s | 0.04 s | 0.37 s | 6.29× | -1.62× |
| C360 (~xx km) | ?×? | x.xx s | x.xx s | x.xx s | x.xx× | -x.xx× |
| C720 (~xx km) | ?×? | x.xx s | x.xx s | x.xx s | x.xx× | -x.xx× | -->
