# GEOS Integration Results

Four components of GEOS has been ported to NDSL:
- Finite Volume Cubed (FV3) Dynamical Core
- Geophysical Fluid Dynamics Lab (GFDL) Single-Moment Microphysics
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

Finally, benchmarking is performed BY WAVING FLORIAN'S MAGICAL WAND AT THE GPU.



















---

## University of Washington Shallow Convection And Moist Turbulence Scheme

Work concluded August 2025.

### Validation

Histograms of differences between the reference Fortran and the CPU performance backend after 7 days of simulation. While most differences are centered near zero, there are non-zero outliers — particularly for relative humidity and wind — likely due to small numerical differences in the UW scheme that can grow over a 7-day run.

![Histograms of diagnostic variables differences](../img/hist__dace_cpu_C180_v_Fortran__sfc.png)

Temperature fields after 7 days, comparing the reference Fortran and the NDSL CPU backend. Patterns are very similar overall, with some local differences where temperature gradients are slightly displaced.

![Temperature Field - Reference Fortran](../img/UW_T_fortran_world_C180.png)

![Temperature Field - NDSL CPU (dace:cpu)](../img/UW_T_dacecpu_world_C180.png)

![Temperature Field Diffs - NDSL CPU (dace:cpu)](../img/UW_T_diff_world_C180.png)

### Benchmarks

| Resolution | Layout | Fortran | NDSL GPU (dace:gpu) | NDSL CPU (gt:cpu_kfirst) | Speedup (GPU) | Speedup (CPU) |
|---|---|---|---|---|---|---|
| C180 (~51 km) | 4×4 | 0.23 s | 0.04 s | 0.37 s | 6.29× | -1.62× |

---

## Geophysical Fluid Dynamics Lab (GFDL) Single-Moment Microphysics

Work concluded June 2026.

### Validation

Histograms of differences between the Fortran and NDSL simulations for temperature, relative humidity, zonal wind, and meridional wind after 7 days. Distributions are centered near zero, indicating strong agreement.

=== "C48 CPU"
    ![Histogram](../img/hist_gfdl1m_dace_cpu_C48_v_Fortran__sfc.png)

=== "C48 GPU"
    ![Histogram](../img/hist_gfdl1m_dace_cpu_C48_v_Fortran__sfc.png)

Spatial distribution of differences after 7 days. The largest differences are concentrated in regions of active weather near frontal systems and convective activity, where small positional shifts can produce large point-by-point differences even when overall structures remain similar.

=== "C48 T"
    ![Temp](../img/gcm_T_gfdl1m_c48_l72_7days.png)

=== "C48 QV"
    ![QV](../img/gcm_QV_gfdl1m_c48_l72_7days.png)

=== "C48 U"
    ![U](../img/gcm_U_gfdl1m_c48_l72_7days.png)

=== "C48 V"
    ![V](../img/gcm_V_gfdl1m_c48_l72_7days.png)

### Benchmarks

| Resolution | Layout | Fortran | NDSL GPU (st:dace:gpu) | NDSL CPU (st:dace:cpu:KJI) | Speedup (GPU) | Speedup (CPU) |
|---|---|---|---|---|---|---|
| C180 (~51 km) | 4×4 | 0.23 s | 0.04 s | 0.37 s | 6.29× | -1.62× |
| C360 (~xx km) | ?×? | x.xx s | x.xx s | x.xx s | x.xx× | -x.xx× |
| C720 (~xx km) | ?×? | x.xx s | x.xx s | x.xx s | x.xx× | -x.xx× |

---

## GF2020 Deep Convection

Work concluded June 2026.

### Validation

Histograms of differences between the Fortran and NDSL simulations for temperature, relative humidity, zonal wind, and meridional wind after 7 days. Distributions are centered near zero, indicating strong agreement.

=== "C48 CPU"
    ![Histogram](../img/hist_gf2020_dace_cpu_C48_v_Fortran__sfc.png)

=== "C48 GPU"
    ![Histogram](../img/hist_gf2020_dace_cpu_C48_v_Fortran__sfc.png)

Spatial distribution of differences after 7 days.

=== "C48 T"
    ![Temp](../img/gcm_T_gf2020_c48_l72_7days.png)

=== "C48 QV"
    ![QV](../img/gcm_QV_gf2020_c48_l72_7days.png)

=== "C48 U"
    ![U](../img/gcm_U_gf2020_c48_l72_7days.png)

=== "C48 V"
    ![V](../img/gcm_V_gf2020_c48_l72_7days.png)

### Benchmarks

| Resolution | Layout | Fortran | NDSL GPU (st:dace:gpu) | NDSL CPU (st:dace:cpu:KJI) | Speedup (GPU) | Speedup (CPU) |
|---|---|---|---|---|---|---|
| C180 (~51 km) | 4×4 | 0.23 s | 0.04 s | 0.37 s | 6.29× | -1.62× |
| C360 (~xx km) | ?×? | x.xx s | x.xx s | x.xx s | x.xx× | -x.xx× |
| C720 (~xx km) | ?×? | x.xx s | x.xx s | x.xx s | x.xx× | -x.xx× |


# GEOS-SCM Results

Last updated: May 14th, 2026

Early scientific validation results running the SCM on Discover HPC:

- NDSL 2026.03.00
- GEOS v11.8.1

## Experiments

- `bomex`: [Barbados Oceanographic and Meteorological Experiment (BOMEX)](https://www.eol.ucar.edu/field_projects/bomex) — a standard benchmark case used to evaluate shallow cumulus convection over tropical oceans.
- `arm_97jun`: ARM Summer 1997 Intensive Observation Period — used to test parameterizations, specifically deep convection triggers, by analyzing cloud development and convective activity.
- `armtwp_ice`: [Tropical Warm Pool – International Cloud Experiment (TWP-ICE)](https://armgov.svcs.arm.gov/research/campaigns/twp2006twp-ice) — primarily used in the SCM community to evaluate how well models simulate convective ice microphysics.

## UW Shallow Convection

=== "`bomex` 6 hr, 72 lev"
    ![Hovmoller](../../img/scm_uw_bomex_hovmoller_gpu_72.png)

=== "`bomex` 6 hr, 181 lev"
    ![Hovmoller](../../img/scm_uw_bomex_hovmoller_gpu_181.png)

=== "`arm_97jun` 14 hr, 72 lev"
    ![Hovmoller](../../img/scm_uw_arm_97jun_hovmoller_gpu_72.png)

=== "`arm_97jun` 14 hr, 181 lev"
    ![Hovmoller](../../img/scm_uw_arm_97jun_hovmoller_gpu_181.png)

## GF2020 Deep Convection

=== "`armtwp_ice` 6 days, 72 lev"
    ![Timeseries](../../img/scm_gf_armtwp_ice_timeseries_gpu_72.png)
    ![Hovmoller](../../img/scm_gf_armtwp_ice_hovmoller_gpu_72.png)

=== "`armtwp_ice` 6 days, 181 lev"
    ![Timeseries](../../img/scm_gf_armtwp_ice_timeseries_gpu_181.png)
    ![Hovmoller](../../img/scm_gf_armtwp_ice_hovmoller_gpu_181.png)

## GFDL1M Microphysics

=== "`armtwp_ice` 6 days, 72 lev"
    ![Timeseries](../../img/scm_gfdl1m_armtwp_ice_timeseries_gpu_72.png)
    ![Hovmoller](../../img/scm_gfdl1m_armtwp_ice_hovmoller_gpu_72.png)

=== "`armtwp_ice` 6 days, 181 lev"
    ![Timeseries](../../img/scm_gfdl1m_armtwp_ice_timeseries_gpu_181.png)
    ![Hovmoller](../../img/scm_gfdl1m_armtwp_ice_hovmoller_gpu_181.png)

## Moist Physics

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
