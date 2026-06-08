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
