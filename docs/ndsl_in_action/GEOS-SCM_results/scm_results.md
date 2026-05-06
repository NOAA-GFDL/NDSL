# GEOS-SCM Results 
##### Last updated: May 1st, 2026


Early scientific validation results running the SCM on Discover HPC:

 - NDSL 2026.03.00
 - GEOS v11.8.1


## SCM experiments

- `bomex`: [Barbados Oceanographic and Meteorological Experiment (BOMEX)](https://www.eol.ucar.edu/field_projects/bomex); a standard, benchmark case used to evaluate shallow cumulus convection over tropical oceans.
- `arm_97jun`: ARM Summer 1997 Intensive Observation Period; used to test parameterizations, specifically deep convection triggers, by analyzing cloud development and convective activity.
- `armtwp_ice`: [Tropical Warm Pool – International Cloud Experiment (TWP-ICE)](https://armgov.svcs.arm.gov/research/campaigns/twp2006twp-ice); primarily used in the SCM community to evaluate how well models simulate convective ice microphysics.


## UW Shallow Convection

### `bomex` (6 hr, 72 levels)
![Hovmoller](../../img/scm_bomex_hovmoller_gpu_72.png)

### `bomex` (6 hr, 181 levels)
![Hovmoller](../../img/scm_bomex_hovmoller_gpu_181.png)

### `arm_97jun` (14 hr, 72 levels)
<!-- ![Profiles](../../img/scm_arm_97jun_profiles_uw72.png) -->

<!-- ![Timeseries](../../img/scm_arm_97jun_timeseries_uw72.png) -->

![Hovmoller](../../img/scm_arm_97jun_hovmoller_gpu_72.png)


### `arm_97jun` (14 hr, 181 levels)
<!-- ![Profiles](../../img/scm_arm_97jun_profiles_uw181.png) -->

<!-- ![Timeseries](../../img/scm_arm_97jun_timeseries_uw181.png) -->

![Hovmoller](../../img/scm_arm_97jun_hovmoller_gpu_181.png)

## GF2020 Deep Convection


### `armtwp_ice` (6 days, 72 levels)

![Timeseries](../../img/scm_GF_armtwp_ice_timeseries_gpu_72.png)

![Hovmoller](../../img/scm_GF_armtwp_ice_hovmoller_gpu_72.png)

### `armtwp_ice` (6 days, 181 levels)

![Timeseries](../../img/scm_GF_armtwp_ice_timeseries_gpu_181.png)

![Hovmoller](../../img/scm_GF_armtwp_ice_hovmoller_gpu_181.png)


## GFDL1M Microphysics


### `armtwp_ice` (6 days, 72 levels)

<!-- ![T](../../img/scm_armtwp_ice_multi_layer_T.png)

![QV](../../img/scm_armtwp_ice_multi_layer_QV.png)

![QL](../../img/scm_armtwp_ice_multi_layer_QL.png)

![CLOUD](../../img/scm_armtwp_ice_multi_layer_CLOUD.png)

![TSAIR](../../img/scm_armtwp_ice_surface_TSAIR.png)

![PS](../../img/scm_armtwp_ice_surface_PS.png)

![SH](../../img/scm_armtwp_ice_surface_SH.png) -->

![Timeseries](../../img/scm_armtwp_ice_timeseries_gpu_72.png)

![Hovmoller](../../img/scm_armtwp_ice_hovmoller_gpu_72.png)

### `armtwp_ice` (6 days, 181 levels)

![Timeseries](../../img/scm_armtwp_ice_timeseries_gpu_181.png)

![Hovmoller](../../img/scm_armtwp_ice_hovmoller_gpu_181.png)



## Moist Physics
### `bomex` (6 hr, 72 levels)

![Hovmoller](../../img/scm_moist_bomex_hovmoller_gpu_72.png)

### `arm_97jun` (14 hr, 72 levels)
![Hovmoller](../../img/scm_moist_arm_97jun_hovmoller_gpu_72.png)

### `armtwp_ice` (6 days, 72 levels)

![Timeseries](../../img/scm_moist_armtwp_ice_timeseries_gpu_72.png)

![Hovmoller](../../img/scm_moist_armtwp_ice_hovmoller_gpu_72.png)