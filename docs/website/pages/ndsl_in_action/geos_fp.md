# GEOS-FP Results

Validation and early benchmarking results from GEOS-FP integrations, with individual physics schemes replaced by their NDSL implementations. All other model components remain in their original Fortran form.

Validation uses 7-day GEOS-FP integrations at C180, C360, and C720 horizontal resolutions. Benchmark timings are measured at the Fortran interface level; GPU timings are recorded after device synchronization, so reported times include data movement overhead between CPU and GPU.

**Speedup** is calculated relative to the reference Fortran — positive means NDSL is faster, negative means NDSL is slower.

---

## UW Shallow Convection

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

## GFDL1M Microphysics

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
