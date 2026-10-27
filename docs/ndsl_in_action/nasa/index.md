# NDSL at NASA

NASA has used NDSL to accelerate targeted portions of the Goddard Earth Observing System (GEOS). To allow for optimal performance while ensuring numerical accuracy of
all new code, this process has been completed in two distinct steps: **numerical validation** and **benchmarking**.

##### Numerical Validation

The first step, porting the Fortran code to NDSL, was completed with two mandates: ensure numerical accuracy is maintained, while eliminating or rewriting unused,
redundant, or poorly organized code. In most cases, these dual mandates were fulfilled in their entirety; however, on occasion small errors (generally no more than
100 ULP, or roughly six orders of magnitude less than the observed value) on a small fraction (<5%) of grid points were considered acceptable.

Numerical validation has been performed by comparing the results of two GEOS runs: one using the original Fortran and another using the integrated NDSL component.
For the purposes of this work, numerical validation has been performed on both single column model (SCM) runs and general circulation model (GCM) runs. To avoid
conflating errors associated with different hardware types with true porting errors, numerical validation is performed with all optimization turned off - for both
the Fortran and NDSL.

A selection of results can be seen below. More details on how numerical validation is performed and access to full results and associated data sets can be
found [here](./full_results.md).

##### Benchmarking

The following table summarizes performance numbers for the Dynamics with a quick breakdown. We also show performance numbers for a microphysics parametrization ("GFDL 1M").

|  |  | Fortran (reference) |  NDSL `v2026.10.x` |
| - | - | - | - |
| Dynamics | | 257 |  231 |
|   | Acoustics | 153 | 105 |
|   | Tracers |  42 |  43 |
|   | Remapping |  58 |  71 |
|  | Missing from core | - |  10 |
|  | Interface to Fortran | - | 25 |
| GFDL 1M (microphysics) | | 170 | 124 |

This is for a single tile of the size 24x24 with 72 vertical levels. Timings are time to solution and given in milliseconds (less is better).
