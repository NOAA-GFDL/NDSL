
<h1 style="color:#185FA5;">
Numerical Validation & Benchmarking
</h1>

!!! bug
    A one-liner of bench & validation results

<h2 style="color:#185FA5;">
Table of results
</h2>

| Science Code               | Numerical Validation(C24) | Scientific Validation(C180) | CPU Benchmark (C180) |   GPU Benchmark (C180) | GPU Benchmark (C720) |
| ------------               | --------------------      | ---------------------       | -------------        |   -------------        | ---------------      |
| Dynamics (Pace)            | ❌ |          ❌                 |          ❌          |          ❌            |         ❌           |
| Dynamics (GEOS v11.4.2)    |        ❌                 |          ❌                 |          ❌          |          ❌            |         ❌           |
| Moist - Microphysics       |        ❌                 |          ❌                 |          ❌          |          ❌            |         ❌           |
| Moist - Shallow Convection |        ❌                 |          ❌                 |          ❌          |          ❌            |         ❌           |

C24 resolutions have 72 levels. All other resolutions have 137 levels.

<h2 style="color:#185FA5;">
Methodology
</h2>

!!! bug
    Expand

<h3 style="color:#185FA5;">
Validation
</h3>

_Numerical validation:_

Numerical validation is done comparing a single time step between Fortran and NDSL on the CPU. It was the base of the porting.

<INSERT PPT IMAGE TO SHOWCASE TRANSLATE TEST>

- Both codes run with `-O0`, e.g., with compiler optimization turned off.
- A multi-modal metric is used for measuring difference that combines absolute and relative differences, and ULP measurement.
- Differences are expected (compilers, different codes) and reasonable thresholding will be used.

_Scientific validation:_

Scientific validation is done by comparing the results of 7 days of GEOS runs.

<!-- See more details and discussion in the [overview](../performance_and_benchmarking/benchmark_overview.md) -->

<TODO INSERT EXPLANATION ON METRICS USED RMSE, ETC.>

<h3 style="color:#185FA5;">
Benchmark
</h3>

Benchmark is done both on CPU and GPU at C180 L137. To showcase the difference in device bandwidth, we also run GPU on C720 L137.

Benchmark are done online in GEOS but measure several performance which are all interconnected

<TODO INSERT GRAPH SHOWING GEOS / COMP / F-PY INTERFACE / DATA MANIP / NUMERICS >


<h3 style="color:#185FA5;">
Hardware
</h3>

NCCS's Discover A100 partition, referred as **"Discover"**, per node:

- 4x A100 GPUs – 40 GB (released 2021)
- 1x EPYC 7402 – 96 cores (released 2020)
- Dual HDR Infiniband 2x200 Gbps

NCCS's PRISM GH partition, referred as **"Prism GH"**, per node:

- GH200: 1x H100 (96 GB HBME3) + 1 Grace (72 cores @ 2GHz- 480 GB LPPDR5) on the same die (released 2023)
- Dual HDR Infiniband 2x100 Gbps

<h3 style="color:#185FA5;">
Software stack
</h3>

!!! bug
    TODO

<h2 style="color:#185FA5;">
Data
</h2>
The data used for numerical validation are available on NCCS datashare:

- [Dynamics (Pace)](https://portal.nccs.nasa.gov/datashare/astg/smt/pace-regression-data/)
- [Dynamics (GEOS v11.4.2)](https://portal.nccs.nasa.gov/datashare/astg/smt/geos-fp/translate/11.5.2/x86_GNU/Dycore/TBC_C24_L72_Debug/)
- [Moist (GEOS v11.8.1)](https://portal.nccs.nasa.gov/datashare/astg/smt/geos-fp/translate/11.5.2/arm_clang/)

<h2 style="color:#185FA5;">
Overviews and earlier results
</h2>

<!-- - [Validation - overview](./validation_overview.md) -->
- [Benchmark - Overview](../performance_and_benchmarking/benchmark_overview.md)
- [GFDL1M Microphysics](./early_microphys.md)
- [UW Shallow Convection](./early_uw.md)
