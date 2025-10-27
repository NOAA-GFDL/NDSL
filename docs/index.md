# NDSL Documentation

NDSL is a middleware for climate and weather modelling developed jointly by NOAA and NASA. It allows atmospheric scientists to focus on what matters in model development and essentially decouples performance engineering from model development.

## Portable performance

NDSL brings together [GT4Py](https://github.com/GridTools/gt4py/) and [DaCe](https://github.com/spcl/dace/), two libraries developed for high-performance and portability. On top of those pillars, NDSL deploys a series of optimized APIs for common operations, e.g. halo exchange or domain decomposition, and tools to port existing models.

## Batteries-included for FV-based models

Historically, NDSL was developed to port the FV3 dynamical core on the cubed-sphere. Therefore, the middleware ships with ready-to-execute specialization for models based on cubed-sphere grids and FV-based models in particular.

Next: get [up and running](./quickstart.md).
