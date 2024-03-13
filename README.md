[![Lint](https://github.com/NOAA-GFDL/NDSL/actions/workflows/lint.yaml/badge.svg?branch=develop)](https://github.com/NOAA-GFDL/NDSL/actions/workflows/lint.yaml)
[![Unit tests](https://github.com/NOAA-GFDL/NDSL/actions/workflows/unit_tests.yaml/badge.svg?branch=develop)](https://github.com/NOAA-GFDL/NDSL/actions/workflows/unit_tests.yaml)

# NOAA/NASA Domain Specific Language middleware

NDSL is a middleware for climate and weather modelling developed jointly by NOAA and NASA. The middleware brings together [GT4Py](https://github.com/GridTools/gt4py/) (the `cartesian` flavor), ETH CSCS's stencil DSL, and [DaCE](https://github.com/spcl/dace/), ETH SPCL's data flow framework, both developed for high-performance and portability. On top of those pillars, NDSL deploys a series of optimized APIs for common operations (Halo exchange, domain decomposition, MPI...), a set of bespoke optimizations for the models targeted by the middleware and tools to port existing models.

## Battery-included for FV-based models

Historically NDSL was developed to port the FV3 dynamical core on the cube-sphere. Therefore, the middleware ships with ready-to-execute specialization for models based on cube-sphere grid and FV-based model in particular.

## Quickstart

Recommended Python is `3.11.x` all other dependencies will be pulled during install.

NDSL submodules `gt4py` and `dace` to point to vetted versions, use `git clone --recurse-submodule`.

NDSL is __NOT__ available on `pypi`. Installation of the package has to be local, via `pip install ./NDSL` (`-e` supported). The packages has a few options:

- `ndsl[test]`: installs the test packages (based on `pytest`)
- `ndsl[develop]`: installs tools for development and tests.

Tests are available via:

- `pytest -x test`: running CPU serial tests (GPU as well if `cupy` is installed)
- `mpirun -np 6 pytest -x test/mpi`: running CPU parallel tests (GPU as well if `cupy` is installed)
