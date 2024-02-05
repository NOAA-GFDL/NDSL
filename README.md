# NOAA/NASA Domain Specific Language middleware

NDSL is a middleware for climate and weather modelling developped conjointment by NOAA and NASA. The middleware brings together [GT4Py](https://github.com/GridTools/gt4py/) (the `cartesian` flavor), an ETH CSCS's stencil DSL, and [DaCE](https://github.com/spcl/dace/), an ETH SPCL's data flow framework, both developped for high-performance and portability. On top of those pillars, NDSL deploys a series of optimized APIs for common operations (Halo exchange, domain decomposition, MPI...) and a set of bespoke optimizations for the models targeted by the middleware.

## Battery-included for FV-based models

Historically NDSL was developed to port the FV3 dynamical core on the cube-sphere. Therefore, the middleware ships with ready-to-execute specilization for models based on cube-sphere grid and FV-based model in particular.

## Quickstart

NDSL submodules `gt4py` and `dace` to point to vetted versions, use `git clone --recurse-submodule`.

NDSL is __NOT__ available on `pypi`. Installation of the package has to be local, via `pip install ./NDSL` (`-e` supported). The packages has a few options:

- `ndsl[test]`: installs the test packages (based on `pytest`)
- `ndsl[develop]`: installs tools for development and tests.

Tests are available via:

- `pytest -x test`: running CPU serial tests (GPU as well if `cupy` is installed)
- `mpirun -np 6 pytest -x test/mpi`: running CPU parallel tests (GPU as well if `cupy` is installed)
