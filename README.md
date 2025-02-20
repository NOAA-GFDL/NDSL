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
- `ndsl[demos]`: installs extra requirements to run [NDSL exmpales](./examples/NDSL/)
- `ndsl[docs]`: installs extra requirements to build the docs
- `ndsl[develop]`: installs tools for development, docs, and tests.

Tests are available via:

- `pytest -x test`: running CPU serial tests (GPU as well if `cupy` is installed)
- `mpirun -np 6 pytest -x test/mpi`: running CPU parallel tests (GPU as well if `cupy` is installed)

## Requirements & supported compilers

For CPU backends:

- 3.11.x >= Python < 3.12.x
- Compilers:
  - GNU 11.2+
- Libraries:
  - Boost headers 1.76+ (no lib installed, just headers)

For GPU backends (the above plus):

- CUDA 11.2+
- Python package:
  - `cupy` (latest with proper driver support [see install notes](https://docs.cupy.dev/en/stable/install.html))
- Libraries:
  - MPI compiled with cuda support

## Development

### Code/contribution guidelines

TBD

### Documentation

We are using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), which allows us to write the docs in Markdown files and optionally serve it as a static site.

To view the documentation, install NDSL with the `docs` or `develop` extras. Then just run

```bash
mkdocs serve
```

Contributing to the documentation is straight forward:

1. Add and/or change files in the [docs/](./docs/) folder as necessary.
2. [Optional] If you have changes to the navigation, modify [mkdocs.yml](mkdocs.yml).
3. [Optional] Start the development server and look how your changes are rendered.
4. Submit a pull request with your changes.

## Point of Contacts

- NOAA: Rusty Benson: rusty.benson -at- noaa.gov
- NASA: Florian Deconinck florian.g.deconinck -at- nasa.gov
