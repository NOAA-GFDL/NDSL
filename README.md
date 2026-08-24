[![Lint](https://github.com/NOAA-GFDL/NDSL/actions/workflows/lint.yaml/badge.svg?branch=develop)](https://github.com/NOAA-GFDL/NDSL/actions/workflows/lint.yaml)
[![Unit tests](https://github.com/NOAA-GFDL/NDSL/actions/workflows/unit_tests.yaml/badge.svg?branch=develop)](https://github.com/NOAA-GFDL/NDSL/actions/workflows/unit_tests.yaml)

# NOAA/NASA Domain Specific Language

NDSL is a modern domain-specific language (DSL) for portable, high-performance atmospheric modeling. It let's you build portable atmospheric modeling workflows across CPUs, GPUs, and emerging computing architectures from a single codebase. NDSL is developed jointly by NOAA and NASA.

## Quickstart

As a user, with `uv` as project and dependency manager, it's quick to get up and running.

```shell
# Setup a project
uv init --python=3.12 example/
cd example/
uv add git+https://github.com/noaa-gfdl/ndsl.git --tag 2026.08.00 --extra openmpi
# Note: drop `--extra openmpi` if you want to use your system's MPI

# Generate & run NDSL code
uv run ndsl-gencode --exec --output ndsl_example.py
uv run ndsl_example.py
```

Check [our documentation](https://noaa-gfdl.github.io/NDSL/) for a more detailed setup guide.

## Batteries-included for FV-based models

Historically, NDSL was developed to port the FV3 dynamical core on the cubed-sphere. Therefore, the middleware ships with ready-to-execute specialization for models based on cubed-sphere grids and FV-based models in particular.

## Backed by science

NDSL unites the Cartesian flavor of [GT4Py](https://github.com/GridTools/gt4py/) with [DaCe](https://github.com/spcl/dace/). On top of those pillars, NDSL deploys a series of optimized APIs for common operations (Halo exchange, domain decomposition, MPI, ...), a set of bespoke optimizations for the models targeted by the middleware and tools to port existing models.

[GT4Py](https://github.com/GridTools/gt4py/) is a stencil DSL developed by the ETH-affiliated [Swiss National Computing Center (CSCS)](https://www.cscs.ch/), developed for high-performance and portability. [DaCe](https://github.com/spcl/dace/) is [ETH SPCL's](https://spcl.inf.ethz.ch/) data flow framework and was used in projects that won the ACM Gordon Bell Prize for Climate Modelling.

## Development

NDSL is under active development. If you feel like helping, please reach out (see [points of contact](#points-of-contact) below).

```shell
$ git clone git@github.com:NOAA-GFDL/NDSL.git
$ cd NDSL/
$ uv sync
# `.venv` contains an editable install of NDSL and includes all dev tools (linter, docs, ...)
$ source .venv/bin/activate
# Run linting and type checks before every commit
(ndsl) $ pre-commit install
```

For more a complicated developer setup installing multiple repositories in editable mode, please refer to [the team's dev setup guide](https://github.com/GEOS-ESM/SMT-Nebulae#dev-setup-guide).

### Running tests

Tests are written with `pytest` and located in the [tests/](./tests/) folder. To run serial tests on CPU, run

```shell
(ndsl) $ pytest -m "not parallel and not gpu and not zarr and not pyfms" tests/
```

To run parallel tests on CPU, run

```shell
(ndsl) $ mpirun -np 6 pytest -m "parallel and not gpu" tests/
```

add `--oversubscribe` if your computer provides less than 6 processors.

### Code/contribution guidelines

1. Code quality is enforced by `pre-commit` as part of CI. Run `pre-commit install` to install the pre-commit hooks locally or make sure to run `pre-commit run -a` before submitting a pull request.
2. While we don't strictly enforce type hints, we add them on new code.

### Documentation

Documentation is available [online](https://noaa-gfdl.github.io/NDSL/). We are using [Zensical](https://zensical.org/), which allows us to write the docs in Markdown files and serve it as a static site. To view the documentation locally, install NDSL in development mode (see above). Then, start the development server

```shell
(ndsl) $ zensical serve
```

and view the documentation at [localhost:8000](http://localhost:8000).

Typo in the docs? Contributing to the documentation is straight forward:

1. Add and/or change files in the [docs/](./docs/) folder as necessary.
2. [Optional] If you have changes to the navigation, modify [zensical.toml](./zensical.toml).
3. [Optional] Start the development server and look how your changes are rendered.
4. Submit a pull request with your changes.

## Points of contact

Primary contacts:

- NOAA: Rusty Benson: rusty.benson -at- noaa.gov
- NASA: Florian Deconinck florian.g.deconinck -at- nasa.gov

Visit [our documentation](https://noaa-gfdl.github.io/NDSL/community/) to see the full team and join the community.
