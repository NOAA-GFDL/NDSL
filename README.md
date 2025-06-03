[![Lint](https://github.com/NOAA-GFDL/NDSL/actions/workflows/lint.yaml/badge.svg?branch=develop)](https://github.com/NOAA-GFDL/NDSL/actions/workflows/lint.yaml)
[![Unit tests](https://github.com/NOAA-GFDL/NDSL/actions/workflows/unit_tests.yaml/badge.svg?branch=develop)](https://github.com/NOAA-GFDL/NDSL/actions/workflows/unit_tests.yaml)

# NOAA/NASA Domain Specific Language middleware

NDSL is a middleware for climate and weather modelling developed jointly by NOAA and NASA. The middleware brings together [GT4Py](https://github.com/GridTools/gt4py/) (the `cartesian` flavor), ETH CSCS's stencil DSL, and [DaCe](https://github.com/spcl/dace/), ETH SPCL's data flow framework, both developed for high-performance and portability. On top of those pillars, NDSL deploys a series of optimized APIs for common operations (Halo exchange, domain decomposition, MPI, ...), a set of bespoke optimizations for the models targeted by the middleware and tools to port existing models.

## Batteries-included for FV-based models

Historically, NDSL was developed to port the FV3 dynamical core on the cubed-sphere. Therefore, the middleware ships with ready-to-execute specialization for models based on cubed-sphere grids and FV-based models in particular.

## Quickstart

Currently, NDSL requires Python version `3.11.x`. All other dependencies installed during package installation. We recommend using virtual (or conda) environment.

```shell
# We have submodules for GT4Py and DaCe. Don't forget to pull them
git clone --recurse-submodules git@github.com:NOAA-GFDL/NDSL.git

cd NDSL/

# We strongly recommend using a virtual environment (or conda)
python -m venv .venv/
source ./venv/bin/activate

# Choose pip install -e .[develop] if you'd like to contribute
pip install .[demos]
```

Now, checkout [examples/NDSL](./examples/NDSL/) and ran through the Jupyter notebooks. Note that you have to install NDSL locally, as it is not available on `pypi`.

## The slightly longer version

NDSL is under active development and may only work with specific setups. This is what we know works for us.

### Requirements and supported compilers

The run the CPU backends you will need:

- Python: 3.11.x
- CXX compiler:  GNU 11.2+
- Libraries: MPI

To run the GPU backends, you'll need:

- Python: 3.11.x
- CXX compiler:  GNU 11.2+
- Libraries: MPI compiled with CUDA support
- CUDA 11.2+
- Python package:
  - `cupy` (latest with proper driver support [see install notes](https://docs.cupy.dev/en/stable/install.html))

### Installation options

See [quickstart](#quickstart) above on how to pull and setup a virtual environment. The packages has a few options:

- `ndsl[test]`: extra dependencies to run tests (based on `pytest`)
- `ndsl[demos]`: extra dependencies to run [NDSL examples](./examples/NDSL/)
- `ndsl[docs]`: extra dependencies to build the docs
- `ndsl[develop]`: installs tools for development, docs, and tests.

### Running tests

Tests are available via `pytest` (don't forget to install the `test` or `develop` extras). Before you run tests, make sure to create expected input files:

```bash
python tests/grid/generate_eta_files.py
```

To run serial tests on CPU (GPU tests also run if `cupy` is available)

```bash
pytest tests/
```

To run parallel tests on CPU (GPU tests also run if `cupy` is available)

```bash
mpirun -np 6 pytest tests/mpi
```

## Development

### Code/contribution guidelines

1. Code quality is enforced by `pre-commit` (which is part of the "develop" extra). Run `pre-commit install`  to install the pre-commit hooks locally or make sure to run `pre-commit run -a`  before submitting a pull request.
2. While we don't strictly enforce type hints, we add them on new code.
3. Pull requests have to merged as "squash merge" to keep the `git` history clean.

### Documentation

We are using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), which allows us to write the docs in Markdown files and optionally serve it as a static site.

To view the documentation, install NDSL with the `docs` or `develop` extras. Then  run the following:

```bash
mkdocs serve
```

Contributing to the documentation is straight forward:

1. Add and/or change files in the [docs/](./docs/) folder as necessary.
2. [Optional] If you have changes to the navigation, modify [mkdocs.yml](mkdocs.yml).
3. [Optional] Start the development server and look how your changes are rendered.
4. Submit a pull request with your changes.

## Points of contact

- NOAA: Rusty Benson: rusty.benson -at- noaa.gov
- NASA: Florian Deconinck florian.g.deconinck -at- nasa.gov
