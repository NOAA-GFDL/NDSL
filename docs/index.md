# NDSL Documentation

NDSL allows atmospheric scientists to write focus on what matters in model development and hides away the complexities of coding for a super computer.

## Quick Start

Python `3.11.x` is required for NDSL and all its third party dependencies for installation.

NDSL submodules `gt4py` and `dace` to point to vetted versions, use `git clone --recurse-submodule` to update the git submodules.

NDSL is **NOT** available on `pypi`. Installation of the package has to be local, via `pip install ./NDSL` (`-e` supported). The packages have a few options:

- `ndsl[test]`: installs the test packages (based on `pytest`)
- `ndsl[develop]`: installs tools for development and tests.

NDSL uses pytest for its unit tests, the tests are available via:

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

## NDSL installation and testing

NDSL is not available at `pypi`, it uses

```bash
pip install NDSL
```

to install NDSL locally.

NDSL has a few options:

- `ndsl[test]`: installs the test packages (based on `pytest`)
- `ndsl[develop]`: installs tools for development and tests.

Tests are available via:

- `pytest -x test`: running CPU serial tests (GPU as well if `cupy` is installed)
- `mpirun -np 6 pytest -x test/mpi`: running CPU parallel tests (GPU as well if `cupy` is installed)

## Configurations for Pace

Configurations for Pace to use NDSL with different backend:

- FV3_DACEMODE=Python[Build|BuildAndRun|Run] controls the full program optimizer behavior

  - Python: default, use stencil only, no full program optimization

  - Build: will build the program then exit. This _build no matter what_. (backend must be `dace:gpu` or `dace:cpu`)

  - BuildAndRun: same as above but after build the program will keep executing (backend must be `dace:gpu` or `dace:cpu`)

  - Run: load pre-compiled program and execute, fail if the .so is not present (_no hash check!_) (backend must be `dace:gpu` or `dace:cpu`)

- PACE_FLOAT_PRECISION=64 control the floating point precision throughout the program.

Install Pace with different NDSL backend:

- Shell scripts to install Pace using NDSL backend on specific machines such as Gaea can be found in `examples/build_scripts/`.
- When cloning Pace you will need to update the repository's submodules as well:

```bash
git clone --recursive https://github.com/ai2cm/pace.git
```

  or if you have already cloned the repository:

```bash
git submodule update --init --recursive
```

- Pace requires GCC > 9.2, MPI, and Python 3.8 on your system, and CUDA is required to run with a GPU backend.
You will also need the headers of the boost libraries in your `$PATH` (boost itself does not need to be installed).
If installed outside the standard header locations, gt4py requires that `$BOOST_ROOT` be set:

```bash
cd BOOST/ROOT
wget https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
tar -xzf boost_1_79_0.tar.gz
mkdir -p boost_1_79_0/include
mv boost_1_79_0/boost boost_1_79_0/include/
export BOOST_ROOT=BOOST/ROOT/boost_1_79_0
```

- We recommend creating a python `venv` or conda environment specifically for Pace.

```bash
python3 -m venv venv_name
source venv_name/bin/activate
```

- Inside of your pace `venv` or conda environment pip install the Python requirements, GT4Py, and Pace:

```bash
pip3 install -r requirements_dev.txt -c constraints.txt
```

- There are also separate requirements files which can be installed for linting (`requirements_lint.txt`) and building documentation   (`requirements_docs.txt`).
