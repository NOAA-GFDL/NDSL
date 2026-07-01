# Getting started

NDSL is a modern domain-specific language (DSL) for portable, high-performance atmospheric modeling. It let's you build portable atmospheric modeling workflows across CPUs, GPUs, and emerging computing architectures from a single codebase.

## Requirements

NDSL is a python DSL. You can install NDSL as a python package and write your code in python.

!!! note "Prerequisites"

    Before installing NDSL, make sure your environment includes:

    - A python environment. We support python versions `3.11`, `3.12`, `3.13`.
    - A compiler toolchain for C, C++. Optionally, a Fortran compiler if you're bridging to Fortran code.
    - MPI (can be installed via python, see below)

All other dependencies are installed with NDSL.

## Clone the repository

NDSL uses Git submodules for dependencies including GT4Py and DaCe, so be sure to clone recursively.

```shell
git clone --recurse-submodules git@github.com:NOAA-GFDL/NDSL.git
cd NDSL/
```

!!! note "Why clone the repository?"

    NDSL is currently not available on PyPI, so installation requires cloning the source repository.

## Create a virtual environment

We strongly recommend using a virtual environment. Create and activate it like this:

```shell
python -m venv .venv
source .venv/bin/activate
```

## Install MPI

If your system does not already provide MPI, you can install OpenMPI using `pip`.

```shell
pip install openmpi
```

## Install NDSL

Install NDSL along with demo dependencies.

```shell
pip install .[demos]
```

And that's it! If you are new to NDSL, we suggest you start be exploring the samples.

## Run the examples

Launch the notebooks located under `examples/NDSL` to start experimenting with NDSL. 🚀
