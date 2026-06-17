# Getting Started

Install NDSL and start building portable atmospheric modeling
workflows across CPUs and GPUs in just a few minutes.

## Requirements


Prep

Before installing NDSL, make sure your environment includes:

Python 3.11

GNU compiler toolchain gcc / gfortran

We strongly recommend using either a virtual environment or Conda environment for installation.


Step 1

## Clone the Repository

NDSL uses Git submodules for dependencies including GT4Py and DaCe, so be sure to clone recursively.

`git clone --recurse-submodules git@github.com:NOAA-GFDL/NDSL.git`

`cd NDSL/`


Why clone the repository?

NDSL is currently not available on PyPI, so installation requires cloning the source repository.

Step 2

## Create a Virtual Environment

Create and activate a clean Python environment.

`python -m venv .venv`

`source .venv/bin/activate`

Optional

## Install MPI

If your system does not already provide MPI, you can install OpenMPI using pip.

`pip install openmpi`

Step 3

## Install NDSL

Install NDSL along with demo dependencies.

`pip install .[demos]`

Next Steps

## Run the Examples

Launch the notebooks located in:

`examples/NDSL`

Start experimenting with NDSL 🚀

## Supported Compilers

GNU Compiler Required

NDSL currently supports the GNU compiler toolchain only.

Using `clang` may result in undefined OpenMP flag errors.

For macOS users, `gcc-14` installed through Homebrew is known to work successfully.
