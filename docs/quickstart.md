# Quickstart

Alright - let's get you up an running!

NDSL requires Python version `3.11` and a GNU compiler. We strongly recommend using a conda or virtual environment.

```shell
# We have submodules for GT4Py and DaCe. Don't forget to pull them
git clone --recurse-submodules git@github.com:NOAA-GFDL/NDSL.git

cd NDSL/

# We strongly recommend using conda or a virtual environment
python -m venv .venv/
source ./venv/bin/activate

# [optional] Install MPI if you don't have a system installation.
pip install openmpi

# Finally, install NDSL
pip install .[demos]
```

Now you can run through the Jupyter notebooks in `examples/NDSL` :rocket:.

Read on in the [user manual](./user/index.md).

!!! note "Supported compilers"

    NDSL currently only works with the GNU compiler. Using `clang` will result in errors related to undefined OpenMP flags.

    For MacOS users, we know that `gcc` version 14 from homebrew works.

!!! question "Why cloning the repository?"

    We are cloning the repository because NDSL is not available on `pypi`.
