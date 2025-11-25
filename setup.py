from pathlib import Path

from setuptools import setup


def local_pkg(name: str, relative_path: str) -> str:
    """Returns an absolute path to a local package."""
    return f"{name} @ file://{Path(__file__).absolute().parent / relative_path}"


requirements: list[str] = [
    local_pkg("gt4py", "external/gt4py"),
    local_pkg("dace", "external/dace"),
    "mpi4py>=4.1",
    "cftime",
    "xarray>=2025.01.2",  # datatree + fixes
    "f90nml>=1.1.0",
    "netcdf4==1.7.2",
    "scipy",  # restart capacities only
    "h5netcdf",  # for xarray
    "dask",  # for xarray
    "numpy==1.26.4",
    "matplotlib",  # for plotting in boilerplate
    "cartopy",  # for plotting in ndsl.viz
    "pytest-subtests",  # for translate tests
    "dacite",  # for state
    "networkx<=3.5",
]


setup(install_requires=requirements)
