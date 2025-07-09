import os
from pathlib import Path
from typing import List

from setuptools import setup


def local_pkg(name: str, relative_path: str) -> str:
    """Returns an absolute path to a local package."""
    path = f"{name} @ file://{Path(os.path.abspath(__file__)).parent / relative_path}"
    return path


requirements: List[str] = [
    local_pkg("gt4py", "external/gt4py"),
    local_pkg("dace", "external/dace"),
    "mpi4py==3.1.5",
    "cftime",
    "xarray>=2025.01.2",  # datatree + fixes
    "f90nml>=1.1.0",
    "fsspec",
    "netcdf4==1.7.1",
    "scipy",  # restart capacities only
    "h5netcdf",  # for xarray
    "dask",  # for xarray
    "numpy==1.26.4",
    "matplotlib",  # for plotting in boilerplate
    "cartopy",  # for plotting in ndsl.viz
]


setup(
    install_requires=requirements,
<<<<<<< HEAD
=======
    extras_require=extras_requires,
    name="ndsl",
    license="Apache 2.0 license",
    packages=find_namespace_packages(include=["ndsl", "ndsl.*"]),
    include_package_data=True,
    url="https://github.com/NOAA-GFDL/NDSL",
    version="2025.05.00",
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "ndsl-serialbox_to_netcdf = ndsl.stencils.testing.serialbox_to_netcdf:entry_point",
        ]
    },
>>>>>>> develop
)
