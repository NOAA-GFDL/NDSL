import os
from pathlib import Path
from typing import List

from setuptools import find_namespace_packages, setup


def local_pkg(name: str, relative_path: str) -> str:
    """Returns an absolute path to a local package."""
    path = f"{name} @ file://{Path(os.path.abspath(__file__)).parent / relative_path}"
    return path


test_requirements = ["pytest", "pytest-subtests", "coverage"]
develop_requirements = test_requirements + ["pre-commit"]

extras_requires = {"test": test_requirements, "develop": develop_requirements}

requirements: List[str] = [
    local_pkg("gt4py", "external/gt4py"),
    local_pkg("dace", "external/dace"),
    "mpi4py==3.1.5",
    "cftime",
    "xarray",
    "f90nml>=1.1.0",
    "fsspec",
    "netcdf4",
    "scipy",  # restart capacities only
    "h5netcdf",  # for xarray
    "dask",  # for xarray
]


setup(
    author="NOAA/NASA",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=requirements,
    extras_require=extras_requires,
    name="ndsl",
    license="BSD license",
    packages=find_namespace_packages(include=["ndsl", "ndsl.*"]),
    include_package_data=True,
    url="https://github.com/NOAA-GFDL/NDSL",
    version="2024.06.00",
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "ndsl-serialbox_to_netcdf = ndsl.stencils.testing.serialbox_to_netcdf:entry_point",
        ]
    },
)
