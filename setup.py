from pathlib import Path

from setuptools import setup


def local_pkg(name: str, relative_path: str) -> str:
    """Returns an absolute path to a local package."""
    return f"{name} @ file://{Path(__file__).absolute().parent / relative_path}"


docs_requirements = ["mkdocs-material", "mkdocstrings[python]"]
demos_requirements = ["ipython", "ipykernel"]
test_requirements = ["pytest", "pytest-subtests", "coverage"]

develop_requirements = test_requirements + docs_requirements + ["pre-commit"]

extras_requires = {
    "demos": demos_requirements,
    "develop": develop_requirements,
    "docs": docs_requirements,
    "test": test_requirements,
}

requirements = [
    local_pkg("gt4py", "external/gt4py"),
    local_pkg("dace", "external/dace"),
    "mpi4py>=4.1",
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
    "pytest-subtests",  # for translate tests
]


setup(install_requires=requirements)
