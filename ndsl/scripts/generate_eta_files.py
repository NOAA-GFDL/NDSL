from pathlib import Path

import click

from ndsl.data.eta import eta_79_km, eta_91_km


"""
This script uses the python xarray module to create an eta_file containing ak and bk coefficients for km=79 and km=91.
The coefficients are written out to `eta79.nc` and `eta91.nc` netcdf files respectively.
"""


def generate_eta_79(path: Path):
    eta_79_km.to_netcdf(path=path / "eta79.nc")


def generate_eta_91(path: Path):
    eta_91_km.to_netcdf(path=path / "eta91.nc")


@click.command()
@click.option("--path", default=".", help="Location for eta files")
def generator(path):
    file_path = Path(path)
    generate_eta_79(path=file_path)
    generate_eta_91(path=file_path)


if __name__ == "__main__":
    generator()
