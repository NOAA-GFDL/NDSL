# flake8: noqa
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import tests.grid.generate_eta_files
from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    NullComm,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from ndsl.grid import MetricTerms


"""
This test checks to ensure that ak and bk values are read-in and stored properly. In
addition, this test checks to ensure that the function set_hybrid_pressure_coefficients
fails as expected if the computed eta values vary non-monotonically and if the eta_file
is not provided.
"""


def set_answers(eta_file):
    """
    Read in the expected values of ak and bk arrays from the input eta NetCDF files.
    """

    data = xr.open_dataset(eta_file)
    return data["ak"].values, data["bk"].values


def write_non_mono_eta_file(in_eta_file, out_eta_file):
    """
    Reads in file eta79.nc and alters randomly chosen ak/bk values.

    This tests the expected failure of set_eta_hybrid_coefficients for coefficients
    that lead to non-monotonically increasing eta values.
    """

    data = xr.open_dataset(in_eta_file)
    data["ak"].values[10] = data["ak"].values[0]
    data["bk"].values[20] = 0.0

    data.to_netcdf(out_eta_file)


@pytest.mark.parametrize("km", [79, 91])
def test_set_hybrid_pressure_coefficients_correct(km, eta_file_path):
    """
    This test checks to see that the ak and bk arrays are read-in correctly and are
    stored as expected. Both values of km=79 and km=91 are tested and both tests are
    expected to pass with the stored ak and bk values agreeing with the values read-in
    directly from the NetCDF file.
    """

    eta_file = eta_file_path / f"eta{km}.nc"

    backend = "numpy"

    layout = (1, 1)

    nz = km
    ny = 48
    nx = 48
    nhalo = 3

    partitioner = CubedSpherePartitioner(TilePartitioner(layout))

    communicator = CubedSphereCommunicator(NullComm(rank=0, total_ranks=6), partitioner)

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=nz,
        n_halo=nhalo,
        extra_dim_lengths={},
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )

    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)

    metric_terms = MetricTerms(
        quantity_factory=quantity_factory, communicator=communicator, eta_file=eta_file
    )

    ak_results = metric_terms.ak.data
    bk_results = metric_terms.bk.data
    # ak_answers, bk_answers = set_answers(f"eta{km}.nc")
    ak_answers, bk_answers = set_answers(eta_file)

    assert ak_answers.size == ak_results.size, "Unexpected size of bk"
    assert bk_answers.size == bk_results.size, "Unexpected size of ak"

    assert np.array_equal(ak_answers, ak_results), "Unexpected value of ak"
    assert np.array_equal(bk_answers, bk_results), "Unexpected value of bk"


def test_set_hybrid_pressure_coefficients_nofile():
    """
    This test checks to see that the program fails when the eta_file is not specified
    in the yaml configuration file.
    """

    eta_file = Path("NULL")

    backend = "numpy"

    layout = (1, 1)

    nz = 79
    ny = 48
    nx = 48
    nhalo = 3

    partitioner = CubedSpherePartitioner(TilePartitioner(layout))

    communicator = CubedSphereCommunicator(NullComm(rank=0, total_ranks=6), partitioner)

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=nz,
        n_halo=nhalo,
        extra_dim_lengths={},
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )

    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)

    with pytest.raises(ValueError, match=f"eta file {eta_file} does not exist"):
        MetricTerms(
            quantity_factory=quantity_factory,
            communicator=communicator,
            eta_file=eta_file,
        )


def test_set_hybrid_pressure_coefficients_not_mono(eta_file_path):
    """
    This test checks to see that the program fails when the computed eta values
    increase non-monotonically. For the latter test, the eta_file is specified in
    test_config_not_mono.yaml file and the ak and bk values in the eta_file have been
    changed nonsensically to result in erroneous eta values.
    """

    in_eta_file = eta_file_path / "eta79.nc"
    out_eta_file = Path.cwd() / "eta_not_mono_79.nc"
    write_non_mono_eta_file(in_eta_file, out_eta_file)
    eta_file = out_eta_file

    backend = "numpy"

    layout = (1, 1)

    nz = 79
    ny = 48
    nx = 48
    nhalo = 3

    partitioner = CubedSpherePartitioner(TilePartitioner(layout))

    communicator = CubedSphereCommunicator(NullComm(rank=0, total_ranks=6), partitioner)

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=nz,
        n_halo=nhalo,
        extra_dim_lengths={},
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )

    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)

    with pytest.raises(ValueError, match="ETA values are not monotonically increasing"):
        MetricTerms(
            quantity_factory=quantity_factory,
            communicator=communicator,
            eta_file=eta_file,
        )

    # cleanup
    Path.unlink(out_eta_file, missing_ok=True)


# @pytest.mark.parametrize("km", [79, 91])
# def test_remove_eta_files(km):
#     eta_file = Path.cwd() / f"eta{km}.nc"
#     Path.unlink(eta_file, missing_ok=True)
