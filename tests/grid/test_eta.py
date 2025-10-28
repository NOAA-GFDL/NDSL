from pathlib import Path

import numpy as np
import pytest
import xarray as xr

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


@pytest.mark.parametrize("levels", [79, 91])
def test_set_hybrid_pressure_coefficients_correct(levels):
    """
    This test checks to see that the ak and bk arrays are read-in correctly and are
    stored as expected. Both values of km=79 and km=91 are tested and both tests are
    expected to pass with the stored ak and bk values agreeing with the values read-in
    directly from the NetCDF file.
    """

    eta_file = Path.cwd() / "tests" / "data" / "eta" / f"eta{levels}.nc"
    eta_data = xr.open_dataset(eta_file)

    backend = "numpy"

    layout = (1, 1)

    nz = levels
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
    ak_answers, bk_answers = eta_data["ak"].values, eta_data["bk"].values

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


def test_set_hybrid_pressure_coefficients_not_mono():
    """
    This test checks to see that the program fails when the computed eta values
    increase non-monotonically. For the latter test, the eta_file is specified in
    test_config_not_mono.yaml file and the ak and bk values in the eta_file have been
    changed nonsensically to result in erroneous eta values.
    """

    eta_file = str(Path.cwd()) + "/tests/data/eta/non_mono_eta79.nc"

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
