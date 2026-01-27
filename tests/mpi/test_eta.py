from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    MPIComm,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from ndsl.config import Backend
from ndsl.grid import MetricTerms
from tests.mpi import MPI


@pytest.mark.skipif(MPI is None, reason="pytest is not run in parallel")
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

    backend = Backend.python()

    layout = (1, 1)

    nz = levels
    ny = 48
    nx = 48
    nhalo = 3

    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    communicator = CubedSphereCommunicator(MPIComm(), partitioner)

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=nz,
        n_halo=nhalo,
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
        backend=backend,
    )

    quantity_factory = QuantityFactory(sizer, backend=backend)

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
