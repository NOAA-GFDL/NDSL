from pathlib import Path

import pytest

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    LocalComm,
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

    communicator = CubedSphereCommunicator(
        LocalComm(rank=0, total_ranks=6, buffer_dict={}), partitioner
    )

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

    communicator = CubedSphereCommunicator(
        LocalComm(rank=0, total_ranks=6, buffer_dict={}), partitioner
    )

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

    with pytest.raises(ValueError, match="ETA values are not monotonically increasing"):
        MetricTerms(
            quantity_factory=quantity_factory,
            communicator=communicator,
            eta_file=eta_file,
        )
