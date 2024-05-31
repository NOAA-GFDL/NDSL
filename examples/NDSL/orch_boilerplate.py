import numpy as np
from ndsl import (
    StencilFactory,
    DaceConfig,
    DaCeOrchestration,
    GridIndexing,
    StencilConfig,
    CompilationConfig,
    RunMode,
    SubtileGridSizer,
    NullComm,
    QuantityFactory,
    TileCommunicator,
    TilePartitioner,
)

from typing import Tuple


def get_one_tile_factory_orchestrated(
    nx, ny, nz, nhalo, backend
) -> Tuple[StencilFactory, QuantityFactory]:
    """Create a 1 tile grid - no boundaries"""
    dace_config = DaceConfig(
        communicator=None,
        backend=backend,
        orchestration=DaCeOrchestration.BuildAndRun,
    )

    compilation_config = CompilationConfig(
        backend=backend,
        rebuild=True,
        validate_args=True,
        format_source=False,
        device_sync=False,
        run_mode=RunMode.BuildAndRun,
        use_minimal_caching=False,
    )

    stencil_config = StencilConfig(
        compare_to_numpy=False,
        compilation_config=compilation_config,
        dace_config=dace_config,
    )

    partitioner = TilePartitioner((1, 1))
    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=nz,
        n_halo=nhalo,
        extra_dim_lengths={},
        layout=partitioner.layout,
        tile_partitioner=partitioner,
    )

    tile_comm = TileCommunicator(comm=NullComm(0, 1, 42), partitioner=partitioner)

    grid_indexing = GridIndexing.from_sizer_and_communicator(sizer, tile_comm)
    stencil_factory = StencilFactory(config=stencil_config, grid_indexing=grid_indexing)
    quantity_factory = QuantityFactory(sizer, np)

    return stencil_factory, quantity_factory
