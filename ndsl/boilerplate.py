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


def _get_one_tile_factory(
    nx, ny, nz, nhalo, backend, orchestration
)-> Tuple[StencilFactory, QuantityFactory]:
    """Build a Stencil & Quantity factory for:
        - one tile
        - no MPI communicator
    """
    dace_config = DaceConfig(
        communicator=None,
        backend=backend,
        orchestration=orchestration,
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


def get_one_tile_factory_orchestrated_cpu(
    nx, ny, nz, nhalo
) -> Tuple[StencilFactory, QuantityFactory]:
    """Build a Stencil & Quantity factory for orchestrated CPU"""
    return _get_one_tile_factory(
        nx=nx,
        ny=ny,
        nz=nz,
        nhalo=nhalo,
        backend="dace:cpu",
        orchestration=DaCeOrchestration.BuildAndRun
    )

def get_one_tile_factory_numpy(
    nx, ny, nz, nhalo
) -> Tuple[StencilFactory, QuantityFactory]:
    """Build a Stencil & Quantity factory for Numpy"""
    return _get_one_tile_factory(
        nx=nx,
        ny=ny,
        nz=nz,
        nhalo=nhalo,
        backend="numpy",
        orchestration=DaCeOrchestration.Python
    )
