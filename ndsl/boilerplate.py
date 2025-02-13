from typing import Tuple

import numpy as np

from ndsl import (
    CompilationConfig,
    DaceConfig,
    DaCeOrchestration,
    GridIndexing,
    NullComm,
    QuantityFactory,
    RunMode,
    StencilConfig,
    StencilFactory,
    SubtileGridSizer,
    TileCommunicator,
    TilePartitioner,
)
from ndsl.optional_imports import cupy as cp


def _get_factories(
    nx: int,
    ny: int,
    nz: int,
    nhalo,
    backend: str,
    orchestration: DaCeOrchestration,
    topology: str,
) -> Tuple[StencilFactory, QuantityFactory]:
    """Build a Stencil & Quantity factory for a combination of options.

    Dev Note: We don't expose this function because we want the boilerplate to remain
    as easy and self describing as possible. It should be a very easy call to make.
    The other reason is that the orchestration requires two inputs instead of change
    a backend name for now, making it confusing. Until refactor, we choose to hide this
    pattern for boilerplate use.
    """
    dace_config = DaceConfig(
        communicator=None,
        backend=backend,
        orchestration=orchestration,
    )

    compilation_config = CompilationConfig(
        backend=backend,
        rebuild=False,
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

    if topology == "tile":
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
        comm = TileCommunicator(comm=NullComm(0, 1, 42), partitioner=partitioner)
    else:
        raise NotImplementedError(f"Topology {topology} is not implemented.")

    grid_indexing = GridIndexing.from_sizer_and_communicator(sizer, comm)
    stencil_factory = StencilFactory(config=stencil_config, grid_indexing=grid_indexing)
    quantity_factory = QuantityFactory(
        sizer, cp if stencil_config.is_gpu_backend else np
    )

    return stencil_factory, quantity_factory


def get_factories_single_tile_orchestrated(
    nx, ny, nz, nhalo, on_cpu: bool = True
) -> Tuple[StencilFactory, QuantityFactory]:
    """Build a Stencil & Quantity factory for orchestrated CPU, on a single tile topology."""
    return _get_factories(
        nx=nx,
        ny=ny,
        nz=nz,
        nhalo=nhalo,
        backend="dace:cpu" if on_cpu else "dace:gpu",
        orchestration=DaCeOrchestration.BuildAndRun,
        topology="tile",
    )


def get_factories_single_tile(
    nx, ny, nz, nhalo, backend: str = "numpy"
) -> Tuple[StencilFactory, QuantityFactory]:
    return _get_factories(
        nx=nx,
        ny=ny,
        nz=nz,
        nhalo=nhalo,
        backend=backend,
        orchestration=DaCeOrchestration.Python,
        topology="tile",
    )
