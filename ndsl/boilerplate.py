from typing import Tuple

import matplotlib.pyplot as plt
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
    quantity_factory = QuantityFactory(sizer, np)

    return stencil_factory, quantity_factory


def get_factories_single_tile_orchestrated_cpu(
    nx, ny, nz, nhalo
) -> Tuple[StencilFactory, QuantityFactory]:
    """Build a Stencil & Quantity factory for orchestrated CPU, on a single tile topology."""
    return _get_factories(
        nx=nx,
        ny=ny,
        nz=nz,
        nhalo=nhalo,
        backend="dace:cpu",
        orchestration=DaCeOrchestration.BuildAndRun,
        topology="tile",
    )


def get_factories_single_tile_numpy(
    nx, ny, nz, nhalo
) -> Tuple[StencilFactory, QuantityFactory]:
    """Build a Stencil & Quantity factory for Numpy, on a single tile topology."""
    return _get_factories(
        nx=nx,
        ny=ny,
        nz=nz,
        nhalo=nhalo,
        backend="numpy",
        orchestration=DaCeOrchestration.Python,
        topology="tile",
    )


def plot_field_at_kN(field, k_index=0):

    print("Min and max values:", field[:, :, k_index].min(), field[:, :, k_index].max())
    plt.xlabel("I")
    plt.ylabel("J")

    im = plt.imshow(field[:, :, k_index].transpose(), origin="lower")

    plt.colorbar(im)
    plt.title("Plot at K = " + str(k_index))
    plt.show()
