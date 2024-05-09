from ndsl.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from ndsl.dsl.stencil import GridIndexing, StencilConfig, StencilFactory
from ndsl.dsl.stencil_config import CompilationConfig, RunMode
import matplotlib.pyplot as plt


def get_one_tile_factory(nx, ny, nz, nhalo, backend) -> StencilFactory:

    dace_config = DaceConfig(
        communicator=None, backend=backend, orchestration=DaCeOrchestration.Python
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

    grid_indexing = GridIndexing(
        domain=(nx, ny, nz),
        n_halo=nhalo,
        south_edge=True,
        north_edge=True,
        west_edge=True,
        east_edge=True,
    )

    return StencilFactory(config=stencil_config, grid_indexing=grid_indexing)


def plot_field_at_k0(field):

    print("Min and max values:", field.max(), field.min())

    fig = plt.figure()
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.set_facecolor(".4")

    f1 = ax.pcolormesh(field[:, :, 0])

    cbar = plt.colorbar(f1)
    plt.show()

def plot_field_at_kN(field, k_index=0):

    print("Min and max values:", field[:,:,k_index].max(), field[:,:,k_index].min())
    plt.xlabel("I")
    plt.ylabel("J")

    im = plt.imshow(field[:,:,k_index].transpose(), origin='lower')

    plt.colorbar(im)
    plt.title("Plot at K = " + str(k_index))
    plt.show()
 