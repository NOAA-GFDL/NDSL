from ndsl import GridIndexing, StencilConfig


def make_storage(
    func,
    grid_indexing: GridIndexing,
    stencil_config: StencilConfig,
    *,
    dtype=float,
    aligned_index=(0, 0, 0),
):
    return func(
        backend=stencil_config.compilation_config.backend.as_gt4py(),
        shape=grid_indexing.domain,
        dtype=dtype,
        aligned_index=aligned_index,
    )
