from typing import Callable

import numpy.typing as npt

from ndsl import GridIndexing, StencilConfig


def make_storage(
    func: Callable,
    grid_indexing: GridIndexing,
    stencil_config: StencilConfig,
    *,
    dtype: type = float,
    aligned_index: tuple = (0, 0, 0),
) -> npt.NDArray:
    return func(
        backend=stencil_config.compilation_config.backend.as_gt4py(),
        shape=grid_indexing.domain,
        dtype=dtype,
        aligned_index=aligned_index,
    )
