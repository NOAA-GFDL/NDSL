"""End-to-end nested-grid communication example for NDSL.

This example demonstrates the final split of responsibilities:

1. fine-grid nested ranks perform ordinary same-resolution halo updates;
2. the parent and each nested region exchange a same-resolution coarse
   transport Quantity;
3. this example owns the numerical coarse-to-fine prolongation and optional
   fine-to-coarse reduction.

Adding or moving a nest only requires changing NESTS below. Each nest can
choose its parent tile, offset, extent, refinement ratio, and nested rank layout
independently. Nested world ranks are assigned contiguously after the parent
ranks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

import ndsl.constants as constants
from ndsl import (
    Backend,
    CubedSpherePartitioner,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from ndsl.comm.communicator import (
    CubedSphereCommunicator,
    NestedCommunicator,
    NestTileCommunicator,
)
from ndsl.comm.mpi import MPIComm
from ndsl.comm.partitioner import NestedPartitioner, NestMapping
from ndsl.constants import (
    I_DIM,
    I_INTERFACE_DIM,
    J_DIM,
    J_INTERFACE_DIM,
    N_HALO_DEFAULT,
)
from ndsl.grid import MetricTerms
from ndsl.quantity import Quantity

# -----------------------------------------------------------------------------
# Parent configuration
# -----------------------------------------------------------------------------

PARENT_NX = 12
PARENT_NY = 12
PARENT_LAYOUT = (2, 2)
NZ = 1

# Fine-grid halo used by the actual nested Quantity.
#
# The coarse transport Quantity uses a derived halo width:
# ceil(FINE_HALO / refinement_ratio).
#
# For the current 4x4 parent patch split over a 2x2 nested layout:
#     fine local compute   = 4 x 4
#     coarse local compute = 2 x 2
#
# FINE_HALO = 2 therefore gives:
#     fine halo   = 2
#     coarse halo = 1 for refinement ratio 2
STORAGE_HALO = N_HALO_DEFAULT
FINE_EXCHANGE_HALO = 2


# -----------------------------------------------------------------------------
# Nested-region configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NestConfig:
    """Configuration for one nested region."""

    parent_region: int
    parent_start: tuple[int, int]
    parent_extent: tuple[int, int]
    refinement_ratio: int = 2
    layout: tuple[int, int] = (2, 2)

    coarse_halo: int = 1

    # Optional explicit parent communicator rank used as the mapping anchor.
    # If None, the first rank of parent_region is used.
    parent_rank: int | None = None


NESTS: dict[int, NestConfig] = {
    0: NestConfig(
        parent_region=0,
        parent_start=(3, 3),
        parent_extent=(4, 4),
        refinement_ratio=2,
        layout=(2, 2),
        coarse_halo=1,
    ),
    1: NestConfig(
        parent_region=1,
        parent_start=(3, 3),
        parent_extent=(4, 4),
        refinement_ratio=2,
        layout=(2, 2),
        coarse_halo=1,
    ),
}


STAGGERINGS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("A-grid", (I_DIM, J_DIM)),
    ("I-interface", (I_INTERFACE_DIM, J_DIM)),
    ("J-interface", (I_DIM, J_INTERFACE_DIM)),
)


MetricPayload = dict[str, Any]
ReferenceMetricData = dict[str, MetricPayload]
HorizontalDims = tuple[str, str]


# -----------------------------------------------------------------------------
# Nested physical-grid containers
# -----------------------------------------------------------------------------


@dataclass
class NestedPhysicalGrid:
    """Physical grid data local to one nested rank."""

    agrid_lon_lat: np.ndarray
    dgrid_lon_lat: np.ndarray
    area: Quantity
    dx: Quantity
    dy: Quantity
    dxa: Quantity
    dya: Quantity
    dxc: Quantity
    dyc: Quantity


@dataclass
class NestedGridContext:
    """Fine-grid and coarse-transport factories for one nested rank."""

    fine_factory: QuantityFactory
    coarse_factory: QuantityFactory
    physical_grid: NestedPhysicalGrid


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def horizontal_global_extent(
    dims: HorizontalDims,
    nx: int,
    ny: int,
) -> tuple[int, int]:
    """Return the global horizontal extent implied by NDSL staggering."""

    extent = tuple(
        size + (1 if dim in constants.INTERFACE_DIMS else 0)
        for dim, size in zip(dims, (nx, ny))
    )
    return extent[0], extent[1]


def parent_anchor_rank(
    config: NestConfig,
    parent_tile_ranks: int,
) -> int:
    """Return the parent-communicator rank used as a nest anchor."""

    if config.parent_rank is not None:
        return config.parent_rank

    return config.parent_region * parent_tile_ranks


def make_nested_partitioners(
    parent_partitioner: CubedSpherePartitioner,
) -> dict[int, NestedPartitioner]:
    """Construct all nested partitioners from NESTS."""

    parent_tile_ranks = parent_partitioner.tile.total_ranks

    return {
        nest_id: NestedPartitioner(
            layout=config.layout,
            mapping=NestMapping(
                parent_rank=parent_anchor_rank(
                    config,
                    parent_tile_ranks,
                ),
                parent_region=config.parent_region,
                parent_start=config.parent_start,
                parent_extent=config.parent_extent,
                refinement_ratio=config.refinement_ratio,
            ),
        )
        for nest_id, config in NESTS.items()
    }


def assign_nested_world_ranks(
    parent_size: int,
    nested_partitioners: dict[int, NestedPartitioner],
) -> dict[int, tuple[int, ...]]:
    """Assign nested world ranks contiguously after the parent ranks."""

    next_rank = parent_size
    nested_world_ranks: dict[int, tuple[int, ...]] = {}

    for nest_id, partitioner in nested_partitioners.items():
        stop_rank = next_rank + partitioner.total_ranks
        nested_world_ranks[nest_id] = tuple(range(next_rank, stop_rank))
        next_rank = stop_rank

    return nested_world_ranks


def find_local_nest_id(
    world_rank: int,
    nested_world_ranks: dict[int, tuple[int, ...]],
) -> int | None:
    """Return the nest containing world_rank, if any."""

    matches = [
        nest_id for nest_id, ranks in nested_world_ranks.items() if world_rank in ranks
    ]

    if len(matches) > 1:
        raise RuntimeError(f"world rank {world_rank} belongs to multiple nests")

    return matches[0] if matches else None


# -----------------------------------------------------------------------------
# Parent/fine reference-grid construction
# -----------------------------------------------------------------------------


def make_parent_metric_terms(
    communicator: NestedCommunicator,
    backend: Backend,
    nx_tile: int,
    ny_tile: int,
) -> MetricTerms:
    """Construct ordinary cubed-sphere MetricTerms on parent ranks."""

    assert communicator.is_parent_rank
    assert communicator.parent_comm is not None

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx_tile,
        ny_tile=ny_tile,
        nz=NZ,
        n_halo=STORAGE_HALO,
        layout=PARENT_LAYOUT,
        tile_partitioner=communicator.parent_partitioner.tile,
        tile_rank=communicator.parent_comm.tile.rank,
        backend=backend,
    )

    quantity_factory = QuantityFactory(
        sizer=sizer,
        backend=backend,
    )

    return MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator.parent_comm,
        grid_type=0,
    )


def materialize_metric_terms(
    metrics: MetricTerms,
    names: tuple[str, ...],
) -> None:
    """Force lazy MetricTerms fields to be generated collectively."""

    for name in names:
        _ = getattr(metrics, name).view[:]


def check_physical_refinement(
    communicator: NestedCommunicator,
    coarse_metrics: MetricTerms | None,
    fine_metrics: MetricTerms | None,
    nest_id: int,
) -> None:
    """Verify that a high-resolution reference grid refines one nest region."""

    if not communicator.is_parent_rank:
        return

    assert communicator.parent_comm is not None
    assert coarse_metrics is not None
    assert fine_metrics is not None

    partitioner = communicator.nested_partitioners[nest_id]
    mapping = partitioner.mapping

    tile_comm = communicator.parent_comm.tile

    coarse_dgrid_quantity = tile_comm.gather(coarse_metrics.dgrid_lon_lat)
    fine_dgrid_quantity = tile_comm.gather(fine_metrics.dgrid_lon_lat)
    coarse_area_quantity = tile_comm.gather(coarse_metrics.area)
    fine_area_quantity = tile_comm.gather(fine_metrics.area)
    coarse_dx_quantity = tile_comm.gather(coarse_metrics.dx)
    coarse_dy_quantity = tile_comm.gather(coarse_metrics.dy)
    fine_dx_quantity = tile_comm.gather(fine_metrics.dx)
    fine_dy_quantity = tile_comm.gather(fine_metrics.dy)

    parent_partitioner = communicator.parent_partitioner

    if not isinstance(
        parent_partitioner,
        CubedSpherePartitioner,
    ):
        raise TypeError(
            "nested-grid example requires a " "CubedSpherePartitioner parent"
        )

    parent_root_rank = parent_partitioner.tile_root_rank(mapping.parent_rank)

    if communicator.world_rank != parent_root_rank:
        return

    assert coarse_dgrid_quantity is not None
    assert fine_dgrid_quantity is not None
    assert coarse_area_quantity is not None
    assert fine_area_quantity is not None
    assert coarse_dx_quantity is not None
    assert coarse_dy_quantity is not None
    assert fine_dx_quantity is not None
    assert fine_dy_quantity is not None

    coarse_dgrid = np.asarray(coarse_dgrid_quantity.view[:])
    fine_dgrid = np.asarray(fine_dgrid_quantity.view[:])
    coarse_area = np.asarray(coarse_area_quantity.view[:])
    fine_area = np.asarray(fine_area_quantity.view[:])
    coarse_dx = np.asarray(coarse_dx_quantity.view[:])
    coarse_dy = np.asarray(coarse_dy_quantity.view[:])
    fine_dx = np.asarray(fine_dx_quantity.view[:])
    fine_dy = np.asarray(fine_dy_quantity.view[:])

    ci0, cj0 = mapping.parent_start
    ci1 = ci0 + mapping.parent_extent[0]
    cj1 = cj0 + mapping.parent_extent[1]

    refinement = mapping.refinement_ratio

    fi0 = ci0 * refinement
    fj0 = cj0 * refinement
    fi1 = ci1 * refinement
    fj1 = cj1 * refinement

    coarse_corners = np.asarray(
        [
            coarse_dgrid[ci0, cj0, :],
            coarse_dgrid[ci1, cj0, :],
            coarse_dgrid[ci0, cj1, :],
            coarse_dgrid[ci1, cj1, :],
        ]
    )

    fine_corners = np.asarray(
        [
            fine_dgrid[fi0, fj0, :],
            fine_dgrid[fi1, fj0, :],
            fine_dgrid[fi0, fj1, :],
            fine_dgrid[fi1, fj1, :],
        ]
    )

    corner_error = float(np.max(np.abs(coarse_corners - fine_corners)))

    coarse_patch_area = float(
        np.sum(
            coarse_area[
                ci0:ci1,
                cj0:cj1,
            ]
        )
    )
    fine_patch_area = float(
        np.sum(
            fine_area[
                fi0:fi1,
                fj0:fj1,
            ]
        )
    )

    area_relative_error = abs(fine_patch_area - coarse_patch_area) / coarse_patch_area

    coarse_dx_patch = coarse_dx[
        ci0:ci1,
        cj0 : cj1 + 1,
    ]
    fine_dx_patch = fine_dx[
        fi0:fi1,
        fj0 : fj1 + 1,
    ]
    coarse_dy_patch = coarse_dy[
        ci0 : ci1 + 1,
        cj0:cj1,
    ]
    fine_dy_patch = fine_dy[
        fi0 : fi1 + 1,
        fj0:fj1,
    ]

    dx_ratio = float(np.mean(coarse_dx_patch) / np.mean(fine_dx_patch))
    dy_ratio = float(np.mean(coarse_dy_patch) / np.mean(fine_dy_patch))

    if corner_error > 1.0e-8:
        raise AssertionError(
            f"nest {nest_id}: maximum corner mismatch " f"is {corner_error:.6e} rad"
        )

    if area_relative_error > 1.0e-4:
        raise AssertionError(
            f"nest {nest_id}: relative area mismatch " f"is {area_relative_error:.6e}"
        )

    if not np.isclose(
        dx_ratio,
        refinement,
        rtol=0.10,
    ):
        raise AssertionError(
            f"nest {nest_id}: dx ratio {dx_ratio}, "
            f"expected approximately {refinement}"
        )

    if not np.isclose(
        dy_ratio,
        refinement,
        rtol=0.10,
    ):
        raise AssertionError(
            f"nest {nest_id}: dy ratio {dy_ratio}, "
            f"expected approximately {refinement}"
        )

    print(
        f"\nPhysical refinement check for nest {nest_id}:\n"
        f"  max corner mismatch    = "
        f"{corner_error:.6e} rad\n"
        f"  relative area mismatch = "
        f"{area_relative_error:.6e}\n"
        f"  coarse/fine dx ratio   = "
        f"{dx_ratio:.6f}\n"
        f"  coarse/fine dy ratio   = "
        f"{dy_ratio:.6f}\n"
        f"  expected ratio         = "
        f"{refinement}\n"
        "  physical refinement    = PASSED",
        flush=True,
    )


def reference_quantity_payload(
    quantity: Quantity,
) -> MetricPayload:
    """Copy one gathered MetricTerms field into a serializable payload."""

    return {
        "data": np.asarray(quantity.view[:]).copy(),
        "dims": tuple(quantity.dims),
        "units": quantity.units,
    }


def get_reference_metric_data(
    communicator: NestedCommunicator,
    fine_metrics: MetricTerms | None,
    nest_id: int,
) -> ReferenceMetricData | None:
    """Gather the high-resolution parent tile used as a fine-grid reference."""

    if not communicator.is_parent_rank:
        return None

    assert communicator.parent_comm is not None
    assert fine_metrics is not None

    partitioner = communicator.nested_partitioners[nest_id]
    mapping = partitioner.mapping

    tile_comm = communicator.parent_comm.tile

    names = (
        "agrid_lon_lat",
        "dgrid_lon_lat",
        "area",
        "dx",
        "dy",
        "dxa",
        "dya",
        "dxc",
        "dyc",
    )

    parent_partitioner = communicator.parent_partitioner

    if not isinstance(
        parent_partitioner,
        CubedSpherePartitioner,
    ):
        raise TypeError(
            "nested-grid example requires a " "CubedSpherePartitioner parent"
        )

    parent_root_rank = parent_partitioner.tile_root_rank(mapping.parent_rank)

    reference: ReferenceMetricData = {}

    for name in names:
        tile_quantity = tile_comm.gather(
            getattr(
                fine_metrics,
                name,
            )
        )

        if communicator.world_rank == parent_root_rank:
            assert tile_quantity is not None
            reference[name] = reference_quantity_payload(tile_quantity)

    if communicator.world_rank == parent_root_rank:
        return reference

    return None


# -----------------------------------------------------------------------------
# Nested-grid physical context
# -----------------------------------------------------------------------------


def nested_rank_global_slice(
    *,
    partitioner: NestedPartitioner,
    nested_rank: int,
    dims: HorizontalDims,
    nx: int,
    ny: int,
) -> tuple[slice, slice]:
    """Return one nested rank's compute slice in nested-global coordinates."""

    rank_slice = partitioner.subtile_slice(
        rank=nested_rank,
        global_dims=dims,
        global_extent=horizontal_global_extent(
            dims,
            nx,
            ny,
        ),
        overlap=True,
    )

    i_slice, j_slice = rank_slice

    if not isinstance(
        i_slice,
        slice,
    ) or not isinstance(
        j_slice,
        slice,
    ):
        raise TypeError(f"Expected horizontal slices, got " f"{rank_slice}")

    return i_slice, j_slice


def copy_reference_metric_to_nested_quantity(
    *,
    payload: MetricPayload,
    quantity_factory: QuantityFactory,
    partitioner: NestedPartitioner,
    nested_rank: int,
) -> Quantity:
    """Copy one fine-reference metric compute domain to a nested rank."""

    source = np.asarray(payload["data"])
    dims = tuple(payload["dims"])
    units = payload["units"]

    if len(dims) != 2:
        raise ValueError(f"Expected a 2-D metric field, got dims={dims}")

    horizontal_dims: HorizontalDims = (
        dims[0],
        dims[1],
    )

    quantity = quantity_factory.zeros(
        dims=horizontal_dims,
        units=units,
        dtype=float,
    )

    fine_nx, fine_ny = partitioner.mapping.fine_extent

    i_slice, j_slice = nested_rank_global_slice(
        partitioner=partitioner,
        nested_rank=nested_rank,
        dims=horizontal_dims,
        nx=fine_nx,
        ny=fine_ny,
    )

    if (
        i_slice.start is None
        or i_slice.stop is None
        or j_slice.start is None
        or j_slice.stop is None
    ):
        raise RuntimeError("Nested rank metric slice must be bounded")

    refinement = partitioner.mapping.refinement_ratio

    fine_parent_i0 = partitioner.mapping.parent_start[0] * refinement
    fine_parent_j0 = partitioner.mapping.parent_start[1] * refinement

    source_i_start = fine_parent_i0 + i_slice.start
    source_i_stop = fine_parent_i0 + i_slice.stop
    source_j_start = fine_parent_j0 + j_slice.start
    source_j_stop = fine_parent_j0 + j_slice.stop

    source_compute = source[
        source_i_start:source_i_stop,
        source_j_start:source_j_stop,
    ]

    if source_compute.shape != quantity.view[:].shape:
        raise RuntimeError(
            "Reference metric shape does not match nested compute domain: "
            f"source={source_compute.shape}, "
            f"nested={quantity.view[:].shape}, "
            f"dims={dims}, "
            f"nested_rank={nested_rank}"
        )

    quantity.view[:] = source_compute

    return quantity


def extract_reference_coordinates(
    *,
    payload: MetricPayload,
    partitioner: NestedPartitioner,
    nested_rank: int,
) -> np.ndarray:
    """Extract the fine-reference coordinates for one nested compute domain."""

    source = np.asarray(payload["data"])
    dims = tuple(payload["dims"])

    if len(dims) != 3:
        raise ValueError(
            "Expected coordinates with two horizontal dimensions plus "
            f"lon/lat, got dims={dims}"
        )

    horizontal_dims: HorizontalDims = (
        dims[0],
        dims[1],
    )

    fine_nx, fine_ny = partitioner.mapping.fine_extent

    i_slice, j_slice = nested_rank_global_slice(
        partitioner=partitioner,
        nested_rank=nested_rank,
        dims=horizontal_dims,
        nx=fine_nx,
        ny=fine_ny,
    )

    if (
        i_slice.start is None
        or i_slice.stop is None
        or j_slice.start is None
        or j_slice.stop is None
    ):
        raise RuntimeError("Nested coordinate slice must be bounded")

    refinement = partitioner.mapping.refinement_ratio

    fine_parent_i0 = partitioner.mapping.parent_start[0] * refinement
    fine_parent_j0 = partitioner.mapping.parent_start[1] * refinement

    source_i_start = fine_parent_i0 + i_slice.start
    source_i_stop = fine_parent_i0 + i_slice.stop
    source_j_start = fine_parent_j0 + j_slice.start
    source_j_stop = fine_parent_j0 + j_slice.stop

    return np.asarray(
        source[
            source_i_start:source_i_stop,
            source_j_start:source_j_stop,
            :,
        ]
    ).copy()


def make_nested_grid_context(
    communicator: NestedCommunicator,
    backend: Backend,
    reference: ReferenceMetricData,
    nest_id: int,
    coarse_n_points: int,  # CHANGED: caller supplies coarse halo width
) -> NestedGridContext:
    """Construct fine and coarse-transport contexts on one nested rank."""

    nested_rank = communicator.nested_rank(nest_id)
    assert nested_rank is not None

    partitioner = communicator.nested_partitioners[nest_id]
    mapping = partitioner.mapping

    # NEW: validate caller-selected coarse halo width.
    if coarse_n_points <= 0:
        raise ValueError("coarse_n_points must be positive")

    fine_nx, fine_ny = mapping.fine_extent
    coarse_nx, coarse_ny = mapping.parent_extent

    fine_sizer = SubtileGridSizer.from_tile_params(
        nx_tile=fine_nx,
        ny_tile=fine_ny,
        nz=NZ,
        n_halo=STORAGE_HALO,
        layout=partitioner.layout,
        tile_partitioner=partitioner.tile,
        tile_rank=nested_rank,
        backend=backend,
    )

    coarse_sizer = SubtileGridSizer.from_tile_params(
        nx_tile=coarse_nx,
        ny_tile=coarse_ny,
        nz=NZ,
        # CHANGED:
        # Independent coarse transport halo selected by caller.
        n_halo=coarse_n_points,
        layout=partitioner.layout,
        tile_partitioner=partitioner.tile,
        tile_rank=nested_rank,
        backend=backend,
    )

    fine_factory = QuantityFactory(
        sizer=fine_sizer,
        backend=backend,
    )
    coarse_factory = QuantityFactory(
        sizer=coarse_sizer,
        backend=backend,
    )

    physical_grid = NestedPhysicalGrid(
        agrid_lon_lat=extract_reference_coordinates(
            payload=reference["agrid_lon_lat"],
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dgrid_lon_lat=extract_reference_coordinates(
            payload=reference["dgrid_lon_lat"],
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        area=copy_reference_metric_to_nested_quantity(
            payload=reference["area"],
            quantity_factory=fine_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dx=copy_reference_metric_to_nested_quantity(
            payload=reference["dx"],
            quantity_factory=fine_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dy=copy_reference_metric_to_nested_quantity(
            payload=reference["dy"],
            quantity_factory=fine_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dxa=copy_reference_metric_to_nested_quantity(
            payload=reference["dxa"],
            quantity_factory=fine_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dya=copy_reference_metric_to_nested_quantity(
            payload=reference["dya"],
            quantity_factory=fine_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dxc=copy_reference_metric_to_nested_quantity(
            payload=reference["dxc"],
            quantity_factory=fine_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dyc=copy_reference_metric_to_nested_quantity(
            payload=reference["dyc"],
            quantity_factory=fine_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
    )

    return NestedGridContext(
        fine_factory=fine_factory,
        coarse_factory=coarse_factory,
        physical_grid=physical_grid,
    )


def check_nested_physical_grid(
    communicator: NestedCommunicator,
    context: NestedGridContext | None,
) -> None:
    """Check that nested metrics are finite and physically meaningful."""

    if not communicator.is_nested_rank:
        return

    assert context is not None
    grid = context.physical_grid

    for name in (
        "area",
        "dx",
        "dy",
    ):
        values = np.asarray(
            getattr(
                grid,
                name,
            ).view[:]
        )

        if not np.all(np.isfinite(values)):
            raise AssertionError(f"Nested {name} contains " "non-finite values")

        if not np.all(values > 0.0):
            raise AssertionError(f"Nested {name} contains " "non-positive values")

    if not np.all(np.isfinite(grid.agrid_lon_lat)):
        raise AssertionError("Nested A-grid coordinates " "contain non-finite values")

    if not np.all(np.isfinite(grid.dgrid_lon_lat)):
        raise AssertionError("Nested D-grid coordinates " "contain non-finite values")


def print_nested_grid_summary(
    communicator: NestedCommunicator,
    context: NestedGridContext | None,
    nest_id: int,
) -> None:
    """Print one compact physical-grid summary per nested rank."""

    nested_rank = communicator.nested_rank(nest_id)

    if nested_rank is None:
        return

    assert context is not None

    grid = context.physical_grid
    center_lon = grid.agrid_lon_lat[..., 0]
    center_lat = grid.agrid_lon_lat[..., 1]

    print(
        f"nest {nest_id}, nested rank "
        f"{nested_rank}: "
        f"mean area="
        f"{np.mean(np.asarray(grid.area.view[:])):.6e} m^2, "
        f"mean dx="
        f"{np.mean(np.asarray(grid.dx.view[:])):.6e} m, "
        f"mean dy="
        f"{np.mean(np.asarray(grid.dy.view[:])):.6e} m, "
        f"lon=["
        f"{np.degrees(np.min(center_lon)):.3f}, "
        f"{np.degrees(np.max(center_lon)):.3f}] deg, "
        f"lat=["
        f"{np.degrees(np.min(center_lat)):.3f}, "
        f"{np.degrees(np.max(center_lat)):.3f}] deg",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Test quantities
# -----------------------------------------------------------------------------


def make_parent_quantity(
    communicator: NestedCommunicator,
    backend: Backend,
    dims: HorizontalDims,
) -> Quantity:
    """Allocate one parent test Quantity."""

    assert communicator.is_parent_rank
    assert communicator.parent_comm is not None

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=PARENT_NX,
        ny_tile=PARENT_NY,
        nz=NZ,
        n_halo=STORAGE_HALO,
        layout=PARENT_LAYOUT,
        tile_partitioner=communicator.parent_partitioner.tile,
        tile_rank=communicator.parent_comm.tile.rank,
        backend=backend,
    )

    return QuantityFactory(
        sizer=sizer,
        backend=backend,
    ).zeros(
        dims=dims,
        units="1",
        dtype=float,
    )


def make_nested_fine_quantity(
    context: NestedGridContext,
    dims: HorizontalDims,
) -> Quantity:
    """Allocate one fine nested Quantity."""

    return context.fine_factory.zeros(
        dims=dims,
        units="1",
        dtype=float,
    )


def make_nested_coarse_quantity(
    context: NestedGridContext,
    dims: HorizontalDims,
) -> Quantity:
    """Allocate one coarse transport Quantity on a nested rank."""

    return context.coarse_factory.zeros(
        dims=dims,
        units="1",
        dtype=float,
    )


def initialize_parent_quantity(
    quantity: Quantity,
    communicator: NestedCommunicator,
    nest_id: int,
) -> None:
    """Fill the supplying parent tile with global-coordinate values."""

    assert communicator.is_parent_rank

    parent_rank = communicator.parent_rank
    assert parent_rank is not None

    parent_partitioner = communicator.parent_partitioner
    parent_tile = parent_partitioner.tile

    tile_index = parent_partitioner.tile_index(parent_rank)

    mapping = communicator.nested_partitioners[nest_id].mapping

    if tile_index != mapping.parent_region:
        quantity.view[:] = -100.0 - parent_rank
        return

    dims = tuple(quantity.dims)

    if len(dims) != 2:
        raise ValueError("Expected a 2-D horizontal " f"Quantity, got dims={dims}")

    tile_rank = parent_rank % parent_tile.total_ranks

    rank_slice = parent_tile.subtile_slice(
        rank=tile_rank,
        global_dims=dims,
        global_extent=horizontal_global_extent(
            (
                dims[0],
                dims[1],
            ),
            PARENT_NX,
            PARENT_NY,
        ),
        overlap=True,
    )

    i_slice, j_slice = rank_slice

    if (
        i_slice.start is None
        or i_slice.stop is None
        or j_slice.start is None
        or j_slice.stop is None
    ):
        raise RuntimeError("Parent rank slice must " "be bounded")

    for local_i, global_i in enumerate(
        range(
            i_slice.start,
            i_slice.stop,
        )
    ):
        for local_j, global_j in enumerate(
            range(
                j_slice.start,
                j_slice.stop,
            )
        ):
            quantity.view[
                local_i,
                local_j,
            ] = (
                1000.0 + 100.0 * global_i + global_j
            )


def initialize_nested_quantity(
    quantity: Quantity,
    nested_rank: int,
) -> None:
    """Fill the fine compute domain with a rank-identifying value."""

    quantity.view[:] = 10.0 * (nested_rank + 1)


# -----------------------------------------------------------------------------
# Numerical policy outside NDSL
# -----------------------------------------------------------------------------


def _coarse_index_from_fine(
    fine_index: int,
    dim: str,
    refinement_ratio: int,
) -> int:
    """Piecewise-constant fine->coarse index used by this demo."""

    offset = 0.0 if dim in constants.INTERFACE_DIMS else 0.5

    return math.floor((fine_index + offset) / refinement_ratio)


def prolong_coarse_to_fine_halo(
    *,
    coarse_quantity: Quantity,
    fine_quantity: Quantity,
    partitioner: NestedPartitioner,
    nested_rank: int,
    n_points: int,
) -> None:
    """Piecewise-constant prolongation into the external fine halo."""

    dims = tuple(fine_quantity.dims)

    if tuple(coarse_quantity.dims) != dims or len(dims) != 2:
        raise ValueError("coarse and fine quantities " "must have matching 2-D dims")

    mapping = partitioner.mapping
    refinement = mapping.refinement_ratio

    fine_nx, fine_ny = mapping.fine_extent
    coarse_nx, coarse_ny = mapping.parent_extent

    fine_slice = nested_rank_global_slice(
        partitioner=partitioner,
        nested_rank=nested_rank,
        dims=(
            dims[0],
            dims[1],
        ),
        nx=fine_nx,
        ny=fine_ny,
    )

    coarse_slice = nested_rank_global_slice(
        partitioner=partitioner,
        nested_rank=nested_rank,
        dims=(
            dims[0],
            dims[1],
        ),
        nx=coarse_nx,
        ny=coarse_ny,
    )

    if (
        fine_slice[0].start is None
        or fine_slice[1].start is None
        or coarse_slice[0].start is None
        or coarse_slice[1].start is None
    ):
        raise RuntimeError("nested subtile slices must " "have bounded starts")

    fine_global_extent = horizontal_global_extent(
        (
            dims[0],
            dims[1],
        ),
        fine_nx,
        fine_ny,
    )

    fine_origin_i, fine_origin_j = fine_quantity.origin
    fine_extent_i, fine_extent_j = fine_quantity.extent

    coarse_origin_i, coarse_origin_j = coarse_quantity.origin

    for data_i in range(
        fine_origin_i - n_points,
        fine_origin_i + fine_extent_i + n_points,
    ):
        fine_global_i = fine_slice[0].start + data_i - fine_origin_i

        for data_j in range(
            fine_origin_j - n_points,
            fine_origin_j + fine_extent_j + n_points,
        ):
            fine_global_j = fine_slice[1].start + data_j - fine_origin_j

            outside_nested_domain = (
                fine_global_i < 0
                or fine_global_i >= fine_global_extent[0]
                or fine_global_j < 0
                or fine_global_j >= fine_global_extent[1]
            )

            if not outside_nested_domain:
                continue

            coarse_global_i = _coarse_index_from_fine(
                fine_global_i,
                dims[0],
                refinement,
            )
            coarse_global_j = _coarse_index_from_fine(
                fine_global_j,
                dims[1],
                refinement,
            )

            coarse_data_i = coarse_origin_i + coarse_global_i - coarse_slice[0].start
            coarse_data_j = coarse_origin_j + coarse_global_j - coarse_slice[1].start

            fine_quantity[
                data_i,
                data_j,
            ] = coarse_quantity[
                coarse_data_i,
                coarse_data_j,
            ]


def reduce_fine_to_coarse(
    *,
    fine_quantity: Quantity,
    coarse_quantity: Quantity,
    partitioner: NestedPartitioner,
    nested_rank: int,
) -> None:
    """Simple bin-average reduction of the local fine compute domain."""

    dims = tuple(fine_quantity.dims)

    if tuple(coarse_quantity.dims) != dims or len(dims) != 2:
        raise ValueError("coarse and fine quantities " "must have matching 2-D dims")

    mapping = partitioner.mapping
    refinement = mapping.refinement_ratio

    fine_nx, fine_ny = mapping.fine_extent
    coarse_nx, coarse_ny = mapping.parent_extent

    fine_slice = nested_rank_global_slice(
        partitioner=partitioner,
        nested_rank=nested_rank,
        dims=(
            dims[0],
            dims[1],
        ),
        nx=fine_nx,
        ny=fine_ny,
    )

    coarse_slice = nested_rank_global_slice(
        partitioner=partitioner,
        nested_rank=nested_rank,
        dims=(
            dims[0],
            dims[1],
        ),
        nx=coarse_nx,
        ny=coarse_ny,
    )

    if (
        fine_slice[0].start is None
        or fine_slice[1].start is None
        or coarse_slice[0].start is None
        or coarse_slice[1].start is None
    ):
        raise RuntimeError("nested subtile slices must " "have bounded starts")

    accum = np.zeros(
        coarse_quantity.extent,
        dtype=float,
    )
    counts = np.zeros(
        coarse_quantity.extent,
        dtype=np.int64,
    )

    fine_values = np.asarray(fine_quantity.view[:])

    for local_i in range(fine_values.shape[0]):
        fine_global_i = fine_slice[0].start + local_i

        coarse_global_i = _coarse_index_from_fine(
            fine_global_i,
            dims[0],
            refinement,
        )

        coarse_local_i = coarse_global_i - coarse_slice[0].start

        if coarse_local_i < 0 or coarse_local_i >= coarse_quantity.extent[0]:
            continue

        for local_j in range(fine_values.shape[1]):
            fine_global_j = fine_slice[1].start + local_j

            coarse_global_j = _coarse_index_from_fine(
                fine_global_j,
                dims[1],
                refinement,
            )

            coarse_local_j = coarse_global_j - coarse_slice[1].start

            if coarse_local_j < 0 or coarse_local_j >= coarse_quantity.extent[1]:
                continue

            accum[
                coarse_local_i,
                coarse_local_j,
            ] += fine_values[
                local_i,
                local_j,
            ]

            counts[
                coarse_local_i,
                coarse_local_j,
            ] += 1

    if np.any(counts == 0):
        raise RuntimeError("fine-to-coarse reduction left " "an uncovered coarse point")

    coarse_quantity.view[:] = accum / counts


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def logical_data_with_halo(
    quantity: Quantity,
    n_points: int,
) -> np.ndarray:
    """Return the compute domain plus n_points of logical halo."""

    origin_i, origin_j = quantity.origin
    extent_i, extent_j = quantity.extent

    return np.asarray(
        quantity[
            origin_i - n_points : origin_i + extent_i + n_points,
            origin_j - n_points : origin_j + extent_j + n_points,
        ]
    )


def print_parent_tile_quantity(
    communicator: NestedCommunicator,
    parent_quantity: Quantity | None,
    nest_id: int,
    label: str,
) -> None:
    """Gather and print the parent tile supplying one nest."""

    if not communicator.is_parent_rank:
        return

    assert communicator.parent_comm is not None
    assert parent_quantity is not None

    gathered = communicator.parent_comm.tile.gather(parent_quantity)

    mapping = communicator.nested_partitioners[nest_id].mapping

    parent_partitioner = communicator.parent_partitioner

    if not isinstance(
        parent_partitioner,
        CubedSpherePartitioner,
    ):
        raise TypeError(
            "nested-grid example requires a " "CubedSpherePartitioner parent"
        )

    root_rank = parent_partitioner.tile_root_rank(mapping.parent_rank)

    if communicator.world_rank != root_rank:
        return

    assert gathered is not None

    print(
        "\n"
        "============================================================\n"
        f"PARENT TILE FOR NEST {nest_id}: "
        f"{label}\n"
        "============================================================\n"
        f"parent region = "
        f"{mapping.parent_region}\n"
        f"dims          = "
        f"{tuple(parent_quantity.dims)}\n"
        f"{np.asarray(gathered.view[:])}\n"
        "============================================================",
        flush=True,
    )


def print_nested_quantity_domains(
    communicator: NestedCommunicator,
    fine_quantity: Quantity | None,
    label: str,
    nest_id: int,
) -> None:
    """Print each nested fine Quantity in nested-rank order."""

    nested_rank = communicator.nested_rank(nest_id)

    partitioner = communicator.nested_partitioners[nest_id]

    for rank_to_print in range(communicator.nested_size(nest_id)):
        communicator.comm.Barrier()

        if nested_rank != rank_to_print:
            continue

        assert fine_quantity is not None

        j, i = partitioner.subtile_index(nested_rank)

        logical_data = logical_data_with_halo(
            fine_quantity,
            FINE_EXCHANGE_HALO,
        )

        compute_data = np.asarray(fine_quantity.view[:])

        print(
            "\n"
            "------------------------------------------------------------\n"
            f"NESTED FINE QUANTITY: {label}\n"
            "------------------------------------------------------------\n"
            f"world rank         = "
            f"{communicator.world_rank}\n"
            f"nest id            = "
            f"{nest_id}\n"
            f"nested rank        = "
            f"{nested_rank}\n"
            f"nested position    = "
            f"(j={j}, i={i})\n"
            f"dims               = "
            f"{tuple(fine_quantity.dims)}\n"
            f"origin             = "
            f"{fine_quantity.origin}\n"
            f"compute extent     = "
            f"{fine_quantity.extent}\n"
            f"logical halo shape = "
            f"{logical_data.shape}\n"
            "\nLOGICAL COMPUTE + HALO DOMAIN:\n"
            f"{logical_data}\n"
            "\nCOMPUTE DOMAIN:\n"
            f"{compute_data}\n"
            "------------------------------------------------------------",
            flush=True,
        )

    communicator.comm.Barrier()


def print_configuration(
    communicator: NestedCommunicator,
) -> None:
    """Print the decomposition once from world rank zero."""

    if communicator.world_rank != 0:
        return

    print(
        "\nNDSL nested-grid example\n"
        f"  parent tile size   = "
        f"{PARENT_NX} x {PARENT_NY}\n"
        f"  parent layout/tile = "
        f"{PARENT_LAYOUT}\n"
        f"  fine halo width    = "
        f"{FINE_EXCHANGE_HALO}\n"
        f"  parent ranks       = "
        f"{communicator.parent_size}",
        flush=True,
    )

    for (
        nest_id,
        partitioner,
    ) in communicator.nested_partitioners.items():
        mapping = partitioner.mapping

        config = NESTS[nest_id]

        print(
            f"\n  nest {nest_id}\n"
            f"    parent region       = "
            f"{mapping.parent_region}\n"
            f"    parent anchor rank  = "
            f"{mapping.parent_rank}\n"
            f"    parent start        = "
            f"{mapping.parent_start}\n"
            f"    parent extent       = "
            f"{mapping.parent_extent}\n"
            f"    refinement ratio    = "
            f"{mapping.refinement_ratio}\n"
            f"    nested fine extent  = "
            f"{mapping.fine_extent}\n"
            f"    nested layout       = "
            f"{partitioner.layout}\n"
            f"    coarse halo width   = "
            f"{config.coarse_halo}\n"
            f"    nested world ranks  = "
            f"{communicator.nested_world_ranks[nest_id]}",
            flush=True,
        )


def print_rank_mapping(
    communicator: NestedCommunicator,
) -> None:
    """Print every world's role."""

    for rank_to_print in range(communicator.world_size):
        communicator.comm.Barrier()

        if communicator.world_rank != rank_to_print:
            continue

        if communicator.is_parent_rank:
            parent_rank = communicator.parent_rank
            assert parent_rank is not None

            tile = communicator.parent_partitioner.tile_index(parent_rank)

            print(
                f"world rank "
                f"{communicator.world_rank}: "
                f"parent rank {parent_rank}, "
                f"parent tile {tile}",
                flush=True,
            )
        else:
            if len(communicator.nested_comms) != 1:
                raise RuntimeError(
                    "example expects each nested " "rank to belong to one nest"
                )

            nest_id = next(iter(communicator.nested_comms))

            nested_rank = communicator.nested_rank(nest_id)
            assert nested_rank is not None

            partitioner = communicator.nested_partitioners[nest_id]

            j, i = partitioner.subtile_index(nested_rank)

            print(
                f"world rank "
                f"{communicator.world_rank}: "
                f"nest {nest_id}, "
                f"nested rank {nested_rank}, "
                f"position=(j={j}, i={i})",
                flush=True,
            )

    communicator.comm.Barrier()


def check_configuration(
    communicator: NestedCommunicator,
) -> None:
    """Check communicator sizes and nest mappings."""

    expected_parent_size = 6 * PARENT_LAYOUT[0] * PARENT_LAYOUT[1]

    expected_world_size = expected_parent_size

    assert communicator.parent_size == expected_parent_size

    for nest_id, config in NESTS.items():
        partitioner = communicator.nested_partitioners[nest_id]

        mapping = partitioner.mapping

        assert mapping.parent_region == config.parent_region
        assert mapping.parent_start == config.parent_start
        assert mapping.parent_extent == config.parent_extent
        assert mapping.refinement_ratio == config.refinement_ratio
        assert partitioner.layout == config.layout

        expected_world_size += partitioner.total_ranks

    assert communicator.world_size == expected_world_size


# -----------------------------------------------------------------------------
# Communication test
# -----------------------------------------------------------------------------


def run_communication_case(
    *,
    communicator: NestedCommunicator,
    backend: Backend,
    nested_context: NestedGridContext | None,
    nest_id: int,
    label: str,
    dims: HorizontalDims,
    coarse_n_points: int,  # NEW
) -> None:
    """Run one same-resolution transport + external prolongation case."""

    parent_quantity: Quantity | None = None
    nested_coarse_quantity: Quantity | None = None
    fine_quantity: Quantity | None = None

    if communicator.is_parent_rank:
        parent_quantity = make_parent_quantity(
            communicator=communicator,
            backend=backend,
            dims=dims,
        )

        initialize_parent_quantity(
            parent_quantity,
            communicator,
            nest_id,
        )

    else:
        nested_rank = communicator.nested_rank(nest_id)

        if nested_rank is not None:
            assert nested_context is not None

            nested_coarse_quantity = make_nested_coarse_quantity(
                nested_context,
                dims,
            )

            fine_quantity = make_nested_fine_quantity(
                nested_context,
                dims,
            )

            initialize_nested_quantity(
                fine_quantity,
                nested_rank,
            )

    communicator.comm.Barrier()

    if communicator.world_rank == 0:
        print(
            f"\nTesting nest {nest_id}: " f"{label}, dims={dims}",
            flush=True,
        )

    print_parent_tile_quantity(
        communicator=communicator,
        parent_quantity=parent_quantity,
        nest_id=nest_id,
        label=label,
    )

    # CHANGED:
    # Fine/fine communication and parent->nested coarse transport now
    # have independent halo widths.
    communicator.update_nested_boundaries(
        nest_id=nest_id,
        parent_quantity=parent_quantity,
        nested_coarse_quantity=nested_coarse_quantity,
        fine_quantity=fine_quantity,
        fine_n_points=FINE_EXCHANGE_HALO,
        coarse_n_points=coarse_n_points,
    )

    nested_rank = communicator.nested_rank(nest_id)

    if nested_rank is not None:
        assert nested_coarse_quantity is not None
        assert fine_quantity is not None

        # Numerical interpolation policy remains outside NDSL.
        prolong_coarse_to_fine_halo(
            coarse_quantity=(nested_coarse_quantity),
            fine_quantity=fine_quantity,
            partitioner=(communicator.nested_partitioners[nest_id]),
            nested_rank=nested_rank,
            n_points=FINE_EXCHANGE_HALO,
        )

    print_nested_quantity_domains(
        communicator=communicator,
        fine_quantity=fine_quantity,
        label=label,
        nest_id=nest_id,
    )

    communicator.comm.Barrier()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    world_comm = MPIComm()
    world_rank = world_comm.Get_rank()

    backend = Backend("st:numpy:cpu:IJK")

    parent_partitioner = CubedSpherePartitioner(
        tile=TilePartitioner(layout=PARENT_LAYOUT)
    )

    nested_partitioners = make_nested_partitioners(parent_partitioner)

    parent_size = parent_partitioner.total_ranks

    nested_world_ranks = assign_nested_world_ranks(
        parent_size,
        nested_partitioners,
    )

    this_nest_id = find_local_nest_id(
        world_rank,
        nested_world_ranks,
    )

    is_parent_rank = world_rank < parent_size

    if is_parent_rank:
        color = 0
    else:
        if this_nest_id is None:
            raise RuntimeError(
                f"world rank {world_rank} " "is neither parent nor nested"
            )

        nest_order = list(nested_partitioners).index(this_nest_id)

        color = nest_order + 1

    raw_role_comm = world_comm.Split(
        color=color,
        key=world_rank,
    )

    # MPIComm.Split currently returns raw mpi4py Comm.
    role_comm = MPIComm()
    role_comm._comm = raw_role_comm

    parent_comm = None
    nested_comms: dict[
        int,
        NestTileCommunicator,
    ] = {}

    if is_parent_rank:
        parent_comm = CubedSphereCommunicator(
            comm=role_comm,
            partitioner=(parent_partitioner),
        )
    else:
        assert this_nest_id is not None

        nested_comms[this_nest_id] = NestTileCommunicator(
            comm=role_comm,
            partitioner=(nested_partitioners[this_nest_id]),
        )

    communicator = NestedCommunicator(
        comm=world_comm,
        parent_partitioner=(parent_partitioner),
        nested_partitioners=(nested_partitioners),
        nested_world_ranks=(nested_world_ranks),
        parent_comm=parent_comm,
        nested_comms=nested_comms,
    )

    check_configuration(communicator)
    print_configuration(communicator)
    print_rank_mapping(communicator)

    coarse_metric_terms: MetricTerms | None = None

    reference_metrics_by_ratio: dict[
        int,
        MetricTerms,
    ] = {}

    if communicator.is_parent_rank:
        coarse_metric_terms = make_parent_metric_terms(
            communicator=communicator,
            backend=backend,
            nx_tile=PARENT_NX,
            ny_tile=PARENT_NY,
        )

        materialize_metric_terms(
            coarse_metric_terms,
            (
                "dgrid_lon_lat",
                "area",
                "dx",
                "dy",
            ),
        )

        refinement_ratios = sorted(
            {config.refinement_ratio for config in NESTS.values()}
        )

        for refinement in refinement_ratios:
            metrics = make_parent_metric_terms(
                communicator=communicator,
                backend=backend,
                nx_tile=(PARENT_NX * refinement),
                ny_tile=(PARENT_NY * refinement),
            )

            materialize_metric_terms(
                metrics,
                (
                    "agrid_lon_lat",
                    "dgrid_lon_lat",
                    "area",
                    "dx",
                    "dy",
                    "dxa",
                    "dya",
                    "dxc",
                    "dyc",
                ),
            )

            reference_metrics_by_ratio[refinement] = metrics

    for nest_id, config in NESTS.items():
        fine_metrics = (
            reference_metrics_by_ratio[config.refinement_ratio]
            if communicator.is_parent_rank
            else None
        )

        check_physical_refinement(
            communicator=communicator,
            coarse_metrics=(coarse_metric_terms),
            fine_metrics=fine_metrics,
            nest_id=nest_id,
        )

    reference_data_by_nest: dict[
        int,
        ReferenceMetricData | None,
    ] = {}

    for nest_id, config in NESTS.items():
        fine_metrics = (
            reference_metrics_by_ratio[config.refinement_ratio]
            if communicator.is_parent_rank
            else None
        )

        reference_data_by_nest[nest_id] = get_reference_metric_data(
            communicator=communicator,
            fine_metrics=fine_metrics,
            nest_id=nest_id,
        )

        root = nested_partitioners[nest_id].mapping.parent_rank

        reference_data_by_nest[nest_id] = communicator.comm.bcast(
            reference_data_by_nest[nest_id],
            root=root,
        )

    nested_context: NestedGridContext | None = None

    if communicator.is_nested_rank:
        assert this_nest_id is not None

        reference = reference_data_by_nest[this_nest_id]
        assert reference is not None

        nested_context = make_nested_grid_context(
            communicator=communicator,
            backend=backend,
            reference=reference,
            nest_id=this_nest_id,
            coarse_n_points=NESTS[this_nest_id].coarse_halo,
        )

    check_nested_physical_grid(
        communicator,
        nested_context,
    )

    if communicator.is_nested_rank:
        assert this_nest_id is not None

        print_nested_grid_summary(
            communicator,
            nested_context,
            this_nest_id,
        )

    for nest_id in NESTS:
        for label, dims in STAGGERINGS:
            run_communication_case(
                communicator=communicator,
                backend=backend,
                nested_context=(nested_context if this_nest_id == nest_id else None),
                nest_id=nest_id,
                label=label,
                dims=dims,
                coarse_n_points=NESTS[nest_id].coarse_halo,
            )

    if communicator.world_rank == 0:
        print(
            "\nNested-grid communication PASSED",
            flush=True,
        )


if __name__ == "__main__":
    main()
