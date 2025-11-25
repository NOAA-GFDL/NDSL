from collections.abc import Iterable
from typing import Self

import ndsl.constants as constants
from ndsl.comm.partitioner import TilePartitioner
from ndsl.constants import N_HALO_DEFAULT
from ndsl.initialization.grid_sizer import GridSizer


class SubtileGridSizer(GridSizer):
    @classmethod
    def from_tile_params(
        cls,
        nx_tile: int,
        ny_tile: int,
        nz: int,
        n_halo: int,
        layout: tuple[int, int],
        *,
        data_dimensions: dict[str, int] | None = None,
        tile_partitioner: TilePartitioner | None = None,
        tile_rank: int = 0,
    ) -> Self:
        """Create a SubtileGridSizer from parameters about the full tile.

        Args:
            nx_tile: number of x cell centers on the tile
            ny_tile: number of y cell centers on the tile
            nz: number of vertical levels
            n_halo: number of halo points
            data_dimensions: lengths of any non-x/y/z dimensions,
                such as land or radiation dimensions
            layout: (y, x) number of ranks along tile edges
            tile_partitioner (optional): partitioner object for the tile. By default, a
                TilePartitioner is created with the given layout
            tile_rank (optional): rank of this subtile.
        """
        if data_dimensions is None:
            data_dimensions = {}

        if tile_partitioner is None:
            tile_partitioner = TilePartitioner(layout)
        y_slice, x_slice = tile_partitioner.subtile_slice(
            tile_rank,
            [constants.Y_DIM, constants.X_DIM],
            [ny_tile, nx_tile],
            overlap=True,
        )
        nx = x_slice.stop - x_slice.start
        ny = y_slice.stop - y_slice.start

        # TODO: Remove after vector halo update issue resolved
        if nx <= n_halo:
            raise Exception(
                "SubtileGridSizer::from_tile_params: Compute domain extent must be greater than halo size"
            )
        if ny <= n_halo:
            raise Exception(
                "SubtileGridSizer::from_tile_params: Compute domain extent must be greater than halo size"
            )

        return cls(nx, ny, nz, n_halo, data_dimensions)

    @classmethod
    def from_namelist(
        cls,
        namelist: dict,
        tile_partitioner: TilePartitioner | None = None,
        tile_rank: int = 0,
    ) -> Self:
        """Create a SubtileGridSizer from a Fortran namelist.

        Args:
            namelist: A namelist for the fv3gfs fortran model
            tile_partitioner (optional): a partitioner to use for segmenting the tile.
                By default, a TilePartitioner is used.
            tile_rank (optional): current rank on tile. Default is 0. Only matters if
                different ranks have different domain shapes. If tile_partitioner
                is a TilePartitioner, this argument does not matter.
        """
        if "fv_core_nml" in namelist.keys():
            layout = namelist["fv_core_nml"]["layout"]
            # npx and npy in the namelist are cell centers, but npz is mid levels
            nx_tile = namelist["fv_core_nml"]["npx"] - 1
            ny_tile = namelist["fv_core_nml"]["npy"] - 1
            nz = namelist["fv_core_nml"]["npz"]
        elif "nx_tile" in namelist.keys():
            layout = namelist["layout"]
            # everything is cell centered in this format
            nx_tile = namelist["nx_tile"]
            ny_tile = namelist["nx_tile"]
            nz = namelist["nz"]
        else:
            raise KeyError(
                "Namelist format is unrecognized, "
                "expected to find nx_tile or fv_core_nml"
            )
        return cls.from_tile_params(
            nx_tile=nx_tile,
            ny_tile=ny_tile,
            nz=nz,
            n_halo=N_HALO_DEFAULT,
            layout=layout,
            tile_partitioner=tile_partitioner,
            tile_rank=tile_rank,
        )

    @property
    def dim_extents(self) -> dict[str, int]:
        return_dict = self.data_dimensions.copy()
        return_dict.update(
            {
                constants.X_DIM: self.nx,
                constants.X_INTERFACE_DIM: self.nx + 1,
                constants.Y_DIM: self.ny,
                constants.Y_INTERFACE_DIM: self.ny + 1,
                constants.Z_DIM: self.nz,
                constants.Z_INTERFACE_DIM: self.nz + 1,
            }
        )
        return return_dict

    def get_origin(self, dims: Iterable[str]) -> tuple[int, ...]:
        return_list = [
            self.n_halo if dim in constants.HORIZONTAL_DIMS else 0 for dim in dims
        ]
        return tuple(return_list)

    def get_extent(self, dims: Iterable[str]) -> tuple[int, ...]:
        extents = self.dim_extents
        return tuple(extents[dim] for dim in dims)

    def get_shape(self, dims: Iterable[str]) -> tuple[int, ...]:
        shape_dict = self.data_dimensions.copy()
        # must pad non-interface variables to have the same shape as interface variables
        shape_dict.update(
            {
                constants.X_DIM: self.nx + 1 + 2 * self.n_halo,
                constants.X_INTERFACE_DIM: self.nx + 1 + 2 * self.n_halo,
                constants.Y_DIM: self.ny + 1 + 2 * self.n_halo,
                constants.Y_INTERFACE_DIM: self.ny + 1 + 2 * self.n_halo,
                constants.Z_DIM: self.nz + 1,
                constants.Z_INTERFACE_DIM: self.nz + 1,
            }
        )
        return tuple(shape_dict[dim] for dim in dims)
