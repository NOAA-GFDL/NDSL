import dataclasses
import os
from typing import Dict, Protocol, Union

import numpy as np
import xarray as xr

from ndsl.stencils.testing.grid import Grid  # type: ignore


def dataset_to_dict(ds: xr.Dataset) -> Dict[str, Union[np.ndarray, float, int]]:
    return {
        name: _process_if_scalar(array.values) for name, array in ds.data_vars.items()
    }


def _process_if_scalar(value: np.ndarray) -> Union[np.ndarray, float, int]:
    if len(value.shape) == 0:
        return value.max()  # trick to make sure we get the right type back
    else:
        return value


class DataLoader:
    def __init__(self, rank: int, data_path: str):
        self._data_path = data_path
        self._rank = rank

    def load(
        self,
        name: str,
        postfix: str = "",
        i_call: int = 0,
    ) -> Dict[str, Union[np.ndarray, float, int]]:
        return dataset_to_dict(
            xr.open_dataset(os.path.join(self._data_path, f"{name}{postfix}.nc"))
            .isel(rank=self._rank)
            .isel(savepoint=i_call)
        )


class Translate(Protocol):
    def collect_input_data(self, ds: xr.Dataset) -> dict:
        ...

    def compute(self, data: dict):
        ...

    def extra_data_load(self, data_loader: DataLoader):
        ...


@dataclasses.dataclass
class SavepointCase:
    """
    Represents a savepoint with data on one rank.
    """

    savepoint_name: str
    data_dir: str
    i_call: int
    testobj: Translate
    grid: Grid
    sort_report: str
    no_report: bool

    def __str__(self):
        return f"{self.savepoint_name}-rank={self.grid.rank}-call={self.i_call}"

    @property
    def exists(self) -> bool:
        return (
            xr.open_dataset(
                os.path.join(self.data_dir, f"{self.savepoint_name}-In.nc")
            ).sizes["rank"]
            > self.grid.rank
        )

    @property
    def ds_in(self) -> xr.Dataset:
        return (
            xr.open_dataset(os.path.join(self.data_dir, f"{self.savepoint_name}-In.nc"))
            .isel(rank=self.grid.rank)
            .isel(savepoint=self.i_call)
        )

    @property
    def ds_out(self) -> xr.Dataset:
        return (
            xr.open_dataset(
                os.path.join(self.data_dir, f"{self.savepoint_name}-Out.nc")
            )
            .isel(rank=self.grid.rank)
            .isel(savepoint=self.i_call)
        )
