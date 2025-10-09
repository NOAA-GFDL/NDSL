import collections

import numpy as np
import xarray as xr

from ndsl.checkpointer.base import ArrayLike, Checkpointer, SavepointName, VariableName
from ndsl.optional_imports import cupy as cp


def make_dims(
    savepoint_dim: str, label: str, data_list: list[np.ndarray]
) -> tuple[list[str], np.ndarray]:
    """
    Helper which defines dimension names for an xarray variable.

    Used to ensure no dimensions have the same name but different sizes
    when defining xarray datasets.
    """
    data = np.concatenate([array[None, :] for array in data_list], axis=0)
    dims = [savepoint_dim] + [f"{label}_dim{i}" for i in range(len(data.shape[1:]))]
    if cp and isinstance(data, cp.ndarray):
        data = data.get()
    return dims, data


class _Snapshots:
    def __init__(self) -> None:
        self._savepoints: dict[VariableName, list[SavepointName]] = (
            collections.defaultdict(list)
        )
        self._arrays: dict[VariableName, list[np.ndarray]] = collections.defaultdict(
            list
        )

    def store(
        self,
        savepoint_name: SavepointName,
        variable_name: VariableName,
        python_data: np.ndarray,
    ) -> None:
        self._savepoints[variable_name].append(savepoint_name)
        self._arrays[variable_name].append(python_data)

    @property
    def dataset(self) -> xr.Dataset:
        data_vars = {}
        for variable_name, savepoint_list in self._savepoints.items():
            savepoint_dim = f"sp_{variable_name}"
            data_vars[f"{variable_name}_savepoints"] = ([savepoint_dim], savepoint_list)
            data_vars[f"{variable_name}"] = make_dims(
                savepoint_dim, variable_name, self._arrays[variable_name]
            )
        return xr.Dataset(data_vars=data_vars)


class SnapshotCheckpointer(Checkpointer):
    """
    Checkpointer which can be used to save datasets showing the evolution
    of variables between checkpointer calls.
    """

    def __init__(self, rank: int) -> None:
        self._rank = rank
        self._snapshots = _Snapshots()

    def __call__(self, savepoint_name: SavepointName, **kwargs: ArrayLike) -> None:
        for name, value in kwargs.items():
            array_data = np.copy(value.data)
            self._snapshots.store(savepoint_name, name, array_data)

    @property
    def dataset(self) -> xr.Dataset:
        return self._snapshots.dataset

    def cleanup(self) -> None:
        self.dataset.to_netcdf(f"comparison_rank{self._rank}.nc")
