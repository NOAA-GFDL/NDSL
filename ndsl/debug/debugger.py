import dataclasses
import numbers
import os
import pathlib
from typing import Any

import pandas as pd
import xarray as xr

from ndsl.logging import ndsl_log
from ndsl.quantity import Quantity


@dataclasses.dataclass
class Debugger:
    """Debugger relying on `ndsl.debug.config` for setup capable
    of doing automatic data save on external configuration."""

    # Configuration
    stencils_or_class: list[str] = dataclasses.field(default_factory=list)
    track_parameter_by_name: list[str] = dataclasses.field(default_factory=list)
    save_compute_domain_only: bool = False
    dir_name: str = "./"

    # Runtime data
    rank: int = -1
    calls_count: dict[str, int] = dataclasses.field(default_factory=dict)
    track_parameter_count: dict[str, int] = dataclasses.field(default_factory=dict)

    def _to_xarray(self, data: Any, name: str | None) -> xr.DataArray:
        if isinstance(data, Quantity):
            if self.save_compute_domain_only:
                mem = data.field
                shp = data.field.shape
            else:
                mem = data.data
                shp = data.shape
        elif hasattr(data, "shape"):
            mem = data
            shp = data.shape
        elif (
            pd.api.types.is_numeric_dtype(data)
            or pd.api.types.is_string_dtype(data)
            or isinstance(data, numbers.Number)
        ):
            return xr.DataArray(data, name=name)
        else:
            ndsl_log.error(f"[Debugger] Cannot save data of type {type(data)}")
            return xr.DataArray([0])
        return xr.DataArray(mem, dims=[f"dim_{i}_{s}" for i, s in enumerate(shp)])

    def track_data(self, data_as_dict: dict, source_as_name: str, is_in: bool) -> None:
        for name, data in data_as_dict.items():
            if name not in self.track_parameter_by_name:
                continue

            if name not in self.track_parameter_count:
                self.track_parameter_count[name] = 0
            count = self.track_parameter_count[name]

            path = pathlib.Path(f"{self.dir_name}/debug/tracks/{name}/R{self.rank}/")
            os.makedirs(path, exist_ok=True)
            path = pathlib.Path(
                f"{path}/{count}_{name}_{source_as_name}-{'In' if is_in else 'Out'}.nc4"
            )
            try:
                self._to_xarray(data, name).to_netcdf(path)
            except ValueError as e:
                from ndsl import ndsl_log

                ndsl_log.error(f"[Debugger] Failure to save {data}: {e}")

            self.track_parameter_count[name] += 1

    def save_as_dataset(self, data_as_dict: dict, savename: str, is_in: bool) -> None:
        """Save dictionary of data to NetCDF

        Note: Unknown types in the dictionary won't be saved.
        """
        if savename not in self.stencils_or_class:
            return

        data_arrays = {}
        for name, data in data_as_dict.items():
            if dataclasses.is_dataclass(data):
                for field in dataclasses.fields(data):
                    data_arrays[f"{name}.{field.name}"] = self._to_xarray(
                        getattr(data, field.name), field.name
                    )
            else:
                data_arrays[name] = self._to_xarray(data, name)

        call_count = (
            self.calls_count[savename] if savename in self.calls_count.keys() else 0
        )
        path = pathlib.Path(f"{self.dir_name}/debug/savepoints/R{self.rank}/")
        os.makedirs(path, exist_ok=True)
        path = pathlib.Path(
            f"{path}/{savename}-Call{call_count}-{'In' if is_in else 'Out'}.nc4"
        )
        try:
            xr.Dataset(data_arrays).to_netcdf(path)
        except ValueError as e:
            ndsl_log.error(f"[DebugInfo] Failure to save {savename}: {e}")

    def increment_call_count(self, savename: str) -> None:
        """Increment the call count for this savename"""
        if savename not in self.calls_count.keys():
            self.calls_count[savename] = 0
        self.calls_count[savename] += 1
