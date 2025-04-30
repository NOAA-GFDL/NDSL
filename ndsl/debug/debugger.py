import dataclasses
import os

import xarray as xr

from ndsl.debug.mode import DebugMode
from ndsl.quantity import Quantity
from ndsl.logging import ndsl_log


@dataclasses.dataclass
class Debugger:
    """Debugger relying on `ndsl.debug.config` for setup capable
    of doing automatic data save on external configuration."""

    # Configuration
    mode: DebugMode = DebugMode.NDebug
    stencils_or_class: list[str] = dataclasses.field(default_factory=list)
    track_parameter_by_name: list[str] = dataclasses.field(default_factory=list)
    dir_name: str = "./"

    # Runtime data
    rank: int = -1
    calls_count: dict[str, int] = dataclasses.field(default_factory=dict)
    track_parameter_count: dict[str, int] = dataclasses.field(default_factory=dict)

    def _to_xarray(self, data, name) -> xr.DataArray:
        if isinstance(data, Quantity):
            mem = data.data
            shp = data.data.shape
        elif hasattr(data, "shape"):
            mem = data
            shp = data.shape
        else:
            # Global catch-all attempt
            try:
                d = xr.DataArray(data)
            except ValueError as e:
                ndsl_log.error(f"[DebugInfo] Failure to save {name}: {e}")
                return xr.DataArray([0])
            return d
        return xr.DataArray(mem, dims=[f"dim_{i}_{s}" for i, s in enumerate(shp)])

    def track_data(self, data_as_dict, source_as_name, is_in) -> None:
        for name, data in data_as_dict.items():
            if name not in self.track_parameter_by_name:
                continue

            if name not in self.track_parameter_count:
                self.track_parameter_count[name] = 0
            count = self.track_parameter_count[name]

            path = f"{self.dir_name}/debug/tracks/{name}/R{self.rank}/"
            os.makedirs(path, exist_ok=True)
            path = (
                f"{path}/{count}_{name}_{source_as_name}-{'In' if is_in else 'Out'}.nc4"
            )
            try:
                self._to_xarray(data, name).to_netcdf(path)
            except ValueError as e:
                from ndsl import ndsl_log

                ndsl_log.error(f"[DebugInfo] Failure to save {data}: {e}")

            self.track_parameter_count[name] += 1

    def save_as_dataset(self, data_as_dict, savename, is_in) -> None:
        """Save dictionnary of data to NetCDF

        Note: Unknown types in the dictionnary won't be saved.
        """
        if savename not in self.stencils_or_class:
            return

        data_arrays = {}
        for name, data in data_as_dict.items():
            data_arrays[name] = self._to_xarray(data, name)

        call_count = (
            self.calls_count[savename] if savename in self.calls_count.keys() else 0
        )
        path = f"{self.dir_name}/debug/savepoints/R{self.rank}/"
        os.makedirs(path, exist_ok=True)
        path = f"{path}/{savename}-Call{call_count}-{'In' if is_in else 'Out'}.nc4"
        try:
            xr.Dataset(data_arrays).to_netcdf(path)
        except ValueError as e:
            ndsl_log.error(f"[DebugInfo] Failure to save {savename}: {e}")

    def increment_call_count(self, savename: str):
        """Increment the call count for this savename"""
        if savename not in self.calls_count.keys():
            self.calls_count[savename] = 0
        self.calls_count[savename] += 1
