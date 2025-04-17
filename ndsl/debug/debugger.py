import xarray as xr
from ndsl.debug.mode import DebugMode
import dataclasses
from ndsl.quantity import Quantity
from functools import wraps
import inspect
from typing import Any, Callable


@dataclasses.dataclass
class Debugger:
    """Debugger relying on `ndsl.debug.config` for setup capable
    of doing automatic data save on external configuration."""

    # Configuration
    mode: DebugMode = DebugMode.NDebug
    do_input_dump: bool = False
    do_output_dump: bool = False
    calls_count: dict[str, int] = dataclasses.field(default_factory=dict)
    dir_name: str = "./"
    # Runtime data
    rank: int = -1
    stencils_or_class: list[str] = dataclasses.field(default_factory=list)

    def can_save(self, savename: str) -> bool:
        """Is this savename configured for data (input ot output) save"""
        if self.do_output_dump and not self.do_input_dump:
            return False

        if savename in self.stencils_or_class:
            return True

        return False

    def save_as_dataset(self, data_as_dict, savename, is_in):
        """Save dictionnary of data to NetCDF

        Note: Unknown types in the dictionnary won't be saved.
        """
        if not self.can_save(savename):
            return

        data_arrays = {}
        for name, data in data_as_dict.items():
            if isinstance(data, Quantity):
                data_arrays[name] = xr.DataArray(data.data)
            elif not hasattr(data, "shape"):
                data_arrays[name] = xr.DataArray(data)
            else:
                data_arrays[name] = xr.DataArray(
                    data, dims=[f"dim_{i}_{s}" for i, s in enumerate(data.shape)]
                )

        call_count = (
            self.calls_count[savename] if savename in self.calls_count.keys() else 0
        )
        try:
            xr.Dataset(data_arrays).to_netcdf(
                f"{self.dir_name}/{savename}-R{self.rank}-Call{call_count}-{'In' if is_in else 'Out'}.nc4"
            )
        except ValueError as e:
            from ndsl import ndsl_log

            ndsl_log.error(f"[DebugInfo] Failure to save {savename}: {e}")

    def increment_call_count(self, savename: str):
        """Increment the call count for this savename"""
        if savename not in self.calls_count.keys():
            self.calls_count[savename] = 0
        self.calls_count[savename] += 1

    @staticmethod
    def instrument(func) -> Callable:
        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any):
            savename = func.__qualname__
            if not ndsl_debugger.can_save(savename):
                return func(self, *args, **kwargs)
            params = inspect.signature(func).parameters
            data_as_dict = {}

            # Positional
            positional_count = 0
            for name, param in params.items():
                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    if positional_count == 0:  # self
                        positional_count += 1
                        continue
                    if positional_count < len(args) + 1:
                        data_as_dict[name] = args[positional_count - 1]
                        positional_count += 1
            # Keyword arguments
            for name, value in kwargs.items():
                if name in params:
                    data_as_dict[name] = value
            ndsl_debugger.save_as_dataset(data_as_dict, func.__qualname__, is_in=True)
            r = func(self, *args, **kwargs)
            ndsl_debugger.save_as_dataset(data_as_dict, func.__qualname__, is_in=False)
            ndsl_debugger.increment_call_count(savename)
            return r

        return wrapper
