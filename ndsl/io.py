from typing import TextIO

import cftime
import xarray as xr

import ndsl.filesystem as filesystem
from ndsl.quantity import Quantity


# Calendar constant values copied from time_manager in FMS
THIRTY_DAY_MONTHS = 1
JULIAN = 2
GREGORIAN = 3
NOLEAP = 4
FMS_TO_CFTIME_TYPE = {
    THIRTY_DAY_MONTHS: cftime.Datetime360Day,
    JULIAN: cftime.DatetimeJulian,
    GREGORIAN: cftime.DatetimeGregorian,  # Not a valid calendar in FV3GFS
    NOLEAP: cftime.DatetimeNoLeap,
}


def to_xarray_dataset(state) -> xr.Dataset:
    data_vars = {
        name: value.data_as_xarray for name, value in state.items() if name != "time"
    }
    if "time" in state:
        data_vars["time"] = state["time"]
    return xr.Dataset(data_vars=data_vars)


def write_state(state: dict, filename: str) -> None:
    """Write a model state to a NetCDF file.

    Args:
        state: a model state dictionary
        filename: local or remote location to write the NetCDF file
    """
    if "time" not in state:
        raise ValueError('state must include a value for "time"')
    ds = to_xarray_dataset(state)
    with filesystem.open(filename, "wb") as f:
        ds.to_netcdf(f)


def _extract_time(value: xr.DataArray) -> cftime.datetime:
    """Extract time value from read-in state."""
    if value.ndim > 0:
        raise ValueError(
            f"State must be representative of a single scalar time. Got {value}."
        )
    time = value.item()
    if not isinstance(time, cftime.datetime):
        raise ValueError(
            "Time in stored state does not have the proper metadata "
            "to be decoded as a cftime.datetime object."
        )
    return time


def read_state(filename: str) -> dict:
    """Read a model state from a NetCDF file.

    Args:
        filename: local or remote location of the NetCDF file

    Returns:
        state: a model state dictionary
    """
    out_dict = {}
    with filesystem.open(filename, "rb") as f:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(f, decode_times=time_coder)
        for name, value in ds.data_vars.items():
            if name == "time":
                out_dict[name] = _extract_time(value)
            else:
                out_dict[name] = Quantity.from_data_array(value)
    return out_dict


def _get_integer_tokens(line, n_tokens):
    all_tokens = line.split()
    return [int(token) for token in all_tokens[:n_tokens]]


def get_current_date_from_coupler_res(file: TextIO) -> cftime.datetime:
    (fms_calendar_type,) = _get_integer_tokens(file.readline(), 1)
    file.readline()
    year, month, day, hour, minute, second = _get_integer_tokens(file.readline(), 6)
    return FMS_TO_CFTIME_TYPE[fms_calendar_type](year, month, day, hour, minute, second)
