import argparse
import os
import shutil
from typing import Any, Dict, Optional

import f90nml
import numpy as np
import xarray as xr


try:
    import serialbox
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Serialbox couldn't be imported, make sure it's in your PYTHONPATH or you env"
    )


def get_parser():
    parser = argparse.ArgumentParser("Converts Serialbox data to netcdf")
    parser.add_argument(
        "data_path",
        type=str,
        help="path of serialbox data to convert",
    )
    parser.add_argument(
        "output_path", type=str, help="output directory where netcdf data will be saved"
    )
    parser.add_argument(
        "-dn",
        "--data_name",
        type=str,
        help="[Optional] Give the name of the data, will default to Generator_rankX",
    )
    parser.add_argument(
        "-m",
        "--merge",
        action="store_true",
        default=False,
        help="merges datastreams blocked into separate savepoints",
    )
    return parser


def read_serialized_data(serializer, savepoint, variable):
    data = serializer.read(variable, savepoint)
    if len(data.flatten()) == 1:
        return data[0]
    data[data == 1e40] = 0.0
    return data


def get_all_savepoint_names(serializer):
    savepoint_names = set()
    for savepoint in serializer.savepoint_list():
        savepoint_names.add(savepoint.name)
    return savepoint_names


def get_serializer(data_path: str, rank: int, data_name: Optional[str] = None):
    if data_name:
        name = data_name
    else:
        name = f"Generator_rank{rank}"
    return serialbox.Serializer(serialbox.OpenModeKind.Read, data_path, name)  # type: ignore


def main(
    data_path: str,
    output_path: str,
    merge_blocks: bool,
    data_name: Optional[str] = None,
):
    os.makedirs(output_path, exist_ok=True)
    namelist_filename_in = os.path.join(data_path, "input.nml")

    if not os.path.exists(namelist_filename_in):
        raise FileNotFoundError(f"Can't find input.nml in {data_path}. Required.")

    namelist_filename_out = os.path.join(output_path, "input.nml")
    if namelist_filename_out != namelist_filename_in:
        shutil.copyfile(os.path.join(data_path, "input.nml"), namelist_filename_out)
    namelist = f90nml.read(namelist_filename_out)
    fv_core_nml: Dict[str, Any] = namelist["fv_core_nml"]  # type: ignore
    if fv_core_nml["grid_type"] <= 3:
        total_ranks = 6 * fv_core_nml["layout"][0] * fv_core_nml["layout"][1]
    else:
        total_ranks = fv_core_nml["layout"][0] * fv_core_nml["layout"][1]
    nx = int((fv_core_nml["npx"] - 1) / (fv_core_nml["layout"][0]))
    ny = int((fv_core_nml["npy"] - 1) / (fv_core_nml["layout"][1]))

    # all ranks have the same names, just look at first one
    serializer_0 = get_serializer(data_path, rank=0, data_name=data_name)

    savepoint_names = get_all_savepoint_names(serializer_0)
    for savepoint_name in sorted(list(savepoint_names)):
        rank_list = []
        names_list = list(
            serializer_0.fields_at_savepoint(
                serializer_0.get_savepoint(savepoint_name)[0]
            )
        )
        print(f"Exporting {savepoint_name}")
        serializer_list = []
        for rank in range(total_ranks):
            serializer = get_serializer(data_path, rank, data_name)
            serializer_list.append(serializer)
            savepoints = serializer.get_savepoint(savepoint_name)
            rank_data: Dict[str, Any] = {}
            for name in set(names_list):
                rank_data[name] = []
                for savepoint in savepoints:
                    rank_data[name].append(
                        read_serialized_data(serializer, savepoint, name)
                    )
                nblocks = len(rank_data[name])
                if merge_blocks and len(rank_data[name]) > 1:
                    full_data = np.array(rank_data[name])
                    if len(full_data.shape) > 1:
                        if nx * ny == full_data.shape[0] * full_data.shape[1]:
                            # If we have an (i, x) array from each block reshape it
                            new_shape = (nx, ny) + full_data.shape[2:]
                            full_data = full_data.reshape(new_shape)
                        else:
                            # We have one array for all blocks
                            # could be a k-array or something else, so we take one copy
                            # TODO: is there a decent check for this?
                            full_data = full_data[0]
                    elif len(full_data.shape) == 1:
                        # if it's a scalar from each block then just take one
                        full_data = full_data[0]
                    else:
                        raise IndexError(f"{name} data appears to be empty")
                    rank_data[name] = [full_data]
            rank_list.append(rank_data)
        if merge_blocks:
            n_savepoints = 1
        else:
            n_savepoints = len(savepoints)  # checking from last rank is fine
        data_vars = {}
        if n_savepoints > 0:
            encoding = {}
            names_indices = np.sort(list(set(names_list).difference(["rank"])))
            for varname in names_indices:
                # Check that all ranks have the same size. If not, aggregate and
                # feedback on one rank
                collapse_all_ranks = False
                data_shape = list(rank_list[0][varname][0].shape)
                print(f"  Exporting {varname} - {data_shape}")
                for rank in range(total_ranks):
                    this_shape = list(rank_list[rank][varname][0].shape)
                    if data_shape != this_shape:
                        if len(data_shape) != 1:
                            raise ValueError(
                                "Arrays have different dimensions. "
                                f"E.g. rank 0 is {data_shape} "
                                f"and rank {rank} is {this_shape} "
                            )
                        else:
                            print(
                                f"... different shape for {varname} across ranks, collapsing in on rank."
                            )
                            collapse_all_ranks = True
                            break

                if savepoint_name in [
                    "FVDynamics-In",
                    "FVDynamics-Out",
                    "Driver-In",
                    "Driver-Out",
                ]:
                    if varname in [
                        "qvapor",
                        "qliquid",
                        "qice",
                        "qrain",
                        "qsnow",
                        "qgraupel",
                        "qo3mr",
                        "qsgs_tke",
                    ]:
                        data_vars[varname] = get_data(
                            data_shape, total_ranks, n_savepoints, rank_list, varname
                        )[:, :, 3:-3, 3:-3, :]
                    else:
                        data_vars[varname] = get_data(
                            data_shape, total_ranks, n_savepoints, rank_list, varname
                        )
                elif collapse_all_ranks:
                    data_vars[varname] = get_data_collapse_all_ranks(
                        total_ranks, n_savepoints, rank_list, varname
                    )
                else:
                    data_vars[varname] = get_data(
                        data_shape, total_ranks, n_savepoints, rank_list, varname
                    )
                if len(data_shape) > 2:
                    encoding[varname] = {"zlib": True, "complevel": 1}

            dataset = xr.Dataset(data_vars=data_vars)
            dataset.to_netcdf(
                os.path.join(output_path, f"{savepoint_name}.nc"), encoding=encoding
            )


def get_data_collapse_all_ranks(total_ranks, n_savepoints, output_list, varname):
    if total_ranks <= 0:
        return xr.DataArray([], dims=[])
    # Build array shape - we hypothesis there's only 1 axis
    K_shape = 0
    for rank in range(total_ranks):
        assert len(output_list[rank][varname][0].shape) == 1
        K_shape = K_shape + output_list[rank][varname][0].shape[0]

    array = np.full(
        [n_savepoints, 1] + [K_shape],
        fill_value=np.nan,
        dtype=output_list[0][varname][0].dtype,
    )
    data = xr.DataArray(array, dims=["savepoint", "rank", f"dim_{varname}"])
    last_size = 0
    for rank in range(total_ranks):
        for i_savepoint in range(n_savepoints):
            rank_data = output_list[rank][varname][i_savepoint]
            rank_data_size = rank_data.shape[0]
            data[i_savepoint, 0, last_size : last_size + rank_data_size] = rank_data[:]
            last_size += rank_data_size

    return data


def get_data(data_shape, total_ranks, n_savepoints, output_list, varname):
    if total_ranks <= 0:
        return xr.DataArray([], dims=[])
    # Read in dtype
    varname_dtype = output_list[0][varname][0].dtype
    # Build data array
    array = np.full(
        [n_savepoints, total_ranks] + data_shape,
        fill_value=np.nan,
        dtype=varname_dtype,
    )
    dims = ["savepoint", "rank"] + [
        f"dim_{varname}_{i}" for i in range(len(data_shape))
    ]
    data = xr.DataArray(array, dims=dims)
    for rank in range(total_ranks):
        for i_savepoint in range(n_savepoints):
            if len(data_shape) > 0:
                data[i_savepoint, rank, :] = output_list[rank][varname][i_savepoint]
            else:
                data[i_savepoint, rank] = output_list[rank][varname][i_savepoint]
    return data


def entry_point():
    parser = get_parser()
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        output_path=args.output_path,
        merge_blocks=args.merge,
        data_name=args.data_name,
    )


if __name__ == "__main__":
    entry_point()
