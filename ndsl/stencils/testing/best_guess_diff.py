import argparse
import pathlib

import numpy as np
import xarray as xr
import yaml


def get_parser():
    parser = argparse.ArgumentParser(
        "Attempt to diff two NetCDFs with similar data."
        "Differences that can be reconcialed are strict domain vs halo, variable name mapping."
        "They program will report on assumptions taken."
    )
    parser.add_argument(
        "netcdf_A",
        type=str,
        help="path of NetCDFs, named A in the logs.",
    )
    parser.add_argument(
        "netcdf_B",
        type=str,
        help="path of NetCDFs, named B in the logs.",
    )
    parser.add_argument(
        "-nm",
        "--name_mapping",
        type=str,
        help="[Optional] Yaml file describing the mapping of the names.",
    )
    parser.add_argument(
        "-ha",
        "--halo",
        type=int,
        default=3,
        help="[Optional] Halo size if any, default to 3.",
    )
    return parser


def main(
    netcdf_A: str,
    netcdf_B: str,
    name_mapping: str | None = None,
    halo: int = 3,
):
    A = xr.open_dataset(netcdf_A)
    B = xr.open_dataset(netcdf_B)
    name_map = {}
    if name_mapping is not None:
        with open(name_mapping) as f:
            name_map = yaml.safe_load(f)

    dataset = {}
    for name_A, data_A in A.items():
        print(f"Best guess for {name_A} from A:")
        # Resolve name
        resolved_name_B = None
        if name_A not in B.keys():
            if name_A not in name_map.keys():
                print("  [Failed] name can't be found in B nor in name mapping")
            else:
                resolved_name_B = name_map[name_A]
        else:
            resolved_name_B = name_A

        if resolved_name_B is None:
            continue
        print(f"  [Hyp] use {resolved_name_B} for B")

        # Resolve domain size
        data_B = B[resolved_name_B]
        if len(data_A.shape) >= 5:
            print(
                "  [Hyp] A data dims are >= 5, assuming savepoints/rank are the firt two and going A[0, 0, ::]"
            )
            data_A = data_A[0, 0, ::]
        if len(data_B.shape) >= 5:
            print(
                "  [Hyp] B data dims are >= 5, assuming savepoints/rank are the firt two and going B[0, 0, ::]"
            )
            data_B = data_B[0, 0, ::]

        if len(data_A.shape) != len(data_B.shape):
            print(
                f"  [Failed] A is shape {len(data_A.shape)}, B is shape {len(data_B.shape)}: can't reconcile."
            )
            continue

        # - Assume we have 0 == I, 1 == J now
        resolved_I = None
        A_uses_halo = False
        B_uses_halo = False
        if data_A.shape[0] != data_B.shape[0]:
            if data_A.shape[0] < data_B.shape[0]:
                if data_B.shape[0] - 2 * halo != data_A.shape[0]:
                    print(
                        f"  [Failed] B in dim I is too big, even with halo substracted {data_B.shape[0]} (halo: {halo})"
                    )
                else:
                    B_uses_halo = True
                    resolved_I = data_A.shape[0]
            else:
                if data_A.shape[0] - 2 * halo != data_B.shape[0]:
                    print(
                        f"  [Failed] A in dim I is too big, even with halo substracted {data_A.shape[0]} (halo: {halo})"
                    )
                else:
                    A_uses_halo = True
                    resolved_I = data_B.shape[0]
        else:
            resolved_I = data_A.shape[0]

        if resolved_I is None:
            continue

        print(f"  [Hyp] Using {resolved_I} as I dim size")

        resolved_J = None
        if data_A.shape[1] != data_B.shape[1]:
            if data_A.shape[1] < data_B.shape[1]:
                if data_B.shape[1] - 2 * halo != data_A.shape[1]:
                    print(
                        f"  [Failed] B in dim J is too big, even with halo substracted {data_B.shape[1]} (halo: {halo})"
                    )
                else:
                    resolved_J = data_A.shape[1]
            else:
                if data_A.shape[1] - 2 * halo != data_B.shape[1]:
                    print(
                        f"  [Failed] A in dim J is too big, even with halo substracted {data_A.shape[1]} (halo: {halo})"
                    )
                else:
                    resolved_J = data_B.shape[1]
        else:
            resolved_J = data_A.shape[1]

        if resolved_J is None:
            continue

        print(f"  [Hyp] Using {resolved_J} as J dim size")

        # - Assume 2 == K
        if data_A.shape[2] != data_B.shape[2]:
            print(
                f"  [Failed] Can't reconcile K dim: A ({data_A.shape[2]}) != B ({data_B.shape[2]})"
            )
            continue
        resolved_K = data_A.shape[2]

        print(f"  [Hyp] Using {resolved_K} as K dim size")

        # We should now be ready to diff
        if A_uses_halo:
            data_A = data_A[halo:-halo, halo:-halo, ::]
        if B_uses_halo:
            data_B = data_B[halo:-halo, halo:-halo, ::]

        dims = [f"D{i}_{s}" for i, s in enumerate(data_A.shape)]
        absolute_diff = data_A.data - data_B.data
        dataset[name_A] = xr.DataArray(
            absolute_diff,
            dims=dims,
        )

        # ULP diffs
        max_values = np.maximum(
            np.absolute(data_A.data.flatten()), np.absolute(data_B.data.flatten())
        )
        ulp_diff = np.divide(np.abs(absolute_diff.flatten()), np.spacing(max_values))
        ulp_diff = np.sort(ulp_diff)
        dataset[f"ulp_{name_A}"] = xr.DataArray(
            ulp_diff,
            dims=[f"GP_{ulp_diff.shape[0]}"],
        )

        print("  [Success]")

    xr.Dataset(dataset).to_netcdf(f"best_guest_diff_{pathlib.Path(netcdf_A).stem}.nc4")


def entry_point():
    parser = get_parser()
    args = parser.parse_args()
    main(
        netcdf_A=args.netcdf_A,
        netcdf_B=args.netcdf_B,
        name_mapping=args.name_mapping,
        halo=args.halo,
    )


if __name__ == "__main__":
    entry_point()
