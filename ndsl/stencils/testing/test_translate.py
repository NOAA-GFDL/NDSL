# type: ignore
import copy
import os
from typing import Any, Dict, List

import numpy as np
import pytest

import ndsl.dsl.gt4py_utils as gt_utils
from ndsl.comm.communicator import CubedSphereCommunicator, TileCommunicator
from ndsl.comm.mpi import MPI, MPIComm
from ndsl.comm.partitioner import CubedSpherePartitioner, TilePartitioner
from ndsl.dsl.dace.dace_config import DaceConfig
from ndsl.dsl.stencil import CompilationConfig, StencilConfig
from ndsl.quantity import Quantity
from ndsl.restart._legacy_restart import RESTART_PROPERTIES
from ndsl.stencils.testing.savepoint import DataLoader, SavepointCase, dataset_to_dict
from ndsl.testing.comparison import BaseMetric, LegacyMetric, MultiModalFloatMetric
from ndsl.testing.perturbation import perturb


# this only matters for manually-added print statements
np.set_printoptions(threshold=4096)

OUTDIR = "./.translate-outputs"
GPU_MAX_ERR = 1e-10
GPU_NEAR_ZERO = 1e-15
N_THRESHOLD_SAMPLES = int(os.getenv("NDSL_TEST_N_THRESHOLD_SAMPLES", 0))


def platform():
    in_docker = os.environ.get("IN_DOCKER", False)
    return "docker" if in_docker else "metal"


def process_override(threshold_overrides, testobj, test_name, backend):
    override = threshold_overrides.get(test_name, None)
    if override is not None:
        for spec in override:
            if "platform" not in spec:
                spec["platform"] = platform()
            if "backend" not in spec:
                spec["backend"] = backend
        matches = [
            spec
            for spec in override
            if spec["backend"] == backend and spec["platform"] == platform()
        ]
        if len(matches) == 1:
            match = matches[0]
            if "max_error" in match:
                testobj.max_error = float(match["max_error"])
            if "near_zero" in match:
                testobj.near_zero = float(match["near_zero"])
            if "ignore_near_zero_errors" in match:
                parsed_ignore_zero = match["ignore_near_zero_errors"]
                if isinstance(parsed_ignore_zero, list):
                    testobj.ignore_near_zero_errors.update(
                        {field: True for field in match["ignore_near_zero_errors"]}
                    )
                elif isinstance(parsed_ignore_zero, dict):
                    for key in parsed_ignore_zero.keys():
                        testobj.ignore_near_zero_errors[key] = {}
                        testobj.ignore_near_zero_errors[key]["near_zero"] = float(
                            parsed_ignore_zero[key]
                        )
                    if "all_other_near_zero" in match:
                        for key in testobj.out_vars.keys():
                            if key not in testobj.ignore_near_zero_errors:
                                testobj.ignore_near_zero_errors[key] = {}
                                testobj.ignore_near_zero_errors[key][
                                    "near_zero"
                                ] = float(match["all_other_near_zero"])

                else:
                    raise TypeError(
                        "ignore_near_zero_errors is either a list or a dict"
                    )
            if "multimodal" in match:
                parsed_multimodal = match["multimodal"]
                if "absolute_epsilon" in parsed_multimodal:
                    testobj.mmr_absolute_eps = float(parsed_multimodal["absolute_eps"])
                if "relative_fraction" in parsed_multimodal:
                    testobj.mmr_relative_fraction = float(
                        parsed_multimodal["relative_fraction"]
                    )
                if "ulp_threshold" in parsed_multimodal:
                    testobj.mmr_ulp = float(parsed_multimodal["ulp_threshold"])
            if "skip_test" in match:
                testobj.skip_test = bool(match["skip_test"])
        elif len(matches) > 1:
            raise Exception(
                "Misconfigured threshold overrides file, more than 1 specification for "
                + test_name
                + " with backend="
                + backend
                + ", platform="
                + platform()
            )


def get_thresholds(testobj, input_data):
    _get_thresholds(testobj.compute, input_data)


def get_thresholds_parallel(testobj, input_data, communicator):
    def compute(input):
        return testobj.compute_parallel(input, communicator)

    _get_thresholds(compute, input_data)


def _get_thresholds(compute_function, input_data) -> None:
    if N_THRESHOLD_SAMPLES <= 0:
        return
    output_list = []
    for _ in range(N_THRESHOLD_SAMPLES):
        input = copy.deepcopy(input_data)
        perturb(input)
        output_list.append(compute_function(input))

    output_varnames = output_list[0].keys()
    for varname in output_varnames:
        if output_list[0][varname].dtype in (
            np.float64,
            np.int64,
            np.float32,
            np.int32,
        ):
            samples = [out[varname] for out in output_list]
            pointwise_max_abs_errors = np.max(samples, axis=0) - np.min(samples, axis=0)
            max_rel_diff = np.nanmax(
                pointwise_max_abs_errors / np.min(np.abs(samples), axis=0)
            )
            max_abs_diff = np.nanmax(pointwise_max_abs_errors)
            print(
                f"{varname}: max rel diff {max_rel_diff}, max abs diff {max_abs_diff}"
            )


@pytest.mark.sequential
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() > 1,
    reason="Running in parallel with mpi",
)
def test_sequential_savepoint(
    case: SavepointCase,
    backend,
    print_failures,
    failure_stride,
    subtests,
    caplog,
    threshold_overrides,
    multimodal_metric,
    xy_indices=True,
):
    if case.testobj is None:
        pytest.xfail(
            f"No translate object available for savepoint {case.savepoint_name}."
        )
    stencil_config = StencilConfig(
        compilation_config=CompilationConfig(backend=backend),
        dace_config=DaceConfig(
            communicator=None,
            backend=backend,
        ),
    )
    # Reduce error threshold for GPU
    if stencil_config.is_gpu_backend:
        case.testobj.max_error = max(case.testobj.max_error, GPU_MAX_ERR)
        case.testobj.near_zero = max(case.testobj.near_zero, GPU_NEAR_ZERO)
    if threshold_overrides is not None:
        process_override(
            threshold_overrides, case.testobj, case.savepoint_name, backend
        )
    if case.testobj.skip_test:
        return
    if not case.exists:
        pytest.skip(f"Data at rank {case.grid.rank} does not exist.")
    input_data = dataset_to_dict(case.ds_in)
    input_names = (
        case.testobj.serialnames(case.testobj.in_vars["data_vars"])
        + case.testobj.in_vars["parameters"]
    )
    try:
        input_data = {name: input_data[name] for name in input_names}
    except KeyError as e:
        raise KeyError(
            f"Variable {e} was described in the translate test but cannot be found in the NetCDF."
        )
    original_input_data = copy.deepcopy(input_data)
    # give the user a chance to load data from other savepoints to allow
    # for gathering required data from multiple sources (constants, etc.)
    case.testobj.extra_data_load(DataLoader(case.grid.rank, case.data_dir))
    # run python version of functionality
    output = case.testobj.compute(input_data)
    failing_names: List[str] = []
    passing_names: List[str] = []
    all_ref_data = dataset_to_dict(case.ds_out)
    ref_data_out = {}
    results = {}

    # Assign metrics and report on terminal any failures
    for varname in case.testobj.serialnames(case.testobj.out_vars):
        ignore_near_zero = case.testobj.ignore_near_zero_errors.get(varname, False)
        try:
            ref_data = all_ref_data[varname]
        except KeyError:
            raise KeyError(f"Output {varname} couldn't be found in output data.")
        if hasattr(case.testobj, "subset_output"):
            ref_data = case.testobj.subset_output(varname, ref_data)
        with subtests.test(varname=varname):
            failing_names.append(varname)
            output_data = gt_utils.asarray(output[varname])
            if multimodal_metric:
                metric = MultiModalFloatMetric(
                    reference_values=ref_data,
                    computed_values=output_data,
                    absolute_eps_override=case.testobj.mmr_absolute_eps,
                    relative_fraction_override=case.testobj.mmr_relative_fraction,
                    ulp_override=case.testobj.mmr_ulp,
                    sort_report=case.sort_report,
                )
            else:
                metric = LegacyMetric(
                    reference_values=ref_data,
                    computed_values=output_data,
                    eps=case.testobj.max_error,
                    ignore_near_zero_errors=ignore_near_zero,
                    near_zero=case.testobj.near_zero,
                )
            results[varname] = metric
            if not metric.check:
                pytest.fail(str(metric), pytrace=False)
            passing_names.append(failing_names.pop())
        ref_data_out[varname] = [ref_data]

    # Reporting & data save
    if not case.no_report:
        _report_results(case.savepoint_name, case.grid.rank, results)
    if len(failing_names) > 0 and not case.no_report:
        get_thresholds(case.testobj, input_data=original_input_data)
        os.makedirs(OUTDIR, exist_ok=True)
        nc_filename = os.path.join(OUTDIR, f"translate-{case.savepoint_name}.nc")
        input_data_on_host = {}
        for key, _input in input_data.items():
            input_data_on_host[key] = gt_utils.asarray(_input)
        save_netcdf(
            case.testobj,
            [input_data_on_host],
            [output],
            ref_data_out,
            failing_names,
            passing_names,
            nc_filename,
        )
    if failing_names != []:
        pytest.fail(
            f"Only the following variables passed: {passing_names}", pytrace=False
        )
    if len(passing_names) == 0:
        pytest.fail("No tests passed")


def state_from_savepoint(serializer, savepoint, name_to_std_name):
    properties = RESTART_PROPERTIES
    origin = gt_utils.origin
    state = {}
    for name, std_name in name_to_std_name.items():
        array = serializer.read(name, savepoint)
        extent = tuple(np.asarray(array.shape) - 2 * np.asarray(origin))
        state["air_temperature"] = Quantity(
            array,
            dims=reversed(properties["air_temperature"]["dims"]),
            units=properties["air_temperature"]["units"],
            origin=origin,
            extent=extent,
        )
    return state


def get_communicator(comm, layout):
    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    communicator = CubedSphereCommunicator(comm, partitioner)
    return communicator


def get_tile_communicator(comm, layout):
    partitioner = TilePartitioner(layout)
    communicator = TileCommunicator(comm, partitioner)
    return communicator


@pytest.mark.parallel
@pytest.mark.skipif(
    MPI is None or MPI.COMM_WORLD.Get_size() == 1,
    reason="Not running in parallel with mpi",
)
def test_parallel_savepoint(
    case: SavepointCase,
    backend,
    print_failures,
    failure_stride,
    subtests,
    caplog,
    threshold_overrides,
    grid,
    multimodal_metric,
    xy_indices=True,
):
    mpi_comm = MPIComm()
    if mpi_comm.Get_size() % 6 != 0:
        layout = (
            int(mpi_comm.Get_size() ** 0.5),
            int(mpi_comm.Get_size() ** 0.5),
        )
        communicator = get_tile_communicator(mpi_comm, layout)
    else:
        layout = (
            int((mpi_comm.Get_size() // 6) ** 0.5),
            int((mpi_comm.Get_size() // 6) ** 0.5),
        )
        communicator = get_communicator(mpi_comm, layout)
    if case.testobj is None:
        pytest.xfail(
            f"no translate object available for savepoint {case.savepoint_name}"
        )
    stencil_config = StencilConfig(
        compilation_config=CompilationConfig(backend=backend),
        dace_config=DaceConfig(
            communicator=communicator,
            backend=backend,
        ),
    )
    # Increase minimum error threshold for GPU
    if stencil_config.is_gpu_backend:
        case.testobj.max_error = max(case.testobj.max_error, GPU_MAX_ERR)
        case.testobj.near_zero = max(case.testobj.near_zero, GPU_NEAR_ZERO)
    if threshold_overrides is not None:
        process_override(
            threshold_overrides, case.testobj, case.savepoint_name, backend
        )
    if case.testobj.skip_test:
        return
    if (grid == "compute") and not case.testobj.compute_grid_option:
        pytest.xfail(f"Grid compute option not used for test {case.savepoint_name}")
    if not case.exists:
        pytest.skip(f"Data at rank {case.grid.rank} does not exists")
    input_data = dataset_to_dict(case.ds_in)
    # run python version of functionality
    output = case.testobj.compute_parallel(input_data, communicator)
    out_vars = set(case.testobj.outputs.keys())
    out_vars.update(list(case.testobj._base.out_vars.keys()))
    failing_names = []
    passing_names = []
    ref_data: Dict[str, Any] = {}
    all_ref_data = dataset_to_dict(case.ds_out)
    results = {}

    # Assign metrics and report on terminal any failures
    for varname in out_vars:
        ref_data[varname] = []
        new_ref_data = all_ref_data[varname]
        if hasattr(case.testobj, "subset_output"):
            new_ref_data = case.testobj.subset_output(varname, new_ref_data)
        ref_data[varname].append(new_ref_data)
        ignore_near_zero = case.testobj.ignore_near_zero_errors.get(varname, False)
        with subtests.test(varname=varname):
            failing_names.append(varname)
            output_data = gt_utils.asarray(output[varname])
            if multimodal_metric:
                metric = MultiModalFloatMetric(
                    reference_values=ref_data[varname][0],
                    computed_values=output_data,
                    absolute_eps_override=case.testobj.mmr_absolute_eps,
                    relative_fraction_override=case.testobj.mmr_relative_fraction,
                    ulp_override=case.testobj.mmr_ulp,
                    ignore_near_zero_errors=ignore_near_zero,
                    near_zero=case.testobj.near_zero,
                    sort_report=case.sort_report,
                )
            else:
                metric = LegacyMetric(
                    reference_values=ref_data[varname][0],
                    computed_values=output_data,
                    eps=case.testobj.max_error,
                    ignore_near_zero_errors=ignore_near_zero,
                    near_zero=case.testobj.near_zero,
                )
            results[varname] = metric
            if not metric.check:
                pytest.fail(str(metric), pytrace=False)
            passing_names.append(failing_names.pop())

    # Reporting & data save
    _report_results(case.savepoint_name, case.grid.rank, results)
    if len(failing_names) > 0:
        os.makedirs(OUTDIR, exist_ok=True)
        nct_filename = os.path.join(
            OUTDIR, f"translate-{case.savepoint_name}-{case.grid.rank}.nc"
        )
        try:
            input_data_on_host = {}
            for key, _input in input_data.items():
                input_data_on_host[key] = gt_utils.asarray(_input)
            save_netcdf(
                case.testobj,
                inputs_list=[input_data_on_host],
                output_list=[output],
                ref_data=ref_data,
                failing_names=failing_names,
                passing_names=passing_names,
                out_filename=nct_filename,
            )
        except Exception as error:
            print(f"TestParallel SaveNetCDF Error at rank {case.grid.rank}: {error}")
    if failing_names != []:
        pytest.fail(
            f"Only the following variables passed: {passing_names}", pytrace=False
        )
    if len(passing_names) == 0:
        pytest.fail("No tests passed")


def _report_results(
    savepoint_name: str,
    rank: int,
    results: Dict[str, BaseMetric],
) -> None:
    detail_dir = f"{OUTDIR}/details"
    os.makedirs(detail_dir, exist_ok=True)

    # Summary
    with open(f"{OUTDIR}/summary-{savepoint_name}-{rank}.log", "w") as f:
        for varname, metric in results.items():
            f.write(f"{varname}: {metric.one_line_report()}\n")

    # Detailed log
    for varname, metric in results.items():
        log_filename = os.path.join(
            detail_dir, f"{savepoint_name}-{varname}-{rank}.log"
        )
        metric.report(log_filename)


def _save_datatree(
    testobj,
    # first list over rank, second list over savepoint
    inputs_list: List[Dict[str, List[np.ndarray]]],
    output_list: List[Dict[str, List[np.ndarray]]],
    ref_data: Dict[str, List[np.ndarray]],
    names: List[str],
):
    import xarray as xr

    datasets = {}
    indices = np.argsort(names)
    for index in indices:
        data_vars = {}
        varname = names[index]
        # Read in dimensions and attributes
        if hasattr(testobj, "outputs") and testobj.outputs != {}:
            dims = [
                dim_name + f"_{index}" for dim_name in testobj.outputs[varname]["dims"]
            ]
            attrs = {"units": testobj.outputs[varname]["units"]}
        else:
            dims = [
                f"dim_{varname}_{j}" for j in range(len(ref_data[varname][0].shape))
            ]
            attrs = {"units": "unknown"}

        # Try to save inputs
        try:
            data_vars[f"{varname}_input"] = xr.DataArray(
                np.stack([in_data[varname] for in_data in inputs_list]),
                dims=("rank",) + tuple([f"{d}_in" for d in dims]),
                attrs=attrs,
            )
        except KeyError as error:
            print(f"No input data found for {error}")

        # Reference, computed and error computation
        data_vars[f"{varname}_reference"] = xr.DataArray(
            np.stack(ref_data[varname]),
            dims=("rank",) + tuple([f"{d}_out" for d in dims]),
            attrs=attrs,
        )
        data_vars[f"{varname}_computed"] = xr.DataArray(
            np.stack([output[varname] for output in output_list]),
            dims=("rank",) + tuple([f"{d}_out" for d in dims]),
            attrs=attrs,
        )
        absolute_errors = (
            data_vars[f"{varname}_reference"] - data_vars[f"{varname}_computed"]
        )
        data_vars[f"{varname}_absolute_error"] = absolute_errors
        data_vars[f"{varname}_absolute_error"].attrs = attrs
        datasets[varname] = xr.Dataset(data_vars=data_vars)

    return xr.DataTree.from_dict(datasets)


def save_netcdf(
    testobj,
    # first list over rank, second list over savepoint
    inputs_list: List[Dict[str, List[np.ndarray]]],
    output_list: List[Dict[str, List[np.ndarray]]],
    ref_data: Dict[str, List[np.ndarray]],
    failing_names: List[str],
    passing_names: List[str],
    out_filename,
):
    import xarray as xr

    datasets = {}
    datasets["Fail"] = _save_datatree(
        testobj=testobj,
        inputs_list=inputs_list,
        output_list=output_list,
        ref_data=ref_data,
        names=failing_names,
    )
    datasets["Pass"] = _save_datatree(
        testobj=testobj,
        inputs_list=inputs_list,
        output_list=output_list,
        ref_data=ref_data,
        names=passing_names,
    )
    xr.DataTree.from_dict(datasets).to_netcdf(out_filename)
    print(f"File saved to {out_filename}")
