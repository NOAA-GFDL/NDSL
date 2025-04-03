import os
import re
from typing import Optional, Tuple

import f90nml
import pytest
import xarray as xr
import yaml

from ndsl import CompilationConfig, StencilConfig, StencilFactory
from ndsl.comm.communicator import (
    Communicator,
    CubedSphereCommunicator,
    TileCommunicator,
)
from ndsl.comm.mpi import MPI, MPIComm
from ndsl.comm.partitioner import CubedSpherePartitioner, TilePartitioner
from ndsl.dsl.dace.dace_config import DaceConfig
from ndsl.namelist import Namelist
from ndsl.stencils.testing.grid import Grid  # type: ignore
from ndsl.stencils.testing.parallel_translate import ParallelTranslate
from ndsl.stencils.testing.savepoint import SavepointCase, dataset_to_dict
from ndsl.stencils.testing.translate import TranslateGrid


def pytest_addoption(parser):
    """Option for the Translate Test system

    See -h or inline help for details.
    """
    parser.addoption(
        "--backend",
        action="store",
        default="numpy",
        help="Backend to execute the test with, can only be one.",
    )
    parser.addoption(
        "--which_modules",
        action="store",
        help="Whitelist of modules to run. Only the part after Translate, e.g. in TranslateXYZ it'd be XYZ",
    )
    parser.addoption(
        "--skip_modules",
        action="store",
        help="Blacklist of modules to not run. Only the part after Translate, e.g. in TranslateXYZ it'd be XYZ",
    )
    parser.addoption(
        "--which_rank", action="store", help="Restrict test to a single rank"
    )
    parser.addoption(
        "--which_savepoint", action="store", help="Restrict test to a single savepoint"
    )
    parser.addoption(
        "--data_path",
        action="store",
        default="./",
        help="Path of Netcdf input and outputs. Naming pattern needs to be XYZ-In and XYZ-Out for a test class named TranslateXYZ",
    )
    parser.addoption(
        "--threshold_overrides_file",
        action="store",
        default=None,
        help="Path to a yaml overriding the default error threshold for a custom value.",
    )
    parser.addoption(
        "--print_failures",
        action="store_true",
        help="Print the failures detail. Default to True.",
    )
    parser.addoption(
        "--failure_stride",
        action="store",
        default=1,
        help="How many indices of failures to print from worst to best. Default to 1.",
    )
    parser.addoption(
        "--grid",
        action="store",
        default="file",
        help='Grid loading mode. "file" looks for "Grid-Info.nc", "compute" does the same but recomputes MetricTerms, "default" creates a simple grid with no metrics terms. Default to "file".',
    )
    parser.addoption(
        "--topology",
        action="store",
        default="cubed-sphere",
        help='Topology of the grid. "cubed-sphere" means a 6-faced grid, "doubly-periodic" means a 1 tile grid. Default to "cubed-sphere".',
    )
    parser.addoption(
        "--multimodal_metric",
        action="store_true",
        default=False,
        help="Use the multi-modal float metric. Default to False.",
    )
    parser.addoption(
        "--sort_report",
        action="store",
        default="ulp",
        help='Sort the report by "index" (ascending) or along the metric: "ulp", "absolute", "relative" (descending). Default to "ulp"',
    )
    parser.addoption(
        "--no_report",
        action="store_true",
        default=False,
        help="Do not generate logging report or NetCDF in .translate-errors",
    )


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "sequential(name): mark test as running sequentially on ranks"
    )
    config.addinivalue_line(
        "markers", "parallel(name): mark test as running in parallel across ranks"
    )
    config.addinivalue_line(
        "markers",
        "mock_parallel(name): mark test as running in mock parallel across ranks",
    )


@pytest.fixture()
def data_path(pytestconfig):
    return data_path_and_namelist_filename_from_config(pytestconfig)


def data_path_and_namelist_filename_from_config(config) -> Tuple[str, str]:
    data_path = config.getoption("data_path")
    namelist_filename = os.path.join(data_path, "input.nml")
    return data_path, namelist_filename


@pytest.fixture
def threshold_overrides(pytestconfig):
    return thresholds_from_file(pytestconfig)


def thresholds_from_file(config):
    thresholds_file = config.getoption("threshold_overrides_file")
    if thresholds_file is None:
        return None
    return yaml.safe_load(open(thresholds_file, "r"))


def get_test_class(test_name):
    translate_class_name = f"Translate{test_name.replace('-', '_')}"
    try:
        return_class = getattr(translate, translate_class_name)  # noqa: F821
    except AttributeError as err:
        if translate_class_name in err.args[0]:
            return_class = None
        else:
            raise err
    return return_class


def is_parallel_test(test_name):
    test_class = get_test_class(test_name)
    if test_class is None:
        return False
    else:
        return issubclass(test_class, ParallelTranslate)


def get_test_class_instance(test_name, grid, namelist, stencil_factory):
    translate_class = get_test_class(test_name)
    if translate_class is None:
        return None
    else:
        return translate_class(grid, namelist, stencil_factory)


def get_all_savepoint_names(metafunc, data_path):
    only_names = metafunc.config.getoption("which_modules")
    if only_names is None:
        savepoint_names = [
            fname[:-3] for fname in os.listdir(data_path) if re.match(r".*\.nc", fname)
        ]
        savepoint_names = [s[:-3] for s in savepoint_names if s.endswith("-In")]
    else:
        savepoint_names = set(only_names.split(","))
        savepoint_names.discard("")
    skip_names = metafunc.config.getoption("skip_modules")
    if skip_names is not None:
        savepoint_names.difference_update(skip_names.split(","))
    return savepoint_names


def get_sequential_savepoint_names(metafunc, data_path):
    all_names = get_all_savepoint_names(metafunc, data_path)
    sequential_names = []
    for name in all_names:
        if not is_parallel_test(name):
            sequential_names.append(name)
    return sequential_names


def get_parallel_savepoint_names(metafunc, data_path):
    all_names = get_all_savepoint_names(metafunc, data_path)
    parallel_names = []
    for name in all_names:
        if is_parallel_test(name):
            parallel_names.append(name)
    return parallel_names


def get_ranks(metafunc, layout):
    only_rank = metafunc.config.getoption("which_rank")
    topology = metafunc.config.getoption("topology")
    if only_rank is None:
        if topology == "doubly-periodic":
            total_ranks = layout[0] * layout[1]
        elif topology == "cubed-sphere":
            total_ranks = 6 * layout[0] * layout[1]
        else:
            raise NotImplementedError(f"Topology {topology} is unknown.")
        return range(total_ranks)
    else:
        return [int(only_rank)]


def get_savepoint_restriction(metafunc):
    svpt = metafunc.config.getoption("which_savepoint")
    return int(svpt) if svpt else None


def get_namelist(namelist_filename):
    return Namelist.from_f90nml(f90nml.read(namelist_filename))


def get_config(backend: str, communicator: Optional[Communicator]):
    stencil_config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend, rebuild=False, validate_args=True
        ),
        dace_config=DaceConfig(
            communicator=communicator,
            backend=backend,
        ),
    )
    return stencil_config


def sequential_savepoint_cases(metafunc, data_path, namelist_filename, *, backend: str):
    savepoint_names = get_sequential_savepoint_names(metafunc, data_path)
    namelist = get_namelist(namelist_filename)
    stencil_config = get_config(backend, None)
    ranks = get_ranks(metafunc, namelist.layout)
    savepoint_to_replay = get_savepoint_restriction(metafunc)
    grid_mode = metafunc.config.getoption("grid")
    topology_mode = metafunc.config.getoption("topology")
    sort_report = metafunc.config.getoption("sort_report")
    no_report = metafunc.config.getoption("no_report")
    return _savepoint_cases(
        savepoint_names,
        ranks,
        savepoint_to_replay,
        stencil_config,
        namelist,
        backend,
        data_path,
        grid_mode,
        topology_mode,
        sort_report=sort_report,
        no_report=no_report,
    )


def _savepoint_cases(
    savepoint_names,
    ranks,
    savepoint_to_replay,
    stencil_config,
    namelist: Namelist,
    backend: str,
    data_path: str,
    grid_mode: str,
    topology_mode: bool,
    sort_report: str,
    no_report: bool,
):
    return_list = []
    for rank in ranks:
        if grid_mode == "default":
            grid = Grid._make(
                namelist.npx,
                namelist.npy,
                namelist.npz,
                namelist.layout,
                rank,
                backend,
            )
        elif grid_mode == "file" or grid_mode == "compute":
            ds_grid: xr.Dataset = xr.open_dataset(
                os.path.join(data_path, "Grid-Info.nc")
            ).isel(savepoint=0)
            grid = TranslateGrid(
                dataset_to_dict(ds_grid.isel(rank=rank)),
                rank=rank,
                layout=namelist.layout,
                backend=backend,
            ).python_grid()
            if grid_mode == "compute":
                compute_grid_data(
                    grid, namelist, backend, namelist.layout, topology_mode
                )
        else:
            raise NotImplementedError(f"Grid mode {grid_mode} is unknown.")

        stencil_factory = StencilFactory(
            config=stencil_config,
            grid_indexing=grid.grid_indexing,
        )
        for test_name in sorted(list(savepoint_names)):
            testobj = get_test_class_instance(
                test_name, grid, namelist, stencil_factory
            )
            n_calls = xr.open_dataset(
                os.path.join(data_path, f"{test_name}-In.nc")
            ).dims["savepoint"]
            if savepoint_to_replay is not None:
                savepoint_iterator = range(savepoint_to_replay, savepoint_to_replay + 1)
            else:
                savepoint_iterator = range(n_calls)
            for i_call in savepoint_iterator:
                return_list.append(
                    SavepointCase(
                        savepoint_name=test_name,
                        data_dir=data_path,
                        i_call=i_call,
                        testobj=testobj,
                        grid=grid,
                        sort_report=sort_report,
                        no_report=no_report,
                    )
                )
    return return_list


def compute_grid_data(grid, namelist, backend, layout, topology_mode):
    grid.make_grid_data(
        npx=namelist.npx,
        npy=namelist.npy,
        npz=namelist.npz,
        communicator=get_communicator(MPIComm(), layout, topology_mode),
        backend=backend,
    )


def parallel_savepoint_cases(
    metafunc, data_path, namelist_filename, mpi_rank, *, backend: str, comm
):
    namelist = get_namelist(namelist_filename)
    topology_mode = metafunc.config.getoption("topology")
    sort_report = metafunc.config.getoption("sort_report")
    no_report = metafunc.config.getoption("no_report")
    communicator = get_communicator(comm, namelist.layout, topology_mode)
    stencil_config = get_config(backend, communicator)
    savepoint_names = get_parallel_savepoint_names(metafunc, data_path)
    grid_mode = metafunc.config.getoption("grid")
    savepoint_to_replay = get_savepoint_restriction(metafunc)
    return _savepoint_cases(
        savepoint_names,
        [mpi_rank],
        savepoint_to_replay,
        stencil_config,
        namelist,
        backend,
        data_path,
        grid_mode,
        topology_mode,
        sort_report=sort_report,
        no_report=no_report,
    )


def pytest_generate_tests(metafunc):
    backend = metafunc.config.getoption("backend")
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
        if metafunc.function.__name__ == "test_parallel_savepoint":
            generate_parallel_stencil_tests(metafunc, backend=backend)
    elif metafunc.function.__name__ == "test_sequential_savepoint":
        generate_sequential_stencil_tests(metafunc, backend=backend)


def generate_sequential_stencil_tests(metafunc, *, backend: str):
    data_path, namelist_filename = data_path_and_namelist_filename_from_config(
        metafunc.config
    )
    savepoint_cases = sequential_savepoint_cases(
        metafunc, data_path, namelist_filename, backend=backend
    )
    metafunc.parametrize(
        "case", savepoint_cases, ids=[str(item) for item in savepoint_cases]
    )


def generate_parallel_stencil_tests(metafunc, *, backend: str):
    data_path, namelist_filename = data_path_and_namelist_filename_from_config(
        metafunc.config
    )
    # get MPI environment
    comm = MPIComm()
    savepoint_cases = parallel_savepoint_cases(
        metafunc,
        data_path,
        namelist_filename,
        comm.Get_rank(),
        backend=backend,
        comm=comm,
    )
    metafunc.parametrize(
        "case", savepoint_cases, ids=[str(item) for item in savepoint_cases]
    )


def get_communicator(comm, layout, topology_mode):
    if (comm.Get_size() > 1) and (topology_mode == "cubed-sphere"):
        partitioner = CubedSpherePartitioner(TilePartitioner(layout))
        communicator = CubedSphereCommunicator(comm, partitioner)
    else:
        partitioner = TilePartitioner(layout)
        communicator = TileCommunicator(comm, partitioner)
    return communicator


@pytest.fixture()
def print_failures(pytestconfig):
    return pytestconfig.getoption("print_failures")


@pytest.fixture()
def failure_stride(pytestconfig):
    return int(pytestconfig.getoption("failure_stride"))


@pytest.fixture()
def multimodal_metric(pytestconfig):
    return bool(pytestconfig.getoption("multimodal_metric"))


@pytest.fixture()
def grid(pytestconfig):
    return pytestconfig.getoption("grid")


@pytest.fixture()
def topology_mode(pytestconfig):
    return pytestconfig.getoption("topology_mode")


@pytest.fixture()
def backend(pytestconfig):
    return pytestconfig.getoption("backend")
