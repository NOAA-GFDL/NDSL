from collections import namedtuple

import pytest

from ndsl import GridSizer, QuantityFactory, SubtileGridSizer
from ndsl.config import Backend
from ndsl.constants import (
    I_DIM,
    I_INTERFACE_DIM,
    J_DIM,
    J_INTERFACE_DIM,
    K_DIM,
    K_INTERFACE_DIM,
    N_HALO_DEFAULT,
)


@pytest.fixture(params=[48, 96])
def nx_tile(request):
    return request.param


@pytest.fixture(params=[48, 96])
def ny_tile(request, fast):
    if fast and request.param == 96:
        pytest.skip("running in fast mode")
    return request.param


@pytest.fixture(params=[60, 80])
def nz(request, fast):
    if fast and request.param == 80:
        pytest.skip("running in fast mode")
    return request.param


@pytest.fixture
def nx(nx_tile, layout):
    return nx_tile / layout[1]


@pytest.fixture
def ny(ny_tile, layout):
    return ny_tile / layout[0]


@pytest.fixture(params=[(1, 1), (3, 3)])
def layout(request):
    return request.param


@pytest.fixture
def extra_dimension_lengths():
    return {}


@pytest.fixture
def namelist(nx_tile, ny_tile, nz, layout):
    return {
        "fv_core_nml": {
            "npx": nx_tile + 1,
            "npy": ny_tile + 1,
            "npz": nz,
            "layout": layout,
        }
    }


@pytest.fixture(params=["from_namelist", "from_tile_params"])
def sizer(
    request, nx_tile, ny_tile, nz, layout, namelist, extra_dimension_lengths
) -> GridSizer:
    backend = Backend.python()  # original utest case
    if request.param == "from_tile_params":
        return SubtileGridSizer.from_tile_params(
            nx_tile=nx_tile,
            ny_tile=ny_tile,
            nz=nz,
            n_halo=N_HALO_DEFAULT,
            layout=layout,
            data_dimensions=extra_dimension_lengths,
            backend=backend,
        )

    if request.param == "from_namelist":
        return SubtileGridSizer.from_namelist(namelist, backend=backend)

    raise NotImplementedError()


@pytest.fixture
def units():
    return "units_placeholder"


@pytest.fixture(params=[float, int])
def dtype(request):
    return request.param


DimCase = namedtuple("DimCase", ["dims", "origin", "extent", "shape"])


@pytest.fixture(
    params=[
        "x_only",
        "x_interface_only",
        "y_only",
        "y_interface_only",
        "z_only",
        "z_interface_only",
        "x_y",
        "z_y_x",
    ]
)
def dim_case(request, nx, ny, nz) -> DimCase:
    if request.param == "x_only":
        return DimCase(
            (I_DIM,),
            (N_HALO_DEFAULT,),
            (nx,),
            (2 * N_HALO_DEFAULT + nx + 1,),
        )

    if request.param == "x_interface_only":
        return DimCase(
            (I_INTERFACE_DIM,),
            (N_HALO_DEFAULT,),
            (nx + 1,),
            (2 * N_HALO_DEFAULT + nx + 1,),
        )

    if request.param == "y_only":
        return DimCase(
            (J_DIM,),
            (N_HALO_DEFAULT,),
            (ny,),
            (2 * N_HALO_DEFAULT + ny + 1,),
        )

    if request.param == "y_interface_only":
        return DimCase(
            (J_INTERFACE_DIM,),
            (N_HALO_DEFAULT,),
            (ny + 1,),
            (2 * N_HALO_DEFAULT + ny + 1,),
        )

    if request.param == "z_only":
        return DimCase((K_DIM,), (0,), (nz,), (nz + 1,))

    if request.param == "z_interface_only":
        return DimCase((K_INTERFACE_DIM,), (0,), (nz + 1,), (nz + 1,))

    if request.param == "x_y":
        return DimCase(
            (
                I_DIM,
                J_DIM,
            ),
            (N_HALO_DEFAULT, N_HALO_DEFAULT),
            (nx, ny),
            (
                2 * N_HALO_DEFAULT + nx + 1,
                2 * N_HALO_DEFAULT + ny + 1,
            ),
        )

    if request.param == "z_y_x":
        return DimCase(
            (
                K_DIM,
                J_DIM,
                I_DIM,
            ),
            (0, N_HALO_DEFAULT, N_HALO_DEFAULT),
            (nz, ny, nx),
            (
                nz + 1,
                2 * N_HALO_DEFAULT + ny + 1,
                2 * N_HALO_DEFAULT + nx + 1,
            ),
        )

    raise NotImplementedError()


@pytest.mark.cpu_only
def test_subtile_dimension_sizer_origin(sizer, dim_case):
    result = sizer.get_origin(dim_case.dims)
    assert result == dim_case.origin


@pytest.mark.cpu_only
def test_subtile_dimension_sizer_extent(sizer, dim_case):
    result = sizer.get_extent(dim_case.dims)
    assert result == dim_case.extent


@pytest.mark.cpu_only
def test_subtile_dimension_sizer_shape(sizer, dim_case):
    result = sizer.get_shape(dim_case.dims)
    assert result == dim_case.shape


def test_allocator_zeros(numpy, sizer, dim_case, units, dtype):
    allocator = QuantityFactory(sizer, backend=Backend.python())
    quantity = allocator.zeros(dim_case.dims, units, dtype=dtype)
    assert quantity.units == units
    assert quantity.dims == dim_case.dims
    assert quantity.origin == dim_case.origin
    assert quantity.extent == dim_case.extent
    assert quantity.data.shape == dim_case.shape
    assert numpy.all(quantity.data == 0)


def test_allocator_ones(numpy, sizer, dim_case, units, dtype):
    allocator = QuantityFactory(sizer, backend=Backend.python())
    quantity = allocator.ones(dim_case.dims, units, dtype=dtype)
    assert quantity.units == units
    assert quantity.dims == dim_case.dims
    assert quantity.origin == dim_case.origin
    assert quantity.extent == dim_case.extent
    assert quantity.data.shape == dim_case.shape
    assert numpy.all(quantity.data == 1)


def test_allocator_empty(sizer, dim_case, units, dtype):
    allocator = QuantityFactory(sizer, backend=Backend.python())
    quantity = allocator.empty(dim_case.dims, units, dtype=dtype)
    assert quantity.units == units
    assert quantity.dims == dim_case.dims
    assert quantity.origin == dim_case.origin
    assert quantity.extent == dim_case.extent
    assert quantity.data.shape == dim_case.shape


def test_allocator_data_dimensions_operations(sizer):
    quantity_factory = QuantityFactory(sizer, backend=Backend.python())
    quantity_factory.add_data_dimensions({"D0": 11})
    assert "D0" in quantity_factory.sizer.data_dimensions.keys()
    assert quantity_factory.sizer.data_dimensions["D0"] == 11
    quantity_factory.update_data_dimensions({"D0": 22})
    assert quantity_factory.sizer.data_dimensions["D0"] == 22
    with pytest.raises(
        ValueError,
        match="Use `update_data_dimensions` if you meant to update the length.",
    ):
        quantity_factory.add_data_dimensions({"D0": 33})


def test_pad_non_interface_dimensions():
    nx = 10
    ny = 20
    nz = 30
    dd = 40
    layout_xy = 2
    padded_grid_sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=nz,
        n_halo=0,
        layout=(layout_xy, layout_xy),
        data_dimensions={"some_dim": dd},
        backend=Backend.python(),  # original utest case
    )
    padded_shape = padded_grid_sizer.get_shape([I_DIM, J_DIM, K_DIM, "some_dim"])
    assert padded_shape[0] == nx // layout_xy + 1
    assert padded_shape[1] == ny // layout_xy + 1
    assert padded_shape[2] == nz + 1
    assert padded_shape[3] == dd

    non_padded_grid_sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=nz,
        n_halo=0,
        layout=(layout_xy, layout_xy),
        data_dimensions={"some_dim": dd},
        backend=Backend("st:dace:cpu:KJI"),  # Fortran-friendly backend
    )
    non_padded_shape = non_padded_grid_sizer.get_shape(
        [I_DIM, J_DIM, K_DIM, "some_dim"]
    )
    assert non_padded_shape[0] == nx // layout_xy
    assert non_padded_shape[1] == ny // layout_xy
    assert non_padded_shape[2] == nz
    assert non_padded_shape[3] == dd
