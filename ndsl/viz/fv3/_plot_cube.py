from __future__ import annotations

import os
import warnings
from functools import partial

import cartopy
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from matplotlib import pyplot as plt

from ._constants import (
    COORD_X_CENTER,
    COORD_X_OUTER,
    COORD_Y_CENTER,
    COORD_Y_OUTER,
    VAR_LAT_CENTER,
    VAR_LAT_OUTER,
    VAR_LON_CENTER,
    VAR_LON_OUTER,
)
from ._masking import _mask_antimeridian_quads
from ._plot_helpers import (
    _align_grid_var_dims,
    _align_plot_var_dims,
    _get_var_label,
    infer_cmap_params,
)
from .grid_metadata import GridMetadata, GridMetadataFV3, GridMetadataScream


if os.getenv("CARTOPY_EXTERNAL_DOWNLOADER") != "natural_earth":
    # workaround to host our own global-scale coastline shapefile instead
    # of unreliable cartopy source
    cartopy.config["downloaders"][("shapefiles", "natural_earth")].url_template = (
        "https://raw.githubusercontent.com/ai2cm/"
        "vcm-ml-example-data/main/fv3net/fv3viz/coastline_shapefiles/"
        "{resolution}_{category}/ne_{resolution}_{name}.zip"
    )

WRAPPER_GRID_METADATA = GridMetadataFV3(
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
    "tile",
    VAR_LON_CENTER,
    VAR_LON_OUTER,
    VAR_LAT_CENTER,
    VAR_LAT_OUTER,
)


def plot_cube(
    ds: xr.Dataset,
    var_name: str,
    grid_metadata: GridMetadata = WRAPPER_GRID_METADATA,
    plotting_function: str = "pcolormesh",
    ax: plt.axes = None,
    row: str = None,
    col: str = None,
    col_wrap: int = None,
    projection: ccrs.Projection = None,
    colorbar: bool = True,
    cmap_percentiles_lim: bool = True,
    cbar_label: str = None,
    coastlines: bool = True,
    coastlines_kwargs: dict = None,
    **kwargs,
):
    """Plots an xr.DataArray containing tiled cubed sphere gridded data
    onto a global map projection, with optional faceting of additional dims

    Args:
        ds:
            Dataset containing variable to plotted, along with the grid
            variables defining cell center latitudes and longitudes and the
            cell bounds latitudes and longitudes, which must share common
            dimension names
        var_name:
            name of the data variable in `ds` to be plotted
        grid_metadata:
            a vcm.cubedsphere.GridMetadata data structure that
            defines the names of plot and grid variable dimensions and the names
            of the grid variables themselves; defaults to those used by the
            fv3gfs Python wrapper (i.e., 'x', 'y', 'x_interface', 'y_interface' and
            'lat', 'lon', 'latb', 'lonb')
        plotting_function:
            Name of matplotlib 2-d plotting function. Available
            options are "pcolormesh", "contour", and "contourf". Defaults to
            "pcolormesh".
        ax:
            Axes onto which the map should be plotted; must be created with
            a cartopy projection argument. If not supplied, axes are generated
            with a projection. If ax is suppled, faceting is disabled.
        row:
            Name of diemnsion to be faceted along subplot rows. Must not be a
            tile, lat, or lon dimension.  Defaults to no row facets.
        col:
            Name of diemnsion to be faceted along subplot columns. Must not be
            a tile, lat, or lon dimension. Defaults to no column facets.
        col_wrap:
            If only one of `col`, `row` is specified, number of columns to plot
            before wrapping onto next row. Defaults to None, i.e. no limit.
        projection:
            Cartopy projection object to be used in creating axes. Ignored
            if cartopy geo-axes are supplied.  Defaults to Robinson projection.
        colorbar:
            Flag for whether to plot a colorbar. Defaults to True.
        cmap_percentiles_lim:
            If False, use the absolute min/max to set color limits.
            If True, use 2/98 percentile values.
        cbar_label:
            If provided, use this as the color bar label.
        coastlines:
            Whether to plot coastlines on map. Default True.
        coastlines_kwargs:
            Dict of arguments to be passed to cartopy axes's
            `coastline` function if `coastlines` flag is set to True.
        **kwargs: Additional keyword arguments to be passed to the plotting function.

    Returns:
        figure (plt.Figure):
            matplotlib figure object onto which axes grid is created
        axes (np.ndarray):
            Array of `plt.axes` objects assocated with map subplots if faceting;
            otherwise array containing single axes object.
        handles (list):
            List or nested list of matplotlib object handles associated with
            map subplots if faceting; otherwise list of single object handle.
        cbar (plt.colorbar):
            object handle associated with figure, if `colorbar`
            arg is True, else None.
        facet_grid (xarray.plot.facetgrid):
            xarray plotting facetgrid for multi-axes case. In single-axes case,
            retunrs None.

    Example:
        # plot diag winds at two times
        fig, axes, hs, cbar, facet_grid = plot_cube(
            diag_ds.isel(time = slice(2, 4)),
            'VGRD850',
            plotting_function = "contourf",
            col = "time",
            coastlines = True,
            colorbar = True,
            vmin = -20,
            vmax = 20
        )
    """

    mappable_ds = _mappable_var(ds, var_name, grid_metadata)
    array = mappable_ds[var_name].values

    kwargs["vmin"], kwargs["vmax"], kwargs["cmap"] = infer_cmap_params(
        array,
        vmin=kwargs.get("vmin"),
        vmax=kwargs.get("vmax"),
        cmap=kwargs.get("cmap"),
        robust=cmap_percentiles_lim,
    )
    if isinstance(grid_metadata, GridMetadataFV3):
        _plot_func_short = partial(
            _plot_cube_axes,
            lat=mappable_ds.lat.values,
            lon=mappable_ds.lon.values,
            latb=mappable_ds.latb.values,
            lonb=mappable_ds.lonb.values,
            plotting_function=plotting_function,
            **kwargs,
        )
    elif isinstance(grid_metadata, GridMetadataScream):
        _plot_func_short = partial(
            _plot_scream_axes,
            lat=mappable_ds.lat.values,
            lon=mappable_ds.lon.values,
            plotting_function=plotting_function,
            **kwargs,
        )
    else:
        assert ValueError(
            f"grid_metadata needs to be either GridMetadataFV3 or GridMetadataScream, \
              but got {type(grid_metadata)}"
        )

    projection = ccrs.Robinson() if not projection else projection

    if ax is None and (row or col):
        # facets
        facet_grid = xr.plot.FacetGrid(
            data=mappable_ds,
            row=row,
            col=col,
            col_wrap=col_wrap,
            subplot_kws={"projection": projection},
        )
        facet_grid = facet_grid.map(_plot_func_short, var_name)
        fig = facet_grid.fig
        axes = facet_grid.axes
        handles = facet_grid._mappables
    else:
        # single axes
        if ax is None:
            fig, ax = plt.subplots(1, 1, subplot_kw={"projection": projection})
        else:
            fig = ax.figure
        handle = _plot_func_short(array, ax=ax)
        axes = np.array(ax)
        handles = [handle]
        facet_grid = None

    if coastlines:
        coastlines_kwargs = dict() if not coastlines_kwargs else coastlines_kwargs
        [ax.coastlines(**coastlines_kwargs) for ax in axes.flatten()]

    if colorbar:
        if row or col:
            fig.subplots_adjust(
                bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02
            )
            cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        else:
            fig.subplots_adjust(wspace=0.25)
            cb_ax = ax.inset_axes([1.05, 0, 0.02, 1])
        cbar = plt.colorbar(handles[0], cax=cb_ax, extend="both")
        cbar.set_label(cbar_label or _get_var_label(ds[var_name].attrs, var_name))
    else:
        cbar = None

    return fig, axes, handles, cbar, facet_grid


def _mappable_var(
    ds: xr.Dataset,
    var_name: str,
    grid_metadata: GridMetadata = WRAPPER_GRID_METADATA,
):
    """Converts a dataset into a format for plotting across cubed-sphere tiles by
    checking and ordering its grid variable and plotting variable dimensions

    Args:
        ds:
            Dataset containing the variable to be plotted, along with grid variables.
        var_name:
            Name of variable to be plotted.
        grid_metadata:
            vcm.cubedsphere.GridMetadata object describing dim
            names and grid variable names
    Returns:
        ds (xr.Dataset): Dataset containing variable to be plotted as well as grid
            variables, all of whose dimensions are ordered for plotting.
    """
    mappable_ds = xr.Dataset()
    for var, dims in grid_metadata.coord_vars.items():
        mappable_ds[var] = _align_grid_var_dims(ds[var], required_dims=dims)
    if isinstance(grid_metadata, GridMetadataFV3):
        var_da = _align_plot_var_dims(ds[var_name], grid_metadata.y, grid_metadata.x)
        return mappable_ds.merge(var_da)
    elif isinstance(grid_metadata, GridMetadataScream):
        return mappable_ds.merge(ds[var_name])


def pcolormesh_cube(
    lat: np.ndarray, lon: np.ndarray, array: np.ndarray, ax: plt.axes = None, **kwargs
):
    """Plots tiled cubed sphere. This function applies nan to gridcells which cross
    the antimeridian, and then iteratively plots rectangles of array which avoid nan
    gridcells. This is done to avoid artifacts when plotting gridlines with the
    `edgecolor` argument. In comparison to :py:func:`plot_cube`, this function takes
    np.ndarrays of the lat and lon cell corners and the variable to be plotted
    at cell centers, and makes only one plot on an optionally specified axes object.

    Args:
        lat:
            Array of latitudes with dimensions (tile, ny + 1, nx + 1).
            Should be given at cell corners.
        lon:
            Array of longitudes with dimensions (tile, ny + 1, nx + 1).
            Should be given at cell corners.
        array:
            Array of variables values at cell centers, of dimensions (tile, ny, nx)
        ax:
            Matplotlib geoaxes object onto which plotting function will be
            called. Default None uses current axes.
        **kwargs:
            Keyword arguments to be passed to plotting function.

    Returns:
        p_handle (obj):
            matplotlib object handle associated with a segment of the map subplot
    """
    all_handles = _pcolormesh_cube_all_handles(lat, lon, array, ax=ax, **kwargs)
    return all_handles[-1]


def _pcolormesh_cube_all_handles(
    lat: np.ndarray, lon: np.ndarray, array: np.ndarray, ax: plt.axes = None, **kwargs
):
    if lat.shape != lon.shape:
        raise ValueError("lat and lon should have the same shape")
    if ax is None:
        ax = plt.gca()
    central_longitude = ax.projection.proj4_params["lon_0"]
    array = np.where(
        _mask_antimeridian_quads(lon.T, central_longitude), array.T, np.nan
    ).T
    # oddly a PlateCarree transform seems to be needed here even for non-PlateCarree
    # projections?? very puzzling, but it seems to be the case.
    kwargs["transform"] = kwargs.get("transform", ccrs.PlateCarree())
    kwargs["vmin"] = kwargs.get("vmin", np.nanmin(array))
    kwargs["vmax"] = kwargs.get("vmax", np.nanmax(array))

    def plot(x, y, array):
        return ax.pcolormesh(x, y, array, **kwargs)

    handles = _apply_to_non_non_nan_segments(
        plot, lat, center_longitudes(lon, central_longitude), array
    )
    return handles


class UpdateablePColormesh:
    def __init__(self, lat, lon, array: np.ndarray, ax: plt.axes = None, **kwargs):
        self.handles = _pcolormesh_cube_all_handles(lat, lon, array, ax=ax, **kwargs)
        plt.colorbar(self.handles[-1], ax=ax)
        self.lat = lat
        self.lon = lon
        self.ax = ax

    def update(self, array):
        central_longitude = self.ax.projection.proj4_params["lon_0"]
        array = np.where(
            _mask_antimeridian_quads(self.lon.T, central_longitude), array.T, np.nan
        ).T

        iter_handles = iter(self.handles)

        def update_handle(x, y, array):
            handle = next(iter_handles)
            handle.set_array(array.ravel())

        _apply_to_non_non_nan_segments(update_handle, self.lat, self.lon, array)


def _apply_to_non_non_nan_segments(func, lat, lon, array):
    """
    Applies func to disjoint rectangular segments of array covering all non-nan values.

    Args:
        func:
            Function to be applied to non-nan segments of array.
        lat:
            Array of latitudes with dimensions (tile, ny + 1, nx + 1).
            Should be given at cell corners.
        lon:
            Array of longitudes with dimensions (tile, ny + 1, nx + 1).
            Should be given at cell corners.
        array:
            Array of variables values at cell centers, of dimensions (tile, ny, nx)

    Returns:
        list of return values of func
    """
    all_handles = []
    for tile in range(array.shape[0]):
        x = lon[tile, :, :]
        y = lat[tile, :, :]
        for x_plot, y_plot, array_plot in _segment_plot_inputs(x, y, array[tile, :, :]):
            all_handles.append(func(x_plot, y_plot, array_plot))
    return all_handles


def _segment_plot_inputs(x, y, masked_array):
    """Takes in two arrays at corners of grid cells and an array at grid cell centers
    which may contain NaNs. Yields 3-tuples of rectangular segments of
    these arrays which cover all non-nan points without duplicates, and don't contain
    NaNs.
    """
    is_nan = np.isnan(masked_array)
    if np.sum(is_nan) == 0:  # contiguous section, just plot it
        if np.product(masked_array.shape) > 0:
            yield (x, y, masked_array)
    else:
        x_nans = np.sum(is_nan, axis=1) / is_nan.shape[1]
        y_nans = np.sum(is_nan, axis=0) / is_nan.shape[0]
        if x_nans.max() >= y_nans.max():  # most nan-y line is in first dimension
            i_split = x_nans.argmax()
            if x_nans[i_split] == 1.0:  # split cleanly along line
                yield from _segment_plot_inputs(
                    x[: i_split + 1, :],
                    y[: i_split + 1, :],
                    masked_array[:i_split, :],
                )
                yield from _segment_plot_inputs(
                    x[i_split + 1 :, :],
                    y[i_split + 1 :, :],
                    masked_array[i_split + 1 :, :],
                )
            else:
                # split to create segments of complete nans
                # which subsequent recursive calls will split on and remove
                i_start = 0
                i_end = 1
                while i_end < is_nan.shape[1]:
                    while (
                        i_end < is_nan.shape[1]
                        and is_nan[i_split, i_start] == is_nan[i_split, i_end]
                    ):
                        i_end += 1
                    # we have a largest-possible contiguous segment of nans/not nans
                    yield from _segment_plot_inputs(
                        x[:, i_start : i_end + 1],
                        y[:, i_start : i_end + 1],
                        masked_array[:, i_start:i_end],
                    )
                    i_start = i_end  # start the next segment
        else:
            # put most nan-y line in first dimension
            # so the first part of this if block catches it
            yield from _segment_plot_inputs(
                x.T,
                y.T,
                masked_array.T,
            )


def center_longitudes(lon_array, central_longitude):
    return np.where(
        lon_array < (central_longitude + 180.0) % 360.0,
        lon_array,
        lon_array - 360.0,
    )


def _validate_cube_shape(lat_shape, lon_shape, latb_shape, lonb_shape, array_shape):
    if (lon_shape[-1] != 6) or (lat_shape[-1] != 6) or (array_shape[-1] != 6):
        raise ValueError(
            """Last axis of each array must have six elements for
            cubed-sphere tiles."""
        )

    if (
        (lon_shape[0] != lat_shape[0])
        or (lat_shape[0] != array_shape[0])
        or (lon_shape[1] != lat_shape[1])
        or (lat_shape[1] != array_shape[1])
    ):
        raise ValueError(
            """Horizontal axis lengths of lat and lon must be equal to
            those of array."""
        )

    if (len(lonb_shape) != 3) or (len(latb_shape) != 3) or (len(array_shape) != 3):
        raise ValueError("Lonb, latb, and data_var each must be 3-dimensional.")

    if (lonb_shape[-1] != 6) or (latb_shape[-1] != 6) or (array_shape[-1] != 6):
        raise ValueError(
            "Tile axis of each array must have six elements for cubed-sphere tiles."
        )

    if (
        (lonb_shape[0] != latb_shape[0])
        or (latb_shape[0] != (array_shape[0] + 1))
        or (lonb_shape[1] != latb_shape[1])
        or (latb_shape[1] != (array_shape[1] + 1))
    ):
        raise ValueError(
            """Horizontal axis lengths of latb and lonb
            must be one greater than those of array."""
        )

    if (len(lon_shape) != 3) or (len(lat_shape) != 3) or (len(array_shape) != 3):
        raise ValueError("Lon, lat, and data_var each must be 3-dimensional.")


def _plot_cube_axes(
    array: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    latb: np.ndarray,
    lonb: np.ndarray,
    plotting_function: str,
    ax: plt.axes = None,
    **kwargs,
):
    """Plots tiled cubed sphere for a given subplot axis,
        using np.ndarrays for all data

    Args:
        array:
            Array of variables values at cell centers, of dimensions (npy, npx,
            tile)
        lat:
            Array of latitudes of cell centers, of dimensions (npy, npx, tile)
        lon:
            Array of longitudes of cell centers, of dimensions (npy, npx, tile)
        latb:
            Array of latitudes of cell edges, of dimensions (npy + 1, npx + 1,
            tile)
        lonb:
            Array of longitudes of cell edges, of dimensions (npy + 1, npx + 1,
            tile)
        plotting_function:
            Name of matplotlib 2-d plotting function. Available options
            are "pcolormesh", "contour", and "contourf".
        ax:
            Matplotlib geoaxes object onto which plotting function will be
            called. Default None uses current axes.
        **kwargs:
            Keyword arguments to be passed to plotting function.

    Returns:
        p_handle (obj):
            matplotlib object handle associated with map subplot
    """
    _validate_cube_shape(lon.shape, lat.shape, lonb.shape, latb.shape, array.shape)

    if ax is None:
        ax = plt.gca()

    if plotting_function in ["pcolormesh", "contour", "contourf"]:
        _plotting_function = getattr(ax, plotting_function)
    else:
        raise ValueError(
            """Plotting functions only include pcolormesh, contour,
            and contourf."""
        )

    if "vmin" not in kwargs:
        kwargs["vmin"] = np.nanmin(array)

    if "vmax" not in kwargs:
        kwargs["vmax"] = np.nanmax(array)

    if np.isnan(kwargs["vmin"]):
        kwargs["vmin"] = -0.1
    if np.isnan(kwargs["vmax"]):
        kwargs["vmax"] = 0.1

    if plotting_function != "pcolormesh":
        if "levels" not in kwargs:
            kwargs["n_levels"] = 11 if "n_levels" not in kwargs else kwargs["n_levels"]
            kwargs["levels"] = np.linspace(
                kwargs["vmin"], kwargs["vmax"], kwargs["n_levels"]
            )

    central_longitude = ax.projection.proj4_params["lon_0"]

    masked_array = np.where(
        _mask_antimeridian_quads(lonb, central_longitude), array, np.nan
    )

    for tile in range(6):
        if plotting_function == "pcolormesh":
            x = lonb[:, :, tile]
            y = latb[:, :, tile]
        else:
            # contouring
            x = center_longitudes(lon[:, :, tile], central_longitude)
            y = lat[:, :, tile]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_handle = _plotting_function(
                x, y, masked_array[:, :, tile], transform=ccrs.PlateCarree(), **kwargs
            )

    ax.set_global()

    return p_handle


def _plot_scream_axes(
    array: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    plotting_function: str,
    ax: plt.axes = None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    if plotting_function in ["pcolormesh", "contour", "contourf"]:
        mapping = {
            "pcolormesh": "tripcolor",
            "contour": "tricontour",
            "contourf": "tricontourf",
        }
        _plotting_function = getattr(ax, mapping[plotting_function])
    else:
        raise ValueError(
            """Plotting functions only include pcolormesh, contour,
            and contourf."""
        )
    if "vmin" not in kwargs:
        kwargs["vmin"] = np.nanmin(array)

    if "vmax" not in kwargs:
        kwargs["vmax"] = np.nanmax(array)

    if np.isnan(kwargs["vmin"]):
        kwargs["vmin"] = -0.1
    if np.isnan(kwargs["vmax"]):
        kwargs["vmax"] = 0.1

    if plotting_function != "pcolormesh":
        if "levels" not in kwargs:
            kwargs["n_levels"] = 11 if "n_levels" not in kwargs else kwargs["n_levels"]
            kwargs["levels"] = np.linspace(
                kwargs["vmin"], kwargs["vmax"], kwargs["n_levels"]
            )
    lon = np.where(lon > 180, lon - 360, lon)
    p_handle = _plotting_function(
        lon.flatten(),
        lat.flatten(),
        array.flatten(),
        transform=ccrs.PlateCarree(),
        **kwargs,
    )
    ax.set_global()
    return p_handle
