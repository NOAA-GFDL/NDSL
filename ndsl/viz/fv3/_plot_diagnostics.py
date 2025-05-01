"""
Some helper functions for creating diagnostic plots.

These are specifically for usage in fv3net.

Uses the general purpose plotting functions in
fv3viz such as plot_cube.


"""
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import binned_statistic
import xarray as xr

from ._constants import INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER

STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]


def _mask_nan_lines(x, y):
    nan_mask = np.isfinite(y)
    return np.array(x)[nan_mask], np.array(y)[nan_mask]


def plot_diurnal_cycle(
    merged_ds, var, stack_dims=STACK_DIMS, num_time_bins=24, title=None, ylabel=None
):
    """

    Args:
        merged_ds (xr.dataset):
            can either provide a merged dataset with a "dataset" dim
            that will be used to plot separate lines for each variable, or a
            single dataset with no "dataset" dim
        var (str):
            name of variable to plot
        num_time_bins (int):
            number of bins per day
        title(str):
            optional plot title

    Returns:
        matplotlib figure
    """
    plt.clf()
    fig = plt.figure()
    if "dataset" not in merged_ds.dims:
        merged_ds = xr.concat([merged_ds], "dataset")
    for label in merged_ds["dataset"].values:
        # TODO this function mixes computation, plotting, and implicitly
        # I/O via deferred  dask calculations.
        # and should be extensively refactored.
        ds = merged_ds.sel(dataset=label)
        if len([dim for dim in ds.dims if dim in stack_dims]) > 1:
            ds = ds.stack(sample=stack_dims).dropna("sample")
        local_time = ds["local_time"].values.flatten()
        data_var = ds[var].values.flatten()
        bin_means, bin_edges, _ = binned_statistic(
            local_time, data_var, bins=num_time_bins
        )
        bin_centers = [
            0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(num_time_bins)
        ]
        bin_centers, bin_means = _mask_nan_lines(bin_centers, bin_means)
        plt.plot(bin_centers, bin_means, label=label)
    plt.xlabel("local_time [hr]")
    plt.ylabel(ylabel or var)
    plt.legend(loc="lower left")
    if title:
        plt.title(title)
    return fig


# function below here are from the previous design and probably outdated
# leaving for now as it might be adapted to work with new design


def plot_time_series(
    ds,
    vars_to_plot,
    output_dir,
    plot_filename="time_series.png",
    time_var=INIT_TIME_DIM,
    xlabel=None,
    ylabel=None,
    title=None,
):
    """ Plot one or more variables as a time series.

    Args:
        ds (xr.dataset):
            dataset containing time series variables to plot
        vars_to_plot(list[str]):
            data variables to plot
        output_dir (str):
            output directory to save figure into
        plot_filename (str):
            filename to save figure to
        time_var (str):
            name of time dimension
        xlabel (str):
            x axis label
        ylabel (str):
            y axis label
        title (str):
            plot title
    Returns:
        matplotlib figure
    """
    plt.clf()
    for var in vars_to_plot:
        time = ds[time_var].values
        plt.plot(time, ds[var].values, label=var)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    if title:
        plt.title(title)
    plt.savefig(os.path.join(output_dir, plot_filename))
