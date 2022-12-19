"""
This module provides helpful functions for creating various plots

"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import numpy.typing as npt
import matplotlib.pyplot as plt
from pyproj import Transformer
from bokeh.models import WMTSTileSource, ColumnDataSource
from bokeh.palettes import Category10, viridis
from bokeh.plotting import figure
from matplotlib.ticker import StrMethodFormatter


NDArrayFloat = npt.NDArray[np.float64]


plt.close("all")


def set_styling() -> None:
    """Sets some of the matplotlib plotting styling to be consistent throughout any module where
    plotting is implemented.
    """
    font = {"family": "serif", "size": 14}
    mpl.rc("font", **font)
    mpl.rc("text", usetex=False)
    mpl.rcParams["figure.figsize"] = (15, 6)
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["axes.axisbelow"] = True


set_styling()


def coordinateMapping(lon1, lat1, lon2, lat2):
    """Map latitude and longitude to  local cartesian coordinates

    Args:
        lon1(:obj:`numpy array of shape (1, ) or scalar`): longitude of cartesian coordinate system origin
        lat1(:obj:`numpy array of shape (1, ) or scalar`): latitude of cartesian coordinate system origin
        lon2(:obj:`numpy array of shape (n, ) or scalar`): longitude(s) of points of interest
        lat2(:obj:`numpy array of shape (n, ) or scalar`): latitude(s) of points of interest

    Returns:
        Tuple representing cartesian coordinates (x, y); if arguments entered as scalars, returns scalars in tuple,
        if arguments entered as numpy arrays, returns numpy arrays each of shape (n,1)

    """
    R = 6371e3  # Earth radius, in meters

    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    lambda1 = np.radians(lon1)
    lambda2 = np.radians(lon2)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    rho = R * c

    a = np.sin(lambda2 - lambda1) * np.cos(phi2)
    b = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1)

    theta = -1 * np.arctan2(a, b) + np.pi / 2

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return (x, y)


def plot_array(project):
    """Plot locations of turbines and met towers, with labels, on latitude/longitude grid

    Args:
        project(:obj:`plant object`): project to be plotted

    Returns:
        (None)
    """
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    asset_groups = project.asset.df.groupby("type")

    turbines = asset_groups.get_group("turbine")
    X = turbines["longitude"]
    Y = turbines["latitude"]
    labels = turbines["id"].tolist()

    ax.scatter(X, Y, marker="o", color="k")
    for label, x, y in zip(labels, X, Y):
        ax.annotate(label, xy=(x, y), xytext=(-8, 5), textcoords="offset points", fontsize=6)

    towers = asset_groups.get_group("tower")
    X = towers["longitude"]
    Y = towers["latitude"]
    labels = towers["id"].tolist()

    ax.scatter(X, Y, marker="s", color="r")
    for label, x, y in zip(labels, X, Y):
        ax.annotate(
            label, xy=(x, y), xytext=(-8, -10), textcoords="offset points", fontsize=6, color="r"
        )

    ax.set_xlabel("Longitude, [deg]")
    ax.set_ylabel("Latitude, [deg]")

    del X, Y, labels, x, y, label


def subplot_powerRose_array(
    project,
    turbine_ids,
    shift=0,
    direction=1,
    columns=None,
    left_margin=0.1,
    bottom_margin=0.1,
    gap_w_frac=0.2,
    gap_h_frac=0.2,
    aspect=1,
):
    """Wrapper for powerRose_array plotting for multiple subplots

    Args:
        project(:obj:`plant object`): project to be plotted
        turbine_ids(:obj:`list of strings`): ids of turbines to be plotted
        shift(:obj:`list of scalars`): number of degrees to rotate wind direction data, each plotted as new line
        direction(:obj:`-1, 1`): wind direction data measured clockwise (1) or counterclockwise (-1)
        columns(:obj:`scalar integer`): number of subplot columns
        left_margin(:obj:`scalar`): fraction of figure width to include as left margin
        bottom_margin(:obj:`scalar`): fraction of figure height to include as bottom margin
        gap_w_frac(:obj:`scalar`): fraction of figure width to include between subplots
        gap_h_frac(:obj:`scalar`): fraction of figure height to include as between subplots
        aspect(:obj:`scalar`): aspect ratio for subplots

    Returns:
        (None)
    """
    if columns is None:
        if len(turbine_ids) > 3:
            columns = 3
        else:
            columns = len(turbine_ids)

    rows = int(np.ceil(float(len(turbine_ids)) / columns))

    if aspect != 1:
        sp_w_frac = 1.0 / columns
        sp_h_frac = 1.0 / rows
    else:
        sp_w_frac = min(1.0 / columns, 1.0 / rows)
        sp_h_frac = sp_w_frac

    fig = plt.figure()

    for i, tid in enumerate(turbine_ids):
        ir, ic = np.unravel_index(i, (rows, columns))
        rect = [
            left_margin + (sp_w_frac + gap_w_frac) * ic,
            bottom_margin + (sp_h_frac + gap_h_frac) * (rows - ir),
            sp_w_frac,
            sp_h_frac,
        ]

        powerRose_array(fig, rect, tid, shift, direction)


def powerRose_array(project, fig, rect, tid, model_eval, shift=[0], direction=1):
    """Plot power curve on polar coordinates overlaying plot of surrounding array (both local and further distance)

    Args:
        project(:obj:`plant object`): project to be plotted
        fig(:obj:`figure handle`): figure handle
        rect(:obj:`list of four scalars`): [left offset, bottom offset, width, height] as fractions of figure
        width/height
        tid(:obj:`string`): id of turbine to be plotted
        model_eval(:obj:`dict`): JORDAN, WHAT IS THIS SUPPOSED TO BE??
        shift(:obj:`list of scalars`): number of degrees to rotate wind direction data, each plotted as new line
        direction(:obj:`-1, 1`): wind direction data measured clockwise (1) or counterclockwise (-1)

    Returns:
    """
    # the carthesian axis:
    ax_carthesian = fig.add_axes(rect, frameon=True)

    # plotting the line on the carthesian axis
    X0 = project.asset.df.loc[project.asset.df.id == tid, "longitude"]
    Y0 = project.asset.df.loc[project.asset.df.id == tid, "latitude"]
    XY = project.asset.df.apply(
        lambda r: coordinateMapping(X0, Y0, r["longitude"], r["latitude"]), axis=1
    )
    X = XY.apply(lambda r: r[0])
    Y = XY.apply(lambda r: r[1])

    ax_carthesian.scatter(X, Y, s=200, c="black", marker="o")
    ax_carthesian.axis("equal")
    ax_carthesian.set_xlim([-900, 900])
    ax_carthesian.set_ylim([-900, 900])
    ax_carthesian.tick_params(axis="both", which="major", pad=40)
    ax_carthesian.spines["left"].set_visible(True)
    ax_carthesian.spines["left"].set_color("black")
    ax_carthesian.spines["bottom"].set_visible(True)
    ax_carthesian.spines["bottom"].set_color("black")
    ax_carthesian.set_title("Turbine %s" % (tid))

    # the second carthesian axis:
    ax_carthesian_2 = fig.add_axes(rect, frameon=False)

    # plotting the line on the carthesian axis
    ax_carthesian_2.scatter(X, Y, s=50, c="g", marker="x")
    ax_carthesian_2.axis("equal")
    ax_carthesian_2.set_xlim([-3000, 3000])
    ax_carthesian_2.set_ylim([-3000, 3000])
    ax_carthesian_2.tick_params(axis="y", which="major", pad=100, colors="green")
    ax_carthesian_2.tick_params(axis="x", which="major", pad=75, colors="green")
    ax_carthesian_2.yaxis.label.set_color("green")
    ax_carthesian_2.xaxis.label.set_color("green")
    ax_carthesian_2.spines["left"].set_visible(True)
    ax_carthesian_2.spines["left"].set_color("green")
    ax_carthesian_2.spines["bottom"].set_visible(True)
    ax_carthesian_2.spines["bottom"].set_color("green")

    # the polar axis:
    ax_polar = fig.add_axes(rect, polar=True, frameon=False)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)

    # the polar plot
    cm = plt.get_cmap("jet")
    ax_polar.set_color_cycle([cm(1.0 * i / len(shift)) for i in range(len(shift))])
    for i in range(len(shift)):
        ax_polar.plot(
            (model_eval["winddirection"] * direction + shift[i]) * np.pi / 180,
            model_eval[tid],
            linewidth=3.0,
            label=str(shift[i]) + " deg",
        )

    ntick = 3
    ticks = [
        np.round(
            np.min(model_eval[tid])
            + ((np.max(model_eval[tid]) - np.min(model_eval[tid])) / ntick) * (t + 1)
        )
        for t in np.arange(ntick)
    ]

    ax_polar.set_rmax(np.ceil(1.05 * np.max(model_eval[tid])))
    ax_polar.set_rmin(np.floor(0.95 * np.min(model_eval[tid])))

    ax_polar.set_rticks(ticks)  # less radial ticks
    ax_polar.grid(True)
    ax_polar.legend()


def subplt_c1_c2(turbine, axarr, c1, c2, c="Blues", xlim=None, ylim=None, xlabel=None, ylabel=None):
    """hexbin plot of turbine[c1] vs turbine [c2]

    Args:
        turbine(:obj:`pandas dataframe`): data to be plotted
        axarr(:obj:`axis handle`): axis handle
        c1(:obj:`string`): column name of x axis
        c2(:obj:`string`): column name of y axis
        c(:obj:`string` or colormap handle): colormap

    Returns:
        hb(:obj:`plot handle`):
    """
    hb = axarr.hexbin(turbine[c1], turbine[c2], cmap=c, gridsize=128, vmin=0, vmax=8)

    if xlim:
        axarr.set_xlim(xlim)
    if ylim:
        axarr.set_ylim(ylim)
    if xlabel:
        axarr.set_xlabel(xlabel)
    if ylabel:
        axarr.set_ylabel(ylabel)

    return hb


def subplt_c1_c2_flagged(
    turbine,
    axarr,
    c1,
    c2,
    flag_cols,
    flag_value,
    cmap="Blues",
    xlim=None,
    ylim=None,
    xlabel=None,
    ylabel=None,
):
    """hexbin plot of turbine[c1] vs turbine [c2], showing only for which <flag_cols> have <value>

    Args:
        turbine(:obj:`pandas dataframe`): data to be plotted
        axarr(:obj:`axis handle`): axis handle
        c1(:obj:`string`): column name of x axis
        c2(:obj:`string`): column name of y axis
        c(:obj:`string` or colormap handle): colormap
        flag_cols(:obj:`list of strings`): column name(s) for flag columns
        value_cols(:obj:`string`): value in <filter_cols> for which data plotted

    Returns:
        hb(:obj:`plot handle`):
    """
    flag_indices = None
    for c in flag_cols:
        if flag_indices is None:
            flag_indices = turbine.loc[turbine[c] == flag_value].index.values
        else:
            flag_indices = np.append(
                flag_indices, turbine.loc[turbine[c] == flag_value].index.values
            )
        flag_indices = np.unique(flag_indices)

    hb = axarr.hexbin(
        turbine.loc[flag_indices, c1],
        turbine.loc[flag_indices, c2],
        cmap=cmap,
        gridsize=128,
        vmin=0,
        vmax=8,
    )

    if xlim:
        axarr.set_xlim(xlim)
    if ylim:
        axarr.set_ylim(ylim)
    if xlabel:
        axarr.set_xlabel(xlabel)
    if ylabel:
        axarr.set_ylabel(ylabel)

    axarr.text(
        xlim[0] + 0.1 * (xlim[1] - xlim[0]),
        ylim[0] + 0.9 * (ylim[1] - ylim[0]),
        "%.2f%%" % (float(len(flag_indices)) / float(len(turbine)) * 100.0),
    )
    return hb


def subplt_c1_c2_raw_flagged(
    turbine,
    axarr,
    c1,
    c2,
    flag_cols,
    flag_value,
    cmap="Blues",
    markers=["x"],
    colors=["r"],
    xlim=None,
    ylim=None,
    xlabel=None,
    ylabel=None,
):
    """hexbin plot of turbine[c1] vs turbine [c2], showing data <flag_cols> have <value> as overlaid scatter plot

    Args:
        turbine(:obj:`pandas dataframe`): data to be plotted
        axarr(:obj:`axis handle`): axis handle
        c1(:obj:`string`): column name of x axis
        c2(:obj:`string`): column name of y axis
        c(:obj:`string` or colormap handle): colormap
        flag_cols(:obj:`list of strings`): column name(s) for flag columns
        value_cols(:obj:`string`): value in <filter_cols> for which data plotted

    Returns:
        hb(:obj:`plot handle`):
    """

    hb = subplt_c1_c2(turbine, axarr, c1, c2, cmap, xlim=xlim, ylim=ylim)

    if len(markers) == 1:
        flag_indices = []
        for c in flag_cols:
            flag_indices = (
                flag_indices + turbine.loc[turbine[c] == flag_value].index.values.tolist()
            )
            flag_indices = np.unique(flag_indices)
        axarr.scatter(
            turbine.loc[flag_indices, c1],
            turbine.loc[flag_indices, c2],
            marker=markers[0],
            color=colors[0],
        )
    else:
        for ic, c in enumerate(flag_cols):
            flag_indices = turbine.loc[turbine[c] == flag_value].index.values.tolist()
            axarr.scatter(
                turbine.loc[flag_indices, c1],
                turbine.loc[flag_indices, c2],
                marker=markers[ic],
                color=colors[ic],
            )

    if xlim:
        axarr.set_xlim(xlim)
    if ylim:
        axarr.set_ylim(ylim)
    if xlabel:
        axarr.set_xlabel(xlabel)
    if ylabel:
        axarr.set_ylabel(ylabel)

    axarr.text(
        xlim[0] + 0.1 * (xlim[1] - xlim[0]),
        ylim[0] + 0.9 * (ylim[1] - ylim[0]),
        "%.2f%%" % (float(len(flag_indices)) / float(len(turbine)) * 100.0),
    )
    return hb


def subplt_power_curve(turbine, axarr, fig, c3, pc):
    hb = subplt_c1_c2_raw_flagged(turbine, axarr, fig, "windspeed_ms", "power_kw")

    turbine.sort_values(by=c3, inplace=True)
    axarr.plot(turbine[c3], turbine[pc], "r", label="power curve")

    return hb


def turbine_polar_line(
    array, theta, r, line_label, tid, color="b", ax_carthesian=None, ax_polar=None
):
    """Polar plot (<r>, <theta>) overlaying plot of surrounding array, centered on turbine <tid>

    Args:
        array(:obj:`pandas dataframe`): index by (string) labels of asset_df, 'x' and 'y' coordinate columns
        theta(:obj:`pandas series, np array, list`): anglular coordinates of points, in degrees
        r(:obj:`pandas series, np array, list`): radial coordinates of points
        line_label(:obj:`str`): legend label
        tid(:obj:`str`): index of asset on which to center carthesian axes
        ax_carthesian(:obj:`axes handle`): existing carthesian axes on which to add array plot
        ax_polar(:obj:`axes handle`): existing polar axes on which to add plot

    Returns:
        ax_carthesian(:obj:`axes handle`): carthesian axes on which array plotted
        ax_polar(:obj:`axes handle`): polar axes on which data plotted
    """
    # the carthesian axis:

    if ax_carthesian is None:
        fig = plt.figure(figsize=(10.0, 10.0))
        rect = [0.1, 0.1, 0.8, 0.8]
        ax_carthesian = fig.add_axes(rect, frameon=True)

        # plotting the line on the carthesian axis
        x_offset = array.loc[tid, "x"]
        y_offset = array.loc[tid, "y"]

        X = array["x"] - x_offset
        Y = array["y"] - y_offset
        turbine_labels = array.index

        ax_carthesian.scatter(X, Y, marker="o", color="k", s=20)
        for turbine_label, x, y in zip(turbine_labels, X, Y):
            ax_carthesian.annotate(
                turbine_label, xy=(x, y), xytext=(-8, 5), textcoords="offset points"
            )

        ax_carthesian.axis("equal")
        ax_carthesian.set_xlim([-1000, 1000])
        ax_carthesian.set_ylim([-1000, 1000])
        ax_carthesian.tick_params(axis="both", which="major", pad=70)
        ax_carthesian.spines["left"].set_visible(True)
        ax_carthesian.spines["left"].set_color("black")
        ax_carthesian.spines["bottom"].set_visible(True)
        ax_carthesian.spines["bottom"].set_color("black")

    # the polar axis:
    if ax_polar is None:
        ax_polar = fig.add_axes(rect, polar=True, frameon=False)
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        ax_polar.set_rmax(np.ceil(1.05 * np.max(r)))
        ax_polar.set_rmin(np.floor(0.95 * np.min(r)))
    else:
        ax_polar.set_rmax(np.max([np.ceil(1.05 * np.max(r)), ax_polar.get_rmax()]))
        ax_polar.set_rmin(np.min([np.floor(0.95 * np.min(r)), ax_polar.get_rmin()]))

    # the polar plot
    ax_polar.plot((theta) * np.pi / 180, r, linewidth=3.0, label=line_label, color=color)

    ntick = 3
    ticks = [
        np.round(
            ax_polar.get_rmin() + ((ax_polar.get_rmax() - ax_polar.get_rmin()) / ntick) * (t + 1)
        )
        for t in np.arange(ntick)
    ]

    ax_polar.set_rticks(ticks)  # less radial ticks
    ax_polar.grid(True)

    return ax_carthesian, ax_polar


def turbine_polar_4Dscatter(array, tid, theta, r, color, size, cmap="autumn_r"):
    """Polar plot (<r>, <theta>) overlaying plot of surrounding array, centered on turbine <tid>

    Args:
        array(:obj:`pandas dataframe`): index by (string) labels of asset_df, 'x' and 'y' coordinate columns
        tid(:obj:`str`): index of asset on which to center carthesian axes
        theta(:obj:`pandas series, np array, list`): anglular coordinates of points, in degrees
        r(:obj:`pandas series, np array, list`): radial coordinates of points
        color(:obj:`pandas series, np array, list`): color of points
        size(:obj:`pandas series, np array, list`): size of points

    Returns:
        ax_carthesian(:obj:`axes handle`): carthesian axes on which array plotted
        ax_polar(:obj:`axes handle`): polar axes on which data plotted
    """

    # the polar axis:
    fig = plt.figure(figsize=(10.0, 10.0))
    rect = [0.1, 0.1, 0.8, 0.8]
    ax_polar = fig.add_axes(rect, polar=True, frameon=False)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_rmax(np.ceil(1.05 * np.max(r)))
    ax_polar.set_rmin(np.floor(0.95 * np.min(r)))

    # the polar plot
    sc = ax_polar.scatter((theta) * np.pi / 180, r, s=size * 10, c=color, cmap=cmap)

    ntick = 3
    ticks = [
        np.round(
            ax_polar.get_rmin() + ((ax_polar.get_rmax() - ax_polar.get_rmin()) / ntick) * (t + 1)
        )
        for t in np.arange(ntick)
    ]

    ax_polar.set_rticks(ticks)  # less radial ticks
    ax_polar.grid(True)
    ax_polar.legend()

    # the carthesian axis:
    ax_carthesian = fig.add_axes(rect, frameon=True)
    ax_carthesian.patch.set_alpha(0)
    # plotting the line on the carthesian axis
    x_offset = array.loc[tid, "x"]
    y_offset = array.loc[tid, "y"]

    X = array["x"] - x_offset
    Y = array["y"] - y_offset
    turbine_labels = array.index

    ax_carthesian.scatter(X, Y, marker="o", color="k", s=20)
    for turbine_label, x, y in zip(turbine_labels, X, Y):
        ax_carthesian.annotate(turbine_label, xy=(x, y), xytext=(-8, 5), textcoords="offset points")

        ax_carthesian.axis("equal")
        ax_carthesian.set_xlim([-1000, 1000])
        ax_carthesian.set_ylim([-1000, 1000])
        ax_carthesian.tick_params(axis="both", which="major", pad=70)
        ax_carthesian.spines["left"].set_visible(True)
        ax_carthesian.spines["left"].set_color("black")
        ax_carthesian.spines["bottom"].set_visible(True)
        ax_carthesian.spines["bottom"].set_color("black")
        ax_carthesian.set_title("Turbine %s" % (tid))

    box = ax_carthesian.get_position()

    # create color bar
    axColor = plt.axes([box.x0 * 1.1 + box.width * 1.1, box.y0, 0.01, box.height])
    plt.colorbar(sc, cax=axColor, orientation="vertical")

    return ax_carthesian, ax_polar


def turbine_polar_contourf(array, tid, theta, r, c, cmap="autumn_r"):
    """Polar plot (<r>, <theta>) overlaying plot of surrounding array, centered on turbine <tid>

    Args:
        array(:obj:`pandas dataframe`): index by (string) labels of asset_df, 'x' and 'y' coordinate columns
        tid(:obj:`str`): index of asset on which to center carthesian axes
        theta(:obj:`pandas series, np array, list`): anglular coordinates of points, in degrees
        r(:obj:`pandas series, np array, list`): radial coordinates of points
        c(:obj:`pandas series, np array, list`): colors of points

    Returns:
        ax_carthesian(:obj:`axes handle`): carthesian axes on which array plotted
        ax_polar(:obj:`axes handle`): polar axes on which data plotted
    """

    # the polar axis:
    fig = plt.figure(figsize=(10.0, 10.0))
    rect = [0.1, 0.1, 0.8, 0.8]
    ax_polar = fig.add_axes(rect, polar=True, frameon=False)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_rmax(np.ceil(1.05 * np.max(r)))
    ax_polar.set_rmin(np.floor(0.95 * np.min(r)))

    # the polar plot
    cf = ax_polar.contourf((theta) * np.pi / 180, r, c, cmap=cmap)
    plt.colorbar(cf)

    ntick = 3
    ticks = [
        np.round(
            ax_polar.get_rmin() + ((ax_polar.get_rmax() - ax_polar.get_rmin()) / ntick) * (t + 1)
        )
        for t in np.arange(ntick)
    ]

    ax_polar.set_rticks(ticks)  # less radial ticks
    ax_polar.grid(True)
    ax_polar.legend()

    ax_carthesian = fig.add_axes(rect, frameon=True)
    ax_carthesian.patch.set_alpha(0)
    # plotting the line on the carthesian axis
    x_offset = array.loc[tid, "x"]
    y_offset = array.loc[tid, "y"]

    X = array["x"] - x_offset
    Y = array["y"] - y_offset
    turbine_labels = array.index

    ax_carthesian.scatter(X, Y, marker="o", color="k", s=20)
    for turbine_label, x, y in zip(turbine_labels, X, Y):
        ax_carthesian.annotate(turbine_label, xy=(x, y), xytext=(-8, 5), textcoords="offset points")

        ax_carthesian.axis("equal")
        ax_carthesian.set_xlim([-1000, 1000])
        ax_carthesian.set_ylim([-1000, 1000])
        ax_carthesian.tick_params(axis="both", which="major", pad=70)
        ax_carthesian.spines["left"].set_visible(True)
        ax_carthesian.spines["left"].set_color("black")
        ax_carthesian.spines["bottom"].set_visible(True)
        ax_carthesian.spines["bottom"].set_color("black")
        ax_carthesian.set_title("Turbine %s" % (tid))

    return ax_carthesian, ax_polar


def turbine_polar_contour(
    array, tid, theta, r, z, levels, colors, ax_carthesian=None, ax_polar=None, label=""
):
    """Polar plot (<r>, <theta>) overlaying plot of surrounding array, centered on turbine <tid>

    Args:
        array(:obj:`pandas dataframe`): index by (string) labels of asset_df, 'x' and 'y' coordinate columns
        tid(:obj:`str`): index of asset on which to center carthesian axes
        theta(:obj:`pandas series, np array, list`): anglular coordinates of points, in degrees
        r(:obj:`pandas series, np array, list`): radial coordinates of points
        z(:obj:`pandas series, np array, list`): colors of points
        levels(:obj:`list of float`): levels at which to draw contours
        colors(:obj:`list of colormap rows`): colors of drawn contours
        ax_carthesian(:obj:`axes handle`): carthesian axes on which array plotted
        ax_polar(:obj:`axes handle`): polar axes on which data plotted
        label(:obj:`string`): legend label

    Returns:
        ax_carthesian(:obj:`axes handle`): carthesian axes on which array plotted
        ax_polar(:obj:`axes handle`): polar axes on which data plotted
    """

    # the polar axis:
    if ax_polar is None:
        fig = plt.figure(figsize=(10.0, 10.0))
        rect = [0.1, 0.1, 0.8, 0.8]
        ax_polar = fig.add_axes(rect, polar=True, frameon=False)
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        ax_polar.set_rmax(np.ceil(1.05 * np.max(r)))
        ax_polar.set_rmin(np.floor(0.95 * np.min(r)))
    else:
        ax_polar.set_rmax(np.max([np.ceil(1.05 * np.max(r)), ax_polar.get_rmax()]))
        ax_polar.set_rmin(np.min([np.floor(0.95 * np.min(r)), ax_polar.get_rmin()]))

    # the polar plot
    c = ax_polar.contour((theta) * np.pi / 180, r, z, levels=levels, colors=colors)
    artists, labels = c.legend_elements(variable_name=label)

    ntick = 3
    ticks = [
        np.round(
            ax_polar.get_rmin() + ((ax_polar.get_rmax() - ax_polar.get_rmin()) / ntick) * (t + 1)
        )
        for t in np.arange(ntick)
    ]

    ax_polar.set_rticks(ticks)  # less radial ticks
    ax_polar.grid(True)

    if ax_carthesian is None:
        ax_carthesian = fig.add_axes(rect, frameon=True)
        ax_carthesian.patch.set_alpha(0)
        # plotting the line on the carthesian axis
        x_offset = array.loc[tid, "x"]
        y_offset = array.loc[tid, "y"]

        X = array["x"] - x_offset
        Y = array["y"] - y_offset
        turbine_labels = array.index

        ax_carthesian.scatter(X, Y, marker="o", color="k", s=20)
        for turbine_label, x, y in zip(turbine_labels, X, Y):
            ax_carthesian.annotate(
                turbine_label, xy=(x, y), xytext=(-8, 5), textcoords="offset points"
            )

        ax_carthesian.axis("equal")
        ax_carthesian.set_xlim([-1000, 1000])
        ax_carthesian.set_ylim([-1000, 1000])
        ax_carthesian.tick_params(axis="both", which="major", pad=70)
        ax_carthesian.spines["left"].set_visible(True)
        ax_carthesian.spines["left"].set_color("black")
        ax_carthesian.spines["bottom"].set_visible(True)
        ax_carthesian.spines["bottom"].set_color("black")
        ax_carthesian.set_title("Turbine %s" % (tid))

    return ax_carthesian, ax_polar, artists, labels


def luminance(rgb):

    """Calculates the brightness of an rgb 255 color. See https://en.wikipedia.org/wiki/Relative_luminance

    Args:
        rgb(:obj:`tuple`): 255 (red, green, blue) tuple

    Returns:
        luminance(:obj:`scalar`): relative luminance

    Example:

        .. code-block:: python

            >>> rgb = (255,127,0)
            >>> luminance(rgb)
            0.5687976470588235

            >>> luminance((0,50,255))
            0.21243529411764706

    """

    luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255

    return luminance


def color_to_rgb(color):

    """Converts named colors, hex and normalised RGB to 255 RGB values

    Args:
        color(:obj:`color`): RGB, HEX or named color

    Returns:
        rgb(:obj:`tuple`): 255 RGB values

    Example:

        .. code-block:: python

            >>> color_to_rgb("Red")
            (255, 0, 0)

            >>> color_to_rgb((1,1,0))
            (255,255,0)

            >>> color_to_rgb("#ff00ff")
            (255,0,255)
    """

    if isinstance(color, tuple):
        if max(color) > 1:
            color = tuple([i / 255 for i in color])

    rgb = mpl.colors.to_rgb(color)

    rgb = tuple([int(i * 255) for i in rgb])

    return rgb


def plot_windfarm(
    asset_df,
    tile_name="OpenMap",
    plot_width=800,
    plot_height=800,
    marker_size=14,
    kwargs_for_figure={},
    kwargs_for_marker={},
):

    """Plot the windfarm spatially on a map using the Bokeh plotting libaray.

    Args:
        asset_df(:obj:`pd.DataFrame`): PlantData.asset object containing the asset metadata.
        tile_name(:obj:`str`): tile set to be used for the underlay, e.g. OpenMap, ESRI, OpenTopoMap
        plot_width(:obj:`scalar`): width of plot
        plot_height(:obj:`scalar`): height of plot
        marker_size(:obj:`scalar`): size of markers
        kwargs_for_figure(:obj:`dict`): additional figure options for advanced users, see Bokeh docs
        kwargs_for_marker(:obj:`dict`): additional marker options for advanced users, see Bokeh docs. We have some custom behavior around the "fill_color" attribute. If "fill_color" is not defined, OpenOA will use an internally defined color pallete. If "fill_color" is the name of a column in the asset table, OpenOA will use the value of that column as the marker color. Otherwise, "fill_color" is passed through to Bokeh.

    Returns:
        Bokeh_plot(:obj:`axes handle`): windfarm map

    """
    """
    TODO: Add this back in when it can be debugged
    lats, lons = transformer.transform(...) -> ValueError: not enough values to unpack (expected 2, got 0)

    Example:
        .. bokeh-plot::

            import pandas as pd
            from bokeh.plotting import figure, output_file, show

            from openoa.utils.plot import plot_windfarm

            from examples import project_ENGIE

            # Load plant object
            project = project_ENGIE.prepare("../examples/data/la_haute_borne")

            # Create the bokeh wind farm plot
            show(plot_windfarm(project.asset, tile_name="ESRI", plot_width=600, plot_height=600))
    """

    # See https://wiki.openstreetmap.org/wiki/Tile_servers for various tile services
    MAP_TILES = {
        "OpenMap": WMTSTileSource(url="http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png"),
        "ESRI": WMTSTileSource(
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.jpg"
        ),
        "OpenTopoMap": WMTSTileSource(url="https://tile.opentopomap.org/{Z}/{X}/{Y}.png"),
    }

    # Use pyproj to transform longitude and latitude into web-mercator and add to a copy of the asset dataframe
    TRANSFORM_4326_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857")

    asset_df["x"], asset_df["y"] = TRANSFORM_4326_TO_3857.transform(
        asset_df["latitude"], asset_df["longitude"]
    )
    asset_df["coordinates"] = tuple(zip(asset_df["latitude"], asset_df["longitude"]))

    # Define default and then update figure and marker options based on kwargs
    figure_options = {
        "tools": "save,hover,pan,wheel_zoom,reset,help",
        "x_axis_label": "Longitude",
        "y_axis_label": "Latitude",
        "match_aspect": True,
        "tooltips": [("id", "@id"), ("type", "@type"), ("(Lat,Lon)", "@coordinates")],
    }
    figure_options.update(kwargs_for_figure)

    marker_options = {
        "marker": "circle_y",
        "line_width": 1,
        "alpha": 0.8,
        "fill_color": "auto_fill_color",
        "line_color": "auto_line_color",
        "legend_group": "type",
    }
    marker_options.update(kwargs_for_marker)

    # Create an appropriate fill color map and contrasting line color
    if marker_options["fill_color"] == "auto_fill_color":
        color_grouping = marker_options["legend_group"]

        asset_df = asset_df.sort_values(color_grouping)

        if len(set(asset_df[color_grouping])) <= 10:
            color_palette = list(Category10[10])
        else:
            color_palette = viridis(len(set(asset_df[color_grouping])))

        color_mapping = dict(zip(set(asset_df[color_grouping]), color_palette))
        asset_df["auto_fill_color"] = asset_df[color_grouping].map(color_mapping)
        asset_df["auto_fill_color"] = asset_df["auto_fill_color"].apply(color_to_rgb)
        asset_df["auto_line_color"] = [
            "black" if luminance(color) > 0.5 else "white" for color in asset_df["auto_fill_color"]
        ]

    else:
        if marker_options["fill_color"] in asset_df.columns:
            asset_df[marker_options["fill_color"]] = asset_df[marker_options["fill_color"]].apply(
                color_to_rgb
            )
            asset_df["auto_line_color"] = [
                "black" if luminance(color) > 0.5 else "white"
                for color in asset_df[marker_options["fill_color"]]
            ]

        else:
            asset_df["auto_line_color"] = "black"

    # Create the bokeh data source without the "geometry" that isn't compatible with bokeh
    source = ColumnDataSource(asset_df.drop(columns=["geometry"]))

    # Create a bokeh figure with tiles
    plot_map = figure(
        width=plot_width,
        height=plot_height,
        x_axis_type="mercator",
        y_axis_type="mercator",
        **figure_options,
    )

    plot_map.add_tile(MAP_TILES[tile_name])

    # Plot the asset devices
    plot_map.scatter(x="x", y="y", source=source, size=marker_size, **marker_options)

    return plot_map


def plot_by_id(
    df: pd.DataFrame, id_col: str, x_axis: str, y_axis: str, return_fig: bool = False
) -> None:
    """Function to plot any two fields against each other in a dataframe with unique plots for each
    ID.

    Args:
        df(:obj:`pd.DataFrame`): The dataframe for comparing values.
        id_col(:obj:`str`): The id column (or index column) in `df`.
        x_axis(:obj:`str`): Independent variable to plot, should align with a column in `df`.
        y_axis(:obj:`str`): Dependent variable to plot, should align with a column in `df`.
        return_fig(:obj:`bool`): Indicator for if the figure and axes objects should be returned,
            by default False.

    Returns:
        (:obj:`None`)
    """
    # Operate on a totally new copy of the data so that transofrmations don't carry through
    df = df.copy()

    # Get the id_col as the first index in the multi index or ensure it is the primary index
    if not isinstance(df.index, pd.MultiIndex):
        if id_col != df.index.name:
            df = df.set_index(id_col, append=True)
    elif id_col not in df.index.names:
        df = df.set_index(id_col, append=True)

    if isinstance(df.index, pd.MultiIndex):
        df = df.swaplevel(id_col, 0)

    # Check that the columns are valid
    if x_axis not in df.columns:
        raise ValueError(f"'{x_axis}' is not a valid column")
    if y_axis not in df.columns:
        raise ValueError(f"'{x_axis}' is not a valid column")

    # Create the plotting parameters
    id_arrary = df.index.get_level_values(id_col).unique()
    max_cols = 4
    num_id = id_arrary.size
    num_rows = int(np.ceil(num_id / max_cols))

    # Create the plot
    fig, axes_list = plt.subplots(
        num_rows, max_cols, sharex=True, sharey=True, figsize=(15, num_rows * 5)
    )
    for i, (t_id, ax) in enumerate(zip(id_arrary, axes_list.flatten())):
        scada = df.loc[t_id]
        ax.scatter(scada[x_axis], scada[y_axis], s=5)

        ax.set_title(t_id)

        # Only add axis labels for the bottom row and leftmost column
        if np.floor(i / 4) + 1 == num_rows:
            ax.set_xlabel(x_axis)
        if i % 4 == 0:
            ax.set_ylabel(y_axis)

    # Delete the extra axes
    num_axes = axes_list.size
    if i < num_axes - 1:
        for j in range(i + 1, num_axes):
            fig.delaxes(axes_list.flatten()[j])

    fig.tight_layout()
    plt.show()

    if return_fig:
        return fig, axes_list


def column_histograms(df: pd.DataFrame, columns: list = None, return_fig: bool = False):
    """Produces a histogram plot for each numeric column in :py:attr:`df`.

    Args:
        df(:obj:`pd.DataFrame`): The dataframe for plotting.
        return_fig(:obj:`bool`): Indicator for if the figure and axes objects should be returned,
            by default False.

    Returns:
        (None)
    """
    df = df.select_dtypes((int, float)).copy()
    columns = df.columns.tolist() if columns is None else columns
    num_cols = len(columns)
    max_cols = 3
    num_rows = int(np.ceil(num_cols / max_cols))

    fig, axes_list = plt.subplots(num_rows, max_cols, figsize=(15, num_rows * 5))
    for i, (col, ax) in enumerate(zip(columns, axes_list.flatten())):
        data = df.loc[:, col].dropna().values
        ax.hist(data, 40)
        ax.set_title(col)

        # Only add axis labels for the bottom row
        if i % 4 == 0:
            ax.set_ylabel("Count of Occurrences")

    # Delete the extra axes
    num_axes = axes_list.size
    if i < num_axes - 1:
        for j in range(i + 1, num_axes):
            fig.delaxes(axes_list.flatten()[j])

    fig.tight_layout()
    plt.show()
    if return_fig:
        return fig, axes_list


def plot_power_curve(
    wind_speed: pd.Series,
    power: pd.Series,
    flag: np.ndarray | pd.Series,
    flag_labels: tuple[str, str] = ("Flagged Readings", "Power Curve"),
    xlim: tuple[float, float] = (None, None),
    ylim: tuple[float, float] = (None, None),
    legend: bool = False,
    return_fig: bool = False,
    figure_kwargs: dict = {},
    legend_kwargs: dict = {},
    scatter_kwargs: dict = {},
) -> None | tuple[plt.Figure, plt.Axes]:
    """Plots the individual points on a power curve, with an optional `flag` filtering for singling
    out readings in the figure. If `flag` is all false values then no overlaid flagged scatter points
    will be created.

    Args:
        wind_speed (:obj:`pandas.Series`): A pandas Series or numpy array of the recorded wind
            speeds, in m/s.
        power (:obj:`pandas.Series` | `np.ndarray`): A pandas Series or numpy array of
            the recorded power, in kW.
        flag (:obj:`numpy.ndarray` | `pd.Series`): A pandas Series or numpy array of booleans for
            which points to flag in the windspeed and power data.
        flag_labels (:obj:`tuple[str, str]`, optional): The labels to give to the scatter points,
            corresponding to the flagged points and raw points, respectively. Defaults to
            ("Flagged Readings", "Power Curve").
        xlim (:obj:`tuple[float, float]`, optional): A tuple of the x-axis (min, max) values.
            Defaults to (None, None).
        ylim (:obj:`tuple[float, float]`, optional): A tuple of the y-axis (min, max) values.
            Defaults to (None, None).
        legend (:obj:`bool`, optional): Set to True to place a legend in the figure, otherwise set
            to False. Defaults to False.
        return_fig (:obj:`bool`, optional): Set to True to return the figure and axes objects,
            otherwise set to False. Defaults to False.
        figure_kwargs (:obj:`dict`, optional): Additional keyword arguments that should be passed to
            `plt.figure`. Defaults to {}.
        scatter_kwargs (:obj:`dict`, optional): Additional keyword arguments that should be passed
            to `ax.scatter`. Defaults to {}.
        legend_kwargs (:obj:`dict`, optional): Additional keyword arguments that should be passed to
            `ax.legend`. Defaults to {}.

    Returns:
        None | tuple[plt.Figure, plt.Axes]: _description_
    """
    figure_kwargs.setdefault("dpi", 200)
    fig = plt.figure(**figure_kwargs)
    ax = fig.add_subplot(111)

    if ~np.all(flag):
        pc_label = "Power Curve" if flag_labels is None else flag_labels[1]
        ax.scatter(wind_speed, power, label=pc_label, **scatter_kwargs)
    else:
        pc_label = "Power Curve" if flag_labels is None else flag_labels[1]
        flagged_label = "Flagged Readings" if flag_labels is None else flag_labels[0]
        ax.scatter(wind_speed, power, label=pc_label, **scatter_kwargs)
        ax.scatter(wind_speed[flag], power[flag], label=flagged_label, **scatter_kwargs)

    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    if legend:
        ax.legend(**legend_kwargs)

    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power (kW)")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if return_fig:
        return fig, ax

    fig.tight_layout()
    plt.show()


def plot_monthly_reanalysis_windspeed(
    data: dict[str, pd.DataFrame],
    windspeed_col: str,
    plant_por: tuple[datetime.datetime, datetime.datetime],
    normalize: bool = True,
    xlim: tuple[datetime.datetime, datetime.datetime] = (None, None),
    ylim: tuple[float, float] = (None, None),
    return_fig: bool = False,
    figure_kwargs: dict = {},
    plot_kwargs: dict = {},
    legend_kwargs: dict = {},
) -> None | tuple[plt.Figure, plt.Axes]:
    """Make a plot of the normalized annual average wind speeds from reanalysis data to show general
    trends for each, and highlighting the period of record for the plant data.

    Args:
        data(:obj:`dict[pandas.DataFrame]`): The dictionary of reanalysis dataframes.
        windspeed_col(:obj:`str`): The name of the column for the windspeed data to be plot.
        plot_por(:obj:`tuple[datetime.datetime, datetime.datetime]`): The start and end datetimes
            for a plant's period of record (POR).
        normalize(:obj:`bool`): Indicator of if the windspeeds shoudld be normalized (True), or not
            (False). Defaults to True.
        xlim (:obj:`tuple[datetime.datetime, datetime.datetime]`, optional): A tuple of datetimes
            representing the x-axis plotting display limits. Defaults to (None, None).
        ylim (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display limits.
            Defaults to (None, None).
        return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
        figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
            that are passed to `plt.figure()`. Defaults to {}.
        plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
            `ax.plot()`. Defaults to {}.
        legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
            `ax.legend()`. Defaults to {}.

    Returns:
        None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]: If `return_fig` is True, then
            the figure and axes objects are returned for further tinkering/saving.
    """
    # Define parameters needed for plotting
    min_val, max_val = (np.inf, -np.inf) if ylim == (None, None) else ylim

    figure_kwargs.setdefault("figsize", (14, 6))
    figure_kwargs.setdefault("dpi", 200)
    fig = plt.figure(**figure_kwargs)
    ax = fig.add_subplot(111)

    for name, df in data.items():
        # Compute the rolling mean and normalize it over a 12 month average
        ws = df.resample("MS")[windspeed_col].mean().to_frame().rolling(12).mean()
        if normalize:
            ws = ws[windspeed_col] / ws[windspeed_col].mean()

        # Update the min and max values
        min_val = min(min_val, ws.min())
        max_val = max(max_val, ws.max())

        ax.plot(ws, label=name, **plot_kwargs)

    # Plot a vertical line at y = 1
    _xlims = (ws.index[0], ws.index[-1]) if xlim is None else xlim
    ax.hlines(1, *_xlims, colors="k", linestyles="--")

    # Fill in the period of record
    ax.fill_between(
        plant_por,
        [min_val, min_val],
        [max_val, max_val],
        alpha=0.1,
        label="Plant POR",
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized wind speed")
    ax.legend(**legend_kwargs)

    fig.tight_layout()
    plt.show()

    if return_fig:
        return fig, ax


def plot_plant_energy_losses_timeseries(
    data: pd.DataFrame,
    energy_col: str,
    loss_cols: list[str],
    energy_label: str,
    loss_labels: list[str],
    xlim: tuple[datetime.datetime, datetime.datetime] = (None, None),
    ylim_energy: tuple[float, float] = (None, None),
    ylim_loss: tuple[float, float] = (None, None),
    return_fig: bool = False,
    figure_kwargs: dict = {},
    plot_kwargs: dict = {},
    legend_kwargs: dict = {},
):
    """
    Plot timeseries of energy, and the loss categories of interest.

    Args:
        data(:obj:`pandas.DataFrame`): A pandas DataFrame containing energy production and losses.
        energy_col(:obj:`str`): The name of the column in :py:attr:`data` containing the energy production.
        loss_cols(:obj:`list[str]`): The name(s) of the column(s) in :py:attr:`data` containing the loss data.
        energy_label(:obj:`str`): The legend label and y-axis label for the energy plot.
        loss_labels(:obj:`list[str]`): The legend labels losses plot.
        xlim (:obj:`tuple[datetime.datetime, datetime.datetime]`, optional): A tuple of datetimes
            representing the x-axis plotting display limits. Defaults to None.
        ylim_energy (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display
            limits for the gross energy plot (top figure). Defaults to None.
        ylim_loss (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display
            limits for the loss plot (bottom figure). Defaults to (None, None).
        return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
        figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
            that are passed to `plt.figure()`. Defaults to {}.
        plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
            `ax.scatter()`. Defaults to {}.
        legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
            `ax.legend()`. Defaults to {}.

    Returns:
        None | tuple[matplotlib.pyplot.Figure, tuple[matplotlib.pyplot.Axes, matplotlib.pyplot.Axes]]:
            If `return_fig` is True, then the figure and axes objects are returned for further
            tinkering/saving.
    """
    figure_kwargs.setdefault("figsize", (12, 9))
    figure_kwargs.setdefault("dpi", 200)
    fig = plt.figure(**figure_kwargs)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    axes = (ax1, ax2)

    # Plot the gross energy production
    ax1.plot(data[energy_col], ".-", label=energy_label, **plot_kwargs)
    ax1.set_xlabel("Year")
    ax1.set_ylabel(energy_label)

    # Joint availability and curtailment plot
    for col, label in zip(loss_cols, loss_labels):
        ax2.plot(data[col] * 100, ".-", label=label, **plot_kwargs)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Loss (%)")

    for ax in axes:
        ax.legend(**legend_kwargs)

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim_energy)
    ax2.set_ylim(ylim_loss)

    fig.tight_layout()
    plt.show()

    if return_fig:
        return fig, axes


def plot_distributions(
    data: pd.DataFrame,
    which: list[str],
    xlabels: list[str],
    xlim: tuple[tuple[float, float], ...] = None,
    ylim: tuple[tuple[float, float], ...] = None,
    return_fig: bool = False,
    figure_kwargs: dict = {},
    plot_kwargs: dict = {},
    annotate_kwargs: dict = {},
) -> None | tuple[plt.Figure, plt.Axes]:
    """
    Plot a distribution of AEP values from the Monte-Carlo OA method

    Args:
        aep(:obj:`pandas.DataFrame`): The pandas DataFrame of results data.
        which:(:obj:`list[str]`): The list of columns in data that should have their distributions plot.
        xlabels:(obj:`list[str]`): The list of x-axis labels
        xlim(:obj:`tuple[tuple[float, float], ...]`, optional): A tuple of tuples (or None)
            corresponding to each of elements of :py:attr:`which` that get passed to ax.set_xlim().
            Defaults to None.
        ylim(:obj:`tuple[tuple[float, float], ...]`, optional): A tuple of tuples (or None)
            corresponding to each of elements of :py:attr:`which` that get passed to ax.set_ylim().
            Defaults to None.
        return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
        figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
            that are passed to `plt.figure()`. Defaults to {}.
        plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
            `ax.hist()`. Defaults to {}.
        annotate_kwargs (:obj:`dict`, optional): Additional annotation keyword arguments that are
            passed to `ax.annotate()`. Defaults to {}.

    Returns:
        None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]: If `return_fig` is True, then
            the figure and axes objects are returned for further tinkering/saving.
    """
    if xlim is None:
        xlim = tuple([(None, None) for _ in range(len(which))])
    if ylim is None:
        ylim = tuple([(None, None) for _ in range(len(which))])

    if len(which) != len(xlabels) != len(xlim) != len(ylim):
        raise ValueError(
            "The inputs to `which`, `xlabels`, `xlim`, and `ylim` must be the same length."
        )

    annotate_kwargs.setdefault("fontsize", 12)

    figure_kwargs.setdefault("figsize", (14, 12))
    figure_kwargs.setdefault("dpi", 200)
    fig = plt.figure(**figure_kwargs)
    axes = fig.subplots(2, 2, gridspec_kw=dict(wspace=0.1, hspace=0.2))

    for ax, col, label, _xlim, _ylim in zip(axes.flatten(), which, xlabels, xlim, ylim):
        vals = data[col].values
        ax.hist(vals, 40, density=1, **plot_kwargs)
        ax.annotate(
            f"Mean = {vals.mean():.1f}",
            (0.05, 0.9),
            xycoords="axes fraction",
            **annotate_kwargs,
        )
        ax.annotate(
            f"Uncentainty = {vals.std() / vals.mean():.1f}",
            (0.05, 0.85),
            xycoords="axes fraction",
            **annotate_kwargs,
        )
        ax.set_xlabel(label)
        ax.set_xlim(_xlim)
        ax.set_ylim(_ylim)

    # Delete the extra axes
    if (n_delete := len(axes.flatten()) - len(which)) > 0:
        n = len(axes.flatten())
        for i in range(1, n_delete + 1):
            fig.delaxes(axes.flatten()[n - i])

    plt.show()

    if return_fig:
        return fig, axes


def _generate_swarm_values(y, n_bins=None, width: float = 0.5):
    """Create the x-coordiantes for `y` so that plotting each value in the distribution of `y` appears
    like that of a `seaborn.swarmplot` without requiring an additional dependency.

    Args:
        y (:obj:`pandas.Series`): The values to generate a matching x-value for a non-overlapping
            scatter plot of all the points in the distribution.
        n_bins (:obj:`int`, optional): The number of bins to use to generate the x-coordinates. If
            `None`, then it is `y.size // 6`. Defaults to None.
        width (:obj:`float`, optional): The maximum width of the x data in either
            direction. Defaults to 0.5.

    Returns:
        :obj:`numpy.ndarray` An array of x-coordinates to plot as a scatter against `y`.
    """
    if n_bins is None:
        n_bins = y.size // 6

    # Get the upper bound of each bin
    x = np.zeros_like(y)
    y_min, y_max = y.min(), y.max()
    dy = (y_max - y_min) / n_bins
    y_bins = np.linspace(y_min + dy, y_max - dy, n_bins - 1)

    # Divide the indices into their appropriate bins
    i = np.arange(y.size)
    ix_bin_groups = [0] * n_bins
    y_bin_groups = [0] * n_bins
    n_max = 0
    for j, y_bin in enumerate(y_bins):
        ix_bin = y <= y_bin
        ix_bin_groups[j], y_bin_groups[j] = i[ix_bin], y[ix_bin]
        n_max = max(n_max, len(ix_bin_groups[j]))
        i, y = i[~ix_bin], y[~ix_bin]

    # Fill in the last bin grouping values
    ix_bin_groups[-1], y_bin_groups[-1] = i, y
    n_max = max(n_max, len(ix_bin_groups[-1]))

    # Assign the x indices in alternating fashion for each bin to ensure the x values are roughly symmetric
    dx = 1 / (n_max // 2)
    for i, vals in zip(ix_bin_groups, y_bin_groups):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(vals)]
            a = i[j::2]
            b = i[j + 1 :: 2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx * width
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx * width

    return x


def plot_boxplot(
    x: pd.Series,
    y: pd.Series,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float | None, float | None] = (None, None),
    with_points: bool = False,
    points_label: str | None = None,
    return_fig: bool = False,
    figure_kwargs: dict = {},
    plot_kwargs_box: dict = {},
    plot_kwargs_points: dict = {},
    legend_kwargs: dict = {},
) -> None | tuple[plt.Figure, plt.Axes]:
    """Plot box plots of AEP results sliced by a specified Monte Carlo parameter

    Args:
        x(:obj:`pandas.Series`): The data that splits the results in y.
        y(:obj:`pandas.Series`): The resulting data to be splity by x.
        xlabel(:obj:`str`): The x-axis label.
        ylabel(:obj:`str`): The y-axis label.
        ylim(:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display limits.
            Defaults to None.
        with_points(:obj:`bool`, optional): Flag to plot the individual points like a seaborn
            `swarmplot`. Defaults to False.
        points_label(:obj:`bool` | None, optional): Legend label for the points, if plotting.
            Defaults to None.
        return_fig(:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
        figure_kwargs(:obj:`dict`, optional): Additional figure instantiation keyword arguments
            that are passed to `plt.figure()`. Defaults to {}.
        plot_kwargs_box(:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
            `ax.boxplot()`. Defaults to {}.
        plot_kwargs_points(:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
            `ax.boxplot()`. Defaults to {}.
        legend_kwargs(:obj:`dict`, optional): Additional legend keyword arguments that are passed to
            `ax.legend()`. Defaults to {}.

    Returns:
        None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes, dict]: If `return_fig` is
            True, then the figure object, axes object, and a dictionary of the boxplot objects are
            returned for further tinkering/saving.
    """
    df = pd.DataFrame(data={"x": x, "y": y})
    figure_kwargs.setdefault("figsize", (8, 6))

    fig = plt.figure(**figure_kwargs)
    ax = fig.add_subplot(111)

    parameters = x.unique()
    parameters.sort()
    y_groups = [df.loc[df["x"] == el, "y"] for el in parameters]

    plot_kwargs_box.setdefault("labels", parameters)
    box_data = ax.boxplot(y_groups, **plot_kwargs_box)

    if with_points:
        widths = plot_kwargs_box.get("widths", np.full(len(y_groups), 0.5))
        plot_kwargs_points.setdefault("marker", "o")
        plot_kwargs_points.setdefault("facecolor", "none")
        plot_kwargs_points.setdefault("edgecolor", "green")
        plot_kwargs_points.setdefault("alpha", 0.5)
        for x_start, (_y, width) in enumerate(zip(y_groups, widths)):
            _x = _generate_swarm_values(_y, width=width * 0.9) + x_start + 1
            label = points_label if x_start == width.size - 1 else None
            ax.scatter(_x, _y, zorder=0, label=label, **plot_kwargs_points)

    handles, labels = [box_data["fliers"][0]], ["Outliers"]
    _handles, _labels = ax.get_legend_handles_labels()
    handles.extend(_handles)
    labels.extend(_labels)
    ax.legend(handles, labels, **legend_kwargs)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_ylim(ylim)

    fig.tight_layout()
    plt.show()

    if return_fig:
        return fig, ax, box_data


def plot_waterfall(
    data: list[float] | NDArrayFloat,
    index: list[str],
    ylabel: str | None = None,
    ylim: tuple[float, float] = (None, None),
    return_fig: bool = False,
    plot_kwargs: dict = {},
    figure_kwargs: dict = {},
) -> None | tuple:
    """
    Produce a waterfall plot showing the progression from the EYA estimates to the calculated OA
    estimates of AEP.

    Args:
        data(array-like): data to be used to create waterfall.
        index(:obj:`list`): List of string values to be used for x-axis labels, which should
            have one more value than the number of points in :py:attr:`data` to account for
            the calculated OA total.
        ylabel(:obj:`str`): The y-axis label. Defaults to None.
        ylim(:obj:`tuple[float | None, float | None]`): The y-axis minimum and maximum display
            range. Defaults to (None, None).
        return_fig(:obj:`bool`, optional): Set to True to return the figure and axes objects,
            otherwise set to False. Defaults to False.
        figure_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be
            passed to `plt.figure`. Defaults to {}.
        plot_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be
            passed to `ax.plot`. Defaults to {}.
        legend_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be
            passed to `ax.legend`. Defaults to {}.

    Returns:
        None | tuple[plt.Figure, plt.Axes]: If :py:attr:`return_fig`, then return the figure
            and axes objects in addition to showing the plot.
    """
    # Store data and create a bottom series to use for the waterfall
    plot_data = pd.DataFrame(data={"amount": data}, index=index[:-1])
    bottom = plot_data.amount.cumsum().shift(1).fillna(0)

    # Get the net total number for the final element in the waterfall
    total = plot_data.sum().amount
    final_name = index[-1]
    plot_data.loc[final_name] = total
    bottom.loc[final_name] = 0

    # Set the defaults for plotting, if none were provided
    figure_kwargs.setdefault("figsize", (12, 6))
    plot_kwargs.setdefault("width", 0.8)
    width = plot_kwargs["width"]

    # Create the figure and axis
    fig = plt.figure(**figure_kwargs)
    ax = fig.add_subplot(111)

    # Plot the bar chart with vertical waterfall lines
    x = np.arange(plot_data.shape[0])
    ax.bar(x, plot_data.amount, bottom=bottom, **plot_kwargs)
    ax.hlines(
        bottom[1:-1].tolist() + [total],
        xmin=x[:-1] - width / 2.0,
        xmax=x[:-1] + 1 + width / 2.0,
        colors="tab:orange",
    )

    # Add the annotations above/below each bar with a +/- label on difference for each category
    offset_pos = plot_data.amount.max() * 0.05
    offset_neg = plot_data.amount.max() * 0.09
    for i, (y, diff) in enumerate(zip(bottom.values, plot_data.amount.values)):
        if i in (0, len(x) - 1):
            continue
        if np.sign(diff) == 1:
            y += diff + offset_pos
        else:
            y += diff - offset_neg
        ax.annotate(f"{diff:+,.1f}", (i, y), ha="center")

    # Add the styling and labeling, as specified by the user
    ax.set_xticks(x)
    ax.set_xticklabels(index)

    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    plt.show()
    if return_fig:
        return fig, ax


def plot_power_curves(
    data: dict[str, pd.DataFrame],
    power_col: str,
    windspeed_col: str,
    flag_col: str = None,
    turbines: list[str] | None = None,
    flag_labels: tuple[str, str] = ("Flagged Readings", "Power Curve"),
    max_cols: int = 3,
    xlim: tuple[float, float] = (None, None),
    ylim: tuple[float, float] = (None, None),
    legend: bool = False,
    return_fig: bool = False,
    figure_kwargs: dict = {},
    legend_kwargs: dict = {},
    plot_kwargs: dict = {},
):
    """Plots a series of power curves for a dictionary of turbine data, allowing for an optional
    filtering for singling out readings in the figure.

    Args:
        data(:obj:`dict[str, pd.DataFrame]`): The dictionary of turbine IDs and and SCADA data.
        wind_speed_col(:obj:`pandas.Series`): A pandas Series or numpy array of the recorded wind
            speeds, in m/s.
        power_col(:obj:`pandas.Series` | :obj:`np.ndarray`): A pandas Series or numpy array of the
            recorded power, in kW.
        flag_col(:obj:`np.ndarray` | :obj:`pd.Series`): A pandas Series or numpy array of booleans for
            which points to flag in the windspeed and power data.
        turbines(:obj:`list[str]`, optional): The list of turbines to be plot, if not all of the
            keys in :py:attr:`data`.
        flag_labels (:obj:`tuple[str, str]`, optional): The labels to give to the scatter points,
            corresponding to the flagged readings and raw readings, respectively. Defaults to
            ("Flagged Readings", "Power Curve").
        max_cols(:obj:`int`, optional): The maximum number of columns in the plot. Defaults to 3.
        xlim(:obj:`tuple[float, float]`, optional): A tuple of the x-axis (min, max) values.
            Defaults to (None, None).
        ylim(:obj:`tuple[float, float]`, optional): A tuple of the y-axis (min, max) values.
            Defaults to (None, None).
        legend(:obj:`bool`, optional): Set to True to place a legend in the figure, otherwise set
            to False. Defaults to False.
        return_fig(:obj:`bool`, optional): Set to True to return the figure and axes objects,
            otherwise set to False. Defaults to False.
        figure_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be passed to
            `plt.figure`. Defaults to {}.
        plot_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be passed
            to `ax.scatter`. Defaults to {}.
        legend_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be passed to
            `ax.legend`. Defaults to {}.

    Returns:
        None | tuple[plt.Figure, plt.Axes]: Returns the figure and axes objects if
            :py:attr:`return_fig` is True.
    """
    turbines = list(data.keys()) if turbines is None else turbines
    num_cols = len(turbines)
    num_rows = int(np.ceil(num_cols / max_cols))

    figure_kwargs.setdefault("dpi", 200)
    figure_kwargs.setdefault("figsize", (15, num_rows * 5))
    fig, axes_list = plt.subplots(num_rows, max_cols, **figure_kwargs)

    for i, (t, ax) in enumerate(zip(turbines, axes_list.flatten())):
        plot_data = data[t]

        label = "Power Curve" if flag_labels is None else flag_labels[1]
        ax.scatter(plot_data[windspeed_col], plot_data[power_col], label=label, **plot_kwargs)

        if flag_col is not None:
            plot_data = plot_data.loc[plot_data[flag_col]]
            label = "Flagged Readings" if flag_labels is None else flag_labels[0]
            ax.scatter(plot_data[windspeed_col], plot_data[power_col], label=label, **plot_kwargs)

        ax.set_title(t)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

        if legend:
            ax.legend(**legend_kwargs)

        if i % max_cols == 0:
            ax.set_ylabel("Power (kW)")

        if i in range(max_cols * (num_rows - 1), num_cols):
            ax.set_xlabel("Wind Speed (m/s)")

    num_axes = axes_list.size
    if i < num_axes - 1:
        for j in range(i + 1, num_axes):
            fig.delaxes(axes_list.flatten()[j])

    fig.tight_layout()
    plt.show()
    if return_fig:
        return fig, ax


def plot_wake_losses(
    bins: NDArrayFloat,
    efficiency_data_por: NDArrayFloat,
    efficiency_data_lt: NDArrayFloat,
    energy_data_por: NDArrayFloat = None,
    energy_data_lt: NDArrayFloat = None,
    bin_axis_label: str = "wd",
    turbine_id: str = None,
    xlim: tuple[float, float] = (None, None),
    ylim_efficiency: tuple[float, float] = (None, None),
    ylim_energy: tuple[float, float] = (None, None),
    return_fig: bool = False,
    figure_kwargs: dict = None,
    plot_kwargs_line: dict = {},
    plot_kwargs_fill: dict = {},
    legend_kwargs: dict = {},
):
    """Plots wake losses in the form of wind farm efficiency as well as normalized wind plant energy
    production for both the period of record and with the long-term correction as a function of either
    wind direction or wind speed. If the data arguments contain two dimensions, 95% confidence intervals
    will be plotted for each variable.

    Args:
        bins (:obj:`np.ndarray`): Wind direction or wind speed bin values representing the x-axis in
            the plots.
        efficiency_data_por (:obj:`np.ndarray`): 1D or 2D array containing wind farm or wind turbine
            efficiency for the period of record for each bin in the `bins` argument. If a 2D array is
            provided, the second dimension should contain results from different Monte Carlo iterations
            and 95% confidence intervals will be plotted.
        efficiency_data_lt (:obj:`np.ndarray`): 1D or 2D array containing long-term corrected wind farm
            or wind turbine efficiency for each bin in the `bins` argument. If a 2D array is provided,
            the second dimension should contain results from different Monte Carlo iterations and 95%
            confidence intervals will be plotted.
        energy_data_por (:obj:`np.ndarray`, optional): Optional 1D or 2D array containing normalized
            energy production for the period of record for each bin in the `bins` argument. If a 2D
            array is provided, the second dimension should contain results from different Monte Carlo
            iterations and 95% confidence intervals will be plotted. If a value of None is provided,
            normalized energy will not be plotted. Defaults to None.
        energy_data_lt (:obj:`np.ndarray`, optional): Optional 1D or 2D array containing normalized
            long-term corrected energy production for each bin in the `bins` argument. If a 2D array
            is provided, the second dimension should contain results from different Monte Carlo
            iterations and 95% confidence intervals will be plotted. If a value of None is provided,
            normalized energy will not be plotted. Defaults to None.
        bin_axis_label (str, optional): The label to use for the bin variable (x) axis. Defaults to None.
        turbine_id (str, optional): Name of turbine if data are provided for a single wind turbine.
            Used to determine title and plot axis labels. Defaults to None.
        xlim (:obj:`tuple[float, float]`, optional): A tuple of floats representing the x-axis
            wind direction plotting display limits (degrees). Defaults to (None, None).
        ylim_efficiency (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display
            limits for the wind farm efficiency plot (top plot). Defaults to (None, None).
        ylim_energy (:obj:`tuple[float, float]`, optional): If `energy_data_por` and `energy_data_lt`
            arguments are provided, a tuple of the y-axis plotting display limits for the wind farm
            energy distribution plot (bottom plot). Defaults to (None, None).
        return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
        figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
            that are passed to `plt.figure()`. Defaults to None.
        plot_kwargs_line (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
            `ax.plot()` for plotting lines for the wind farm efficiency and, if `energy_data_por` and
            `energy_data_lt` arguments are provided, energy distributions subplots. Defaults to {}.
        plot_kwargs_fill (:obj:`dict`, optional): If `UQ` is True, additional plotting keyword arguments
            that are passed to `ax.fill_between()` for plotting shading regions for 95% confidence
            intervals for the wind farm efficiency and, if `energy_data_por` and `energy_data_lt` arguments
            are provided, energy distributions subplots. Defaults to {}.
        legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
            `ax.legend()` for the wind farm efficiency and, if `energy_data_por` and `energy_data_lt`
            arguments are provided, energy distributions subplots. Defaults to {}.
    Returns:
        None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes] | tuple[matplotlib.pyplot.Figure, tuple [matplotlib.pyplot.Axes, matplotlib.pyplot.Axes]]:
            If `return_fig` is True, then the figure and axes object(s), corresponding to the wake
            loss plot or, if `energy_data_por` and `energy_data_lt` arguments are provided, wake loss
            and normalized energy plots, are returned for further tinkering/saving.
    """
    color_codes = ["#4477AA", "#228833"]

    plot_kwargs_fill.setdefault("alpha", 0.2)

    if xlim == (None, None):
        xlim = (bins[0], bins[-1])

    if figure_kwargs is None:
        figure_kwargs = {}

    # determine if confidence intervals should be plotted (i.e., UQ) based on dimension of data
    if (efficiency_data_por.ndim == 1) & (efficiency_data_lt.ndim == 1):
        UQ = False
    elif (efficiency_data_por.ndim == 2) & (efficiency_data_lt.ndim == 2):
        UQ = True
    else:
        raise ValueError(
            "The inputs `efficiency_data_por` and `efficiency_data_por` must have the same dimensions."
        )

    # determine if normalized energy should be plotted
    if (energy_data_por is not None) & (energy_data_lt is not None):
        if (not UQ) & (energy_data_por.ndim == 1) & (energy_data_lt.ndim == 1):
            plot_norm_energy = True
        elif UQ & (energy_data_por.ndim == 2) & (energy_data_lt.ndim == 2):
            plot_norm_energy = True
        else:
            raise ValueError(
                (
                    "The inputs `energy_data_por` and `energy_data_lt` must both have the same dimensions"
                    "as `efficiency_data_por` and `efficiency_data_lt`."
                )
        )
    elif (energy_data_por is None) & (energy_data_lt is None):
        plot_norm_energy = False
    else:
        raise TypeError(
            "The inputs `energy_data_por` and `energy_data_lt` must either both be provided or both be None."
        )

    if plot_norm_energy:
        figure_kwargs.setdefault("figsize", (9, 9.1))
        fig = plt.figure(**figure_kwargs)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        axs = (ax1, ax2)
    else:
        figure_kwargs.setdefault("figsize", (9, 5))
        fig = plt.figure(**figure_kwargs)
        ax1 = fig.add_subplot(111)
        axs = [ax1]
    axs[0].plot(xlim, [1, 1], "k", linewidth=1.5)

    if UQ:
        axs[0].plot(
            bins,
            np.mean(efficiency_data_por, axis=0),
            color=color_codes[0],
            label="Period of Record",
            **plot_kwargs_line,
        )
        axs[0].fill_between(
            bins,
            np.percentile(efficiency_data_por, 2.5, axis=0),
            np.percentile(efficiency_data_por, 97.5, axis=0),
            color=color_codes[0],
            label="_nolegend_",
            **plot_kwargs_fill,
        )

        axs[0].plot(
            bins,
            np.mean(efficiency_data_lt, axis=0),
            color=color_codes[1],
            label="Long-Term Corrected",
            **plot_kwargs_line,
        )
        axs[0].fill_between(
            bins,
            np.percentile(efficiency_data_lt, 2.5, axis=0),
            np.percentile(efficiency_data_lt, 97.5, axis=0),
            color=color_codes[1],
            label="_nolegend_",
            **plot_kwargs_fill,
        )

        if plot_norm_energy:
            axs[1].plot(
                bins,
                np.mean(energy_data_por, axis=0),
                color=color_codes[0],
                label="Period of Record",
                **plot_kwargs_line,
            )
            axs[1].fill_between(
                bins,
                np.percentile(energy_data_por, 2.5, axis=0),
                np.percentile(energy_data_por, 97.5, axis=0),
                color=color_codes[0],
                label="_nolegend_",
                **plot_kwargs_fill,
            )

            axs[1].plot(
                bins,
                np.mean(energy_data_lt, axis=0),
                color=color_codes[1],
                label="Long-Term Corrected",
                **plot_kwargs_line,
            )
            axs[1].fill_between(
                bins,
                np.percentile(energy_data_lt, 2.5, axis=0),
                np.percentile(energy_data_lt, 97.5, axis=0),
                color=color_codes[1],
                label="_nolegend_",
                **plot_kwargs_fill,
            )

    else:  # without UQ
        axs[0].plot(
            bins,
            efficiency_data_por,
            color=color_codes[0],
            label="Period of Record",
            **plot_kwargs_line,
        )

        axs[0].plot(
            bins,
            efficiency_data_lt,
            color=color_codes[1],
            label="Long-Term Corrected",
            **plot_kwargs_line,
        )

        if plot_norm_energy:
            axs[1].plot(
                bins,
                energy_data_por,
                color=color_codes[0],
                label="Period of Record",
                **plot_kwargs_line,
            )

            axs[1].plot(
                bins,
                energy_data_lt,
                color=color_codes[1],
                label="Long-Term Corrected",
                **plot_kwargs_line,
            )

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim_efficiency)
    axs[len(axs) - 1].set_xlabel(bin_axis_label)
    axs[0].legend(**legend_kwargs)
    if turbine_id is not None:
        axs[0].set_title(f"Wind Turbine {turbine_id}")
        axs[0].set_ylabel("Wind Turbine Efficiency (-)")
    else:
        axs[0].set_ylabel("Wind Plant Efficiency (-)")

    if plot_norm_energy:
        axs[1].set_ylim(ylim_energy)
        axs[1].legend(**legend_kwargs)
        axs[1].set_ylabel("Normalized Wind Plant\nEnergy Production (-)")

        plt.tight_layout()
        if return_fig:
            return fig, axs
    else:
        plt.tight_layout()
        if return_fig:
            return fig, ax1
