"""
This module provides helpful functions for creating various plots

"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import matplotlib
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pyproj import Transformer
from bokeh.models import WMTSTileSource, ColumnDataSource
from bokeh.palettes import Category10, viridis
from bokeh.plotting import figure

from openoa.utils import filters


plt.close("all")
font = {"family": "serif", "size": 14}

matplotlib.rc("font", **font)
matplotlib.rc("text", usetex=False)
matplotlib.rcParams["figure.figsize"] = (15, 6)


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

    rgb = matplotlib.colors.to_rgb(color)

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

    # TODO: UPDATE THIS DOCSTRING
    Example:
        .. bokeh-plot::

            import pandas as pd

            from bokeh.plotting import figure, output_file, show

            from openoa.toolkits.pandas_plotting import plot_windfarm
            from openoa.types import PlantData

            from examples.project_ENGIE import Project_Engie

            # Load plant object
            project = Project_Engie("../examples/data/la_haute_borne")

            # Prepare data
            project.prepare()

            # Create the bokeh wind farm plot
            show(plot_windfarm(project,tile_name="ESRI",plot_width=600,plot_height=600))
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

    # Create the bokeh data source
    source = ColumnDataSource(asset_df)

    # Create a bokeh figure with tiles
    plot_map = figure(
        plot_width=plot_width,
        plot_height=plot_height,
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
        id_col(:obj:`String`): The id column (or index column) in `df`.
        x_axis(:obj:'String'): Independent variable to plot, should align with a column in `df`.
        y_axis(:obj:'String'): Dependent variable to plot, should align with a column in `df`.
        return_fig(:obj:`String`): Indicator for if the figure and axes objects should be returned,
            by default False.

    Returns:
        (:obj: `None`)
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

        # Add a grid as the bottom layer
        ax.grid()
        ax.set_axisbelow(True)

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
    """Produces a histogram plot for each numeric column in `df`s.

    Args:
        df(:obj:`pd.DataFrame`): The dataframe for plotting.
        return_fig(:obj:`String`): Indicator for if the figure and axes objects should be returned,
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

        # Add a grid as the bottom layer
        ax.grid()
        ax.set_axisbelow(True)

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
    flag_labels: tuple[str, str] = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
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
        wind_speed (:obj: `pandas.Series`): A pandas Series or numpy array of the recorded wind speeds, in m/s.
        power (:obj: `pandas.Series` | `np.ndarray`): A pandas Series or numpy array of the recorded power, in kW.
        flag (:obj: `np.ndarray` | `pd.Series`): A pandas Series or numpy array of booleans for which points to flag in the windspeed and power data.
        flag_labels (:obj: `tuple[str, str]`, optional): The labels to give to the scatter points, where the 0th entry is the flagged points, and the second entry correpsponds to the standard power curve. Defaults to None.
        xlim (:obj: `tuple[float, float]`, optional): A tuple of the x-axis (min, max) values. Defaults to None.
        ylim (:obj: `tuple[float, float]`, optional): A tuple of the y-axis (min, max) values. Defaults to None.
        legend (:obj:`bool`, optional): Set to True to place a legend in the figure, otherwise set to False. Defaults to False.
        return_fig (:obj:`bool`, optional): Set to True to return the figure and axes objects, otherwise set to False. Defaults to False.
        figure_kwargs (:obj:`dict`, optional): Additional keyword arguments that should be passed to `plt.figure`. Defaults to {}.
        scatter_kwargs (:obj:`dict`, optional): Additional keyword arguments that should be passed to `ax.scatter`. Defaults to {}.
        legend_kwargs (:obj:`dict`, optional): Additional keyword arguments that should be passed to `ax.legend`. Defaults to {}.

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

    if legend:
        ax.legend(**legend_kwargs)

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power (kW)")

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if return_fig:
        return fig, ax
    fig.tight_layout()


def plot_normalized_monthly_reanalysis_windspeed(
    reanalysis: dict[str, pd.DataFrame],
    por_start: pd.Timestamp,
    por_end: pd.Timestamp,
    xlim: tuple[datetime.datetime, datetime.datetime] = None,
    ylim: tuple[float, float] = None,
    return_fig: bool = False,
    figure_kwargs: dict = {},
    plot_kwargs: dict = {},
    legend_kwargs: dict = {},
):
    """Make a plot of the normalized annual average wind speeds from reanalysis data to show general
    trends for each, and highlighting the period of record for the plant data.

    Args:
        reanalysis (:obj:`dict[str, pandas.DataFrame]`): `PlantData.reanalysis` dictionary of reanalysis
            `DataFrame`s.
        por_start (:obj:`pandas.Timestamp`): The start of the period of record; this should be a valid
            datetime or pandas Timestamp object.
        por_end (:obj:`pandas.Timestamp`): The end of the period of record; this should be a valid
            datetime or pandas Timestamp object.
        xlim (:obj:`tuple[datetime.datetime, datetime.datetime]`, optional): A tuple of datetimes
            representing the x-axis plotting display limits. Defaults to None.
        ylim (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display limits.
            Defaults to None.
        return_fig (:obj:`bool`, optional): _description_. Defaults to False.
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
    min_val, max_val = (np.inf, -np.inf) if ylim is None else ylim

    figure_kwargs.setdefault("figsize", (14, 6))
    fig = plt.figure(**figure_kwargs)
    ax = fig.add_subplot(111)

    for name, df in reanalysis.items():
        # Compute the rolling mean and normalize it over a 12 month average
        ws = df.resample("MS")["ws_dens_corr"].mean().to_frame().rolling(12).mean()
        ws_norm = ws["ws_dens_corr"] / ws["ws_dens_corr"].mean()

        # Update the min and max values
        min_val = min(min_val, ws_norm.min())
        max_val = max(max_val, ws_norm.max())

        ax.plot(ws_norm, label=name, **plot_kwargs)

    # Plot a vertical line at y = 1
    _xlims = (ws.index[0], ws.index[-1]) if xlim is None else xlim
    ax.hlines(1, *_xlims, colors="k", linestyles="--")

    # Fill in the period of record
    ax.fill_between(
        [por_start, por_end],
        [min_val, min_val],
        [max_val, max_val],
        alpha=0.1,
        label="Plant POR",
    )

    ax.grid()
    ax.set_axisbelow(True)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized wind speed")
    ax.legend(**legend_kwargs)

    if return_fig:
        return fig, ax

    fig.tight_layout()


# TODO


def plot_reanalysis_gross_energy_data(
    self,
    outlier_thres,
    xlim: tuple[datetime.datetime, datetime.datetime] = None,
    ylim: tuple[float, float] = None,
    return_fig: bool = False,
    figure_kwargs: dict = {},
    plot_kwargs: dict = {},
    legend_kwargs: dict = {},
):
    """
    Make a plot of gross energy vs wind speed for each reanalysis product,
    with outliers highlighted

    Args:
        outlier_thres (float): outlier threshold (typical range of 1 to 4) which adjusts outlier sensitivity detection

    Returns:
        matplotlib.pyplot object
    """

    figure_kwargs.setdefault("figsize", (9, 9))
    fig = plt.figure(**figure_kwargs)
    ax = fig.add_subplot(111)

    valid_aggregate = self._aggregate
    plt.figure(figsize=(9, 9))

    # Loop through each reanalysis product and make a scatterplot of monthly wind speed vs plant energy
    for p in np.arange(0, len(list(self._reanal_products))):
        col_name = self._reanal_products[p]  # Reanalysis column in monthly data frame
        # Plot
        plt.subplot(2, 2, p + 1)

        if (
            self.time_resolution == "M"
        ):  # Monthly case: apply robust linear regression for outliers detection
            x = sm.add_constant(
                valid_aggregate[col_name]
            )  # Define 'x'-values (constant needed for regression function)
            y = (
                valid_aggregate["gross_energy_gwh"] * 30 / valid_aggregate["num_days_expected"]
            )  # Normalize energy data to 30-days

            rlm = sm.RLM(
                y, x, M=sm.robust.norms.HuberT(t=outlier_thres)
            )  # Robust linear regression with HuberT algorithm (threshold equal to outlier_thres)
            rlm_results = rlm.fit()

            r2 = np.corrcoef(
                x.loc[rlm_results.weights == 1, col_name], y[rlm_results.weights == 1]
            )[
                0, 1
            ]  # Get R2 from valid data

            # Continue plotting
            plt.plot(
                x.loc[rlm_results.weights != 1, col_name],
                y[rlm_results.weights != 1],
                "rx",
                label="Outlier",
            )
            plt.plot(
                x.loc[rlm_results.weights == 1, col_name],
                y[rlm_results.weights == 1],
                ".",
                label="Valid data",
            )
            plt.title(col_name + ", R2=" + str(np.round(r2, 3)))
            plt.ylabel("30-day normalized gross energy (GWh)")

        else:  # Daily/hourly case: apply bin filter for outliers detection
            x = valid_aggregate[col_name]
            y = valid_aggregate["gross_energy_gwh"]
            plant_capac = self._plant.metadata.capacity / 1000.0 * self._hours_in_res

            # Apply bin filter
            flag = filters.bin_filter(
                bin_col=y,
                value_col=x,
                bin_width=0.06 * plant_capac,
                threshold=outlier_thres,  # wind bin threshold (stdev outside the median)
                center_type="median",
                bin_min=0.01 * plant_capac,
                bin_max=0.85 * plant_capac,
                threshold_type="std",
                direction="all",  # both left and right (from the median)
            )

            # Continue plotting
            plt.plot(
                x.loc[flag],
                y[flag],
                "rx",
                label="Outlier",
            )
            plt.plot(
                x.loc[~flag],
                y[~flag],
                ".",
                label="Valid data",
            )

            if self.time_resolution == "D":
                plt.ylabel("Daily gross energy (GWh)")
            elif self.time_resolution == "H":
                plt.ylabel("Hourly gross energy (GWh)")
            plt.title(col_name)

        plt.xlabel("Wind speed (m/s)")

    plt.tight_layout()
    return plt


def plot_result_aep_distributions(self):
    """
    Plot a distribution of AEP values from the Monte-Carlo OA method

    Returns:
        matplotlib.pyplot object
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 12))

    sim_results = self.results

    ax = fig.add_subplot(2, 2, 1)
    ax.hist(sim_results["aep_GWh"], 40, density=1)
    ax.text(
        0.05,
        0.9,
        "AEP mean = " + str(np.round(sim_results["aep_GWh"].mean(), 1)) + " GWh/yr",
        transform=ax.transAxes,
    )
    ax.text(
        0.05,
        0.8,
        "AEP unc = "
        + str(np.round(sim_results["aep_GWh"].std() / sim_results["aep_GWh"].mean() * 100, 1))
        + "%",
        transform=ax.transAxes,
    )
    plt.xlabel("AEP (GWh/yr)")

    ax = fig.add_subplot(2, 2, 2)
    ax.hist(sim_results["avail_pct"] * 100, 40, density=1)
    ax.text(
        0.05,
        0.9,
        "Mean = " + str(np.round((sim_results["avail_pct"].mean()) * 100, 1)) + " %",
        transform=ax.transAxes,
    )
    plt.xlabel("Availability Loss (%)")

    ax = fig.add_subplot(2, 2, 3)
    ax.hist(sim_results["curt_pct"] * 100, 40, density=1)
    ax.text(
        0.05,
        0.9,
        "Mean: " + str(np.round((sim_results["curt_pct"].mean()) * 100, 2)) + " %",
        transform=ax.transAxes,
    )
    plt.xlabel("Curtailment Loss (%)")
    plt.tight_layout()
    return plt


def plot_aep_boxplot(self, param, lab):
    """
    Plot box plots of AEP results sliced by a specified Monte Carlo parameter

    Args:
        param(:obj:`list`): The Monte Carlo parameter on which to split the AEP results
        lab(:obj:`str`): The name to use for the parameter when producing the figure

    Returns:
        (none)
    """

    import matplotlib.pyplot as plt

    sim_results = self.results

    tmp_df = pd.DataFrame(data={"aep": sim_results.aep_GWh, "param": param})
    tmp_df.boxplot(column="aep", by="param", figsize=(8, 6))
    plt.ylabel("AEP (GWh/yr)")
    plt.xlabel(lab)
    plt.title("AEP estimates by %s" % lab)
    plt.suptitle("")
    plt.tight_layout()
    return plt


def plot_aggregate_plant_data_timeseries(self):
    """
    Plot timeseries of monthly/daily gross energy, availability and curtailment

    Returns:
        matplotlib.pyplot object
    """
    import matplotlib.pyplot as plt

    valid_aggregate = self._aggregate

    plt.figure(figsize=(12, 9))

    # Gross energy
    plt.subplot(2, 1, 1)
    plt.plot(valid_aggregate.gross_energy_gwh, ".-")
    plt.grid("on")
    plt.xlabel("Year")
    plt.ylabel("Gross energy (GWh)")

    # Availability and curtailment
    plt.subplot(2, 1, 2)
    plt.plot(valid_aggregate.availability_pct * 100, ".-", label="Availability")
    plt.plot(valid_aggregate.curtailment_pct * 100, ".-", label="Curtailment")
    plt.grid("on")
    plt.xlabel("Year")
    plt.ylabel("Loss (%)")
    plt.legend()

    plt.tight_layout()
    return plt
