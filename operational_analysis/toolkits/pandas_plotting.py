"""
This module provides helpful functions for creating various plots

"""
import matplotlib
import matplotlib.pyplot as plt
# Import required packages
import numpy as np

plt.close('all')
font = {'family': 'serif',
        'size': 14}

matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=False)
matplotlib.rcParams['figure.figsize'] = (15, 6)


def coordinateMapping(lon1, lat1, lon2, lat2):
    """ Map latitude and longitude to  local cartesian coordinates

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
    """  Plot locations of turbines and met towers, with labels, on latitude/longitude grid

    Args:
        project(:obj:`plant object`): project to be plotted

    Returns:
        (None)
    """
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    asset_groups = project.asset.df.groupby('type')

    turbines = asset_groups.get_group('turbine')
    X = turbines['longitude']
    Y = turbines['latitude']
    labels = turbines['id'].tolist()

    ax.scatter(X, Y, marker='o', color='k')
    for label, x, y in zip(labels, X, Y):
        ax.annotate(label, xy=(x, y), xytext=(-8, 5), textcoords='offset points', fontsize=6)

    towers = asset_groups.get_group('tower')
    X = towers['longitude']
    Y = towers['latitude']
    labels = towers['id'].tolist()

    ax.scatter(X, Y, marker='s', color='r')
    for label, x, y in zip(labels, X, Y):
        ax.annotate(label, xy=(x, y), xytext=(-8, -10), textcoords='offset points', fontsize=6, color='r')

    ax.set_xlabel('Longitude, [deg]')
    ax.set_ylabel('Latitude, [deg]')

    del X, Y, labels, x, y, label


def subplot_powerRose_array(project, turbine_ids,
                            shift=0, direction=1,
                            columns=None,
                            left_margin=0.1, bottom_margin=0.1,
                            gap_w_frac=0.2, gap_h_frac=0.2, aspect=1):
    """  Wrapper for powerRose_array plotting for multiple subplots

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
        rect = [left_margin + (sp_w_frac + gap_w_frac) * ic,
                bottom_margin + (sp_h_frac + gap_h_frac) * (rows - ir),
                sp_w_frac, sp_h_frac]

        powerRose_array(fig, rect, tid, shift, direction)


def powerRose_array(project, fig, rect, tid, shift=[0], direction=1):
    """  Plot power curve on polar coordinates overlaying plot of surrounding array (both local and further distance)

    Args:
        project(:obj:`plant object`): project to be plotted
        fig(:obj:`figure handle`): figure handle
        rect(:obj:`list of four scalars`): [left offset, bottom offset, width, height] as fractions of figure
        width/height
        tid(:obj:`string`): id of turbine to be plotted
        shift(:obj:`list of scalars`): number of degrees to rotate wind direction data, each plotted as new line
        direction(:obj:`-1, 1`): wind direction data measured clockwise (1) or counterclockwise (-1)

    Returns:
    """
    # the carthesian axis:
    ax_carthesian = fig.add_axes(rect, frameon=True)

    # plotting the line on the carthesian axis
    X0 = project.asset.df.loc[project.asset.df.id == tid, 'longitude']
    Y0 = project.asset.df.loc[project.asset.df.id == tid, 'latitude']
    XY = project.asset.df.apply(lambda r: coordinateMapping(X0, Y0, r['longitude'], r['latitude']), axis=1)
    X = XY.apply(lambda r: r[0])
    Y = XY.apply(lambda r: r[1])

    ax_carthesian.scatter(X, Y, s=200, c='black', marker='o')
    ax_carthesian.axis('equal')
    ax_carthesian.set_xlim([-900, 900])
    ax_carthesian.set_ylim([-900, 900])
    ax_carthesian.tick_params(axis='both', which='major', pad=40)
    ax_carthesian.spines['left'].set_visible(True)
    ax_carthesian.spines['left'].set_color('black')
    ax_carthesian.spines['bottom'].set_visible(True)
    ax_carthesian.spines['bottom'].set_color('black')
    ax_carthesian.set_title('Turbine %s' % (tid))

    # the second carthesian axis:
    ax_carthesian_2 = fig.add_axes(rect, frameon=False)

    # plotting the line on the carthesian axis
    ax_carthesian_2.scatter(X, Y, s=50, c='g', marker='x')
    ax_carthesian_2.axis('equal')
    ax_carthesian_2.set_xlim([-3000, 3000])
    ax_carthesian_2.set_ylim([-3000, 3000])
    ax_carthesian_2.tick_params(axis='y', which='major', pad=100, colors='green')
    ax_carthesian_2.tick_params(axis='x', which='major', pad=75, colors='green')
    ax_carthesian_2.yaxis.label.set_color('green')
    ax_carthesian_2.xaxis.label.set_color('green')
    ax_carthesian_2.spines['left'].set_visible(True)
    ax_carthesian_2.spines['left'].set_color('green')
    ax_carthesian_2.spines['bottom'].set_visible(True)
    ax_carthesian_2.spines['bottom'].set_color('green')

    # the polar axis:
    ax_polar = fig.add_axes(rect, polar=True, frameon=False)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)

    # the polar plot
    cm = plt.get_cmap('jet')
    ax_polar.set_color_cycle([cm(1. * i / len(shift)) for i in range(len(shift))])
    for i in range(len(shift)):
        ax_polar.plot((model_eval['winddirection'] * direction + shift[i]) * np.pi / 180, model_eval[tid],
                      linewidth=3.0, label=str(shift[i]) + ' deg')

    ntick = 3
    ticks = [np.round(np.min(model_eval[tid]) + ((np.max(model_eval[tid]) - np.min(model_eval[tid])) / ntick)
                      * (t + 1))
             for t in np.arange(ntick)]

    ax_polar.set_rmax(np.ceil(1.05 * np.max(model_eval[tid])))
    ax_polar.set_rmin(np.floor(0.95 * np.min(model_eval[tid])))

    ax_polar.set_rticks(ticks)  # less radial ticks
    ax_polar.grid(True)
    ax_polar.legend()


def subplt_c1_c2(turbine, axarr, c1, c2, c='Blues', xlim=None, ylim=None, xlabel=None, ylabel=None):
    """  hexbin plot of turbine[c1] vs turbine [c2]

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


def subplt_c1_c2_flagged(turbine, axarr, c1, c2, flag_cols, flag_value, cmap='Blues', xlim=None, ylim=None,
                         xlabel=None, ylabel=None):
    """  hexbin plot of turbine[c1] vs turbine [c2], showing only for which <flag_cols> have <value>

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
            flag_indices = np.append(flag_indices, turbine.loc[turbine[c] == flag_value].index.values)
        flag_indices = np.unique(flag_indices)

    hb = axarr.hexbin(turbine.loc[flag_indices, c1],
                      turbine.loc[flag_indices, c2],
                      cmap=cmap, gridsize=128, vmin=0, vmax=8)

    if xlim:
        axarr.set_xlim(xlim)
    if ylim:
        axarr.set_ylim(ylim)
    if xlabel:
        axarr.set_xlabel(xlabel)
    if ylabel:
        axarr.set_ylabel(ylabel)

    axarr.text(xlim[0] + 0.1 * (xlim[1] - xlim[0]), ylim[0] + 0.9 * (ylim[1] - ylim[0]),
               '%.2f%%' % (float(len(flag_indices)) / float(len(turbine)) * 100.))
    return hb


def subplt_c1_c2_raw_flagged(turbine, axarr, c1, c2, flag_cols, flag_value,
                             cmap='Blues', markers=['x'], colors=['r'],
                             xlim=None, ylim=None, xlabel=None, ylabel=None):
    """  hexbin plot of turbine[c1] vs turbine [c2], showing data <flag_cols> have <value> as overlaid scatter plot

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
            flag_indices = flag_indices + turbine.loc[turbine[c] == flag_value].index.values.tolist()
            flag_indices = np.unique(flag_indices)
        axarr.scatter(turbine.loc[flag_indices, c1],
                      turbine.loc[flag_indices, c2],
                      marker=markers[0], color=colors[0])
    else:
        for ic, c in enumerate(flag_cols):
            flag_indices = turbine.loc[turbine[c] == flag_value].index.values.tolist()
            axarr.scatter(turbine.loc[flag_indices, c1],
                          turbine.loc[flag_indices, c2],
                          marker=markers[ic], color=colors[ic])

    if xlim:
        axarr.set_xlim(xlim)
    if ylim:
        axarr.set_ylim(ylim)
    if xlabel:
        axarr.set_xlabel(xlabel)
    if ylabel:
        axarr.set_ylabel(ylabel)

    axarr.text(xlim[0] + 0.1 * (xlim[1] - xlim[0]), ylim[0] + 0.9 * (ylim[1] - ylim[0]),
               '%.2f%%' % (float(len(flag_indices)) / float(len(turbine)) * 100.))
    return hb


def subplt_power_curve(turbine, axarr, fig, c3, pc):
    hb = subplt_c1_c2_raw(turbine, axarr, fig, 'windspeed_ms', 'power_kw')

    turbine.sort_values(by=c3, inplace=True)
    axarr.plot(turbine[c3], turbine[pc], 'r', label='power curve')

    return hb


def turbine_polar_line(array, theta, r,
                       line_label, tid, color='b',
                       ax_carthesian=None, ax_polar=None):
    """  Polar plot (<r>, <theta>) overlaying plot of surrounding array, centered on turbine <tid>

    Args:
        array(:obj:`pandas dataframe`): index by (string) labels of assets, 'x' and 'y' coordinate columns
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
        fig = plt.figure(figsize=(10., 10.))
        rect = [0.1, 0.1, 0.8, 0.8]
        ax_carthesian = fig.add_axes(rect, frameon=True)

        # plotting the line on the carthesian axis
        x_offset = array.loc[tid, 'x']
        y_offset = array.loc[tid, 'y']

        X = array['x'] - x_offset
        Y = array['y'] - y_offset
        turbine_labels = array.index

        ax_carthesian.scatter(X, Y, marker='o', color='k', s=20)
        for turbine_label, x, y in zip(turbine_labels, X, Y):
            ax_carthesian.annotate(turbine_label, xy=(x, y), xytext=(-8, 5), textcoords='offset points')

        ax_carthesian.axis('equal')
        ax_carthesian.set_xlim([-1000, 1000])
        ax_carthesian.set_ylim([-1000, 1000])
        ax_carthesian.tick_params(axis='both', which='major', pad=70)
        ax_carthesian.spines['left'].set_visible(True)
        ax_carthesian.spines['left'].set_color('black')
        ax_carthesian.spines['bottom'].set_visible(True)
        ax_carthesian.spines['bottom'].set_color('black')

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
    ticks = [np.round(ax_polar.get_rmin() + ((ax_polar.get_rmax() - ax_polar.get_rmin()) / ntick) * (t + 1)) for t in
             np.arange(ntick)]

    ax_polar.set_rticks(ticks)  # less radial ticks
    ax_polar.grid(True)

    return ax_carthesian, ax_polar


def turbine_polar_4Dscatter(array, tid,
                            theta, r, color, size,
                            cmap='autumn_r'):
    """  Polar plot (<r>, <theta>) overlaying plot of surrounding array, centered on turbine <tid>

    Args:
        array(:obj:`pandas dataframe`): index by (string) labels of assets, 'x' and 'y' coordinate columns
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
    fig = plt.figure(figsize=(10., 10.))
    rect = [0.1, 0.1, 0.8, 0.8]
    ax_polar = fig.add_axes(rect, polar=True, frameon=False)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_rmax(np.ceil(1.05 * np.max(r)))
    ax_polar.set_rmin(np.floor(0.95 * np.min(r)))

    # the polar plot
    sc = ax_polar.scatter((theta) * np.pi / 180, r, s=size * 10, c=color, cmap=cmap)

    ntick = 3
    ticks = [np.round(ax_polar.get_rmin() + ((ax_polar.get_rmax() - ax_polar.get_rmin()) / ntick) * (t + 1)) for t in
             np.arange(ntick)]

    ax_polar.set_rticks(ticks)  # less radial ticks
    ax_polar.grid(True)
    ax_polar.legend()

    # the carthesian axis:
    ax_carthesian = fig.add_axes(rect, frameon=True)
    ax_carthesian.patch.set_alpha(0)
    # plotting the line on the carthesian axis
    x_offset = array.loc[tid, 'x']
    y_offset = array.loc[tid, 'y']

    X = array['x'] - x_offset
    Y = array['y'] - y_offset
    turbine_labels = array.index

    ax_carthesian.scatter(X, Y, marker='o', color='k', s=20)
    for turbine_label, x, y in zip(turbine_labels, X, Y):
        ax_carthesian.annotate(turbine_label, xy=(x, y), xytext=(-8, 5), textcoords='offset points')

        ax_carthesian.axis('equal')
        ax_carthesian.set_xlim([-1000, 1000])
        ax_carthesian.set_ylim([-1000, 1000])
        ax_carthesian.tick_params(axis='both', which='major', pad=70)
        ax_carthesian.spines['left'].set_visible(True)
        ax_carthesian.spines['left'].set_color('black')
        ax_carthesian.spines['bottom'].set_visible(True)
        ax_carthesian.spines['bottom'].set_color('black')
        ax_carthesian.set_title('Turbine %s' % (tid))

    box = ax_carthesian.get_position()

    # create color bar
    axColor = plt.axes([box.x0 * 1.1 + box.width * 1.1, box.y0, 0.01, box.height])
    plt.colorbar(sc, cax=axColor, orientation="vertical")

    return ax_carthesian, ax_polar


def turbine_polar_contourf(array, tid,
                           theta, r, c,
                           cmap='autumn_r'):
    """  Polar plot (<r>, <theta>) overlaying plot of surrounding array, centered on turbine <tid>

    Args:
        array(:obj:`pandas dataframe`): index by (string) labels of assets, 'x' and 'y' coordinate columns
        tid(:obj:`str`): index of asset on which to center carthesian axes
        theta(:obj:`pandas series, np array, list`): anglular coordinates of points, in degrees
        r(:obj:`pandas series, np array, list`): radial coordinates of points
        c(:obj:`pandas series, np array, list`): colors of points

    Returns:
        ax_carthesian(:obj:`axes handle`): carthesian axes on which array plotted
        ax_polar(:obj:`axes handle`): polar axes on which data plotted
    """

    # the polar axis:
    fig = plt.figure(figsize=(10., 10.))
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
    ticks = [np.round(ax_polar.get_rmin() + ((ax_polar.get_rmax() - ax_polar.get_rmin()) / ntick) * (t + 1)) for t in
             np.arange(ntick)]

    ax_polar.set_rticks(ticks)  # less radial ticks
    ax_polar.grid(True)
    ax_polar.legend()

    ax_carthesian = fig.add_axes(rect, frameon=True)
    ax_carthesian.patch.set_alpha(0)
    # plotting the line on the carthesian axis
    x_offset = array.loc[tid, 'x']
    y_offset = array.loc[tid, 'y']

    X = array['x'] - x_offset
    Y = array['y'] - y_offset
    turbine_labels = array.index

    ax_carthesian.scatter(X, Y, marker='o', color='k', s=20)
    for turbine_label, x, y in zip(turbine_labels, X, Y):
        ax_carthesian.annotate(turbine_label, xy=(x, y), xytext=(-8, 5), textcoords='offset points')

        ax_carthesian.axis('equal')
        ax_carthesian.set_xlim([-1000, 1000])
        ax_carthesian.set_ylim([-1000, 1000])
        ax_carthesian.tick_params(axis='both', which='major', pad=70)
        ax_carthesian.spines['left'].set_visible(True)
        ax_carthesian.spines['left'].set_color('black')
        ax_carthesian.spines['bottom'].set_visible(True)
        ax_carthesian.spines['bottom'].set_color('black')
        ax_carthesian.set_title('Turbine %s' % (tid))

    return ax_carthesian, ax_polar


def turbine_polar_contour(array, tid,
                          theta, r, z,
                          levels, colors,
                          ax_carthesian=None, ax_polar=None, label=''):
    """  Polar plot (<r>, <theta>) overlaying plot of surrounding array, centered on turbine <tid>

    Args:
        array(:obj:`pandas dataframe`): index by (string) labels of assets, 'x' and 'y' coordinate columns
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
        fig = plt.figure(figsize=(10., 10.))
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
    ticks = [np.round(ax_polar.get_rmin() + ((ax_polar.get_rmax() - ax_polar.get_rmin()) / ntick) * (t + 1)) for t in
             np.arange(ntick)]

    ax_polar.set_rticks(ticks)  # less radial ticks
    ax_polar.grid(True)

    if ax_carthesian is None:
        ax_carthesian = fig.add_axes(rect, frameon=True)
        ax_carthesian.patch.set_alpha(0)
        # plotting the line on the carthesian axis
        x_offset = array.loc[tid, 'x']
        y_offset = array.loc[tid, 'y']

        X = array['x'] - x_offset
        Y = array['y'] - y_offset
        turbine_labels = array.index

        ax_carthesian.scatter(X, Y, marker='o', color='k', s=20)
        for turbine_label, x, y in zip(turbine_labels, X, Y):
            ax_carthesian.annotate(turbine_label, xy=(x, y), xytext=(-8, 5), textcoords='offset points')

        ax_carthesian.axis('equal')
        ax_carthesian.set_xlim([-1000, 1000])
        ax_carthesian.set_ylim([-1000, 1000])
        ax_carthesian.tick_params(axis='both', which='major', pad=70)
        ax_carthesian.spines['left'].set_visible(True)
        ax_carthesian.spines['left'].set_color('black')
        ax_carthesian.spines['bottom'].set_visible(True)
        ax_carthesian.spines['bottom'].set_color('black')
        ax_carthesian.set_title('Turbine %s' % (tid))

    return ax_carthesian, ax_polar, artists, labels
