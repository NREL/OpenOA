"""
This module provides functions for flagging pandas data series based on a range of criteria. The functions are largely
intended for application in wind plant operational energy analysis, particularly wind speed vs. power curves.
"""

import numpy as np
import pandas as pd
import scipy as sp


def range_flag(data_col, below=-1. * np.inf, above=np.inf):
    """Flag data for which the specified data is outside a specified range

    Args:
        data_col (:obj:`pandas.Series`): data to be flagged
        below (:obj:`float`): upper threshold (inclusive) for data; default np.inf
        above (:obj:`float`): lower threshold (inclusive) for data; default -np.inf

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    flag = ~((data_col <= above) & (data_col >= below)) # Apply the range flag
    return flag  # Return boolean series of data flags


def unresponsive_flag(data_col, threshold=3):
    """Flag time stamps for which the reported data does not change for <threshold> repeated intervals.
    Function includes the option to group by a column in the data frame (e.g. turbine ID)

    Args:
        data_col(:obj:`pandas.Series`): data to be flagged
        threshold(:obj:`int`): number of intervals over which measurment does not change

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    # Get True/False value depending if the difference in successive time steps is not equal to zero
    value_diff = data_col.diff().ne(0)

    # take the rolling sum of the boolean diff column in period lengths defined by threshold
    roll_sum = value_diff.rolling(threshold - 1).sum()

    # Create boolean series that is True if rolling sum is zero
    flag_ind = (roll_sum == 0)

    # Need to flag preceding <threshold> -1 values as well
    for n in np.arange(threshold - 1):
        flag_ind = flag_ind | flag_ind.shift(-1)

    return flag_ind  # Return boolean series of data flags


def std_range_flag(data_col, threshold=2.):
    """Flag time stamps for which the measurement is outside of the threshold number of standard deviations from the
    mean across all passed columns; does not distinguish between asset ids

    Args:
        data_col(:obj:`pandas.Series`): data to be flagged
        threshold(:obj:`float`): multiplicative factor on standard deviation to use in flagging

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    data_mean = data_col.mean()  # Get mean of data
    data_std = data_col.std()  # Get std of data
    flag = ((data_col <= data_mean - threshold * data_std) | (
            data_col >= data_mean + threshold * data_std))  # Apply the range flag

    return flag


def window_range_flag(window_col, window_start, window_end, value_col, value_min, value_max):
    """Flag time stamps for which measurement in column <window> within range [window_start, window_end] and measurement
    in column <value> outside of range [value_min, value_max]

    Args:
        window_col(:obj:`pandas.Series`): data used to define the window
        window_start(:obj:`float`): minimum value for window
        window_end(:obj:`float`): maximum value for window
        value_col(:obj:`pandas.Series`): data to be flagged
        value_max(:obj:`float`): upper threshold for data; default np.inf
        value_min(:obj:`float`): lower threshold for data; default -np.inf

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    flag = ((window_col >= window_start) & (window_col <= window_end)
            & ((value_col < value_min) | (value_col > value_max)))

    return flag


def bin_filter(bin_col, value_col, bin_width, threshold=2, center_type='mean', bin_min=None, bin_max=None,
               threshold_type='std', direction='all'):
    """Flag time stamps for which data in <value_col> when binned by data in <bin_col> into bins of <width>
    is outside <threhsold> bin. The <center_type> of each bin can be either the median or mean, and flagging
    can be applied directionally (i.e. above or below the center, or both)

    Args:
        bin_col(:obj:`pandas.Series`): data to be used for binning
        value_col(:obj:`pandas.Series`): data to be flagged
        bin_width(:obj:`float`): width of bin in units of bin_col
        threshold(:obj:`float`): outlier threshold (multiplicative factor of std of <value_col> in bin)
        bin_min(:obj:`float`): minimum bin value below which flag should not be applied
        bin_max(:obj:`float`): maximum bin value above which flag should not be applied
        threshold_type(:obj:`str`): option to apply a 'std' or 'scalar' based threshold
        center_type(:obj:`str`): option to use a 'mean' or 'median' center for each bin
        direction(:obj:`str`): option to apply flag only to data 'above' or 'below' the mean, otherwise the default is
        'all'

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    # Set bin min and max values if not passed to function
    if bin_min is None:
        bin_min = bin_col.min()
    if bin_max is None:
        bin_max = bin_col.max()

    # Define bin edges
    bin_edges = np.arange(np.round(bin_min), np.round(bin_max), bin_width)

    # Ensure the last bin edge value is bin_max
    if bin_edges[-1] < bin_max:
        bin_edges = np.append(bin_edges, bin_max)
    elif bin_edges[-1] > bin_max:
        bin_edges[-1] = bin_max

    nbins = len(bin_edges) - 1  # Get number of bins

    # Define empty flag of 'False' values with indices matching value_col
    flag = pd.Series(index=value_col.index, data=False)

    # Loop through bins and applying flagging
    for n in np.arange(nbins):
        # Get data that fall wihtin bin
        y_bin = value_col.loc[(bin_col <= bin_edges[n + 1]) & (bin_col > bin_edges[n])]

        # Get center of binned data
        if center_type == 'mean':
            cent = y_bin.mean()
        elif center_type == 'median':
            cent = y_bin.median()
        else:
            print('incorrect center type specified')

        # Define threshold of data flag
        if threshold_type == 'std':
            ran = y_bin.std() * threshold
        elif threshold_type == 'scalar':
            ran = threshold

        # Perform flagging depending on specfied direction
        if direction == 'all':
            flag_bin = (y_bin > (cent + ran)) | (y_bin < (cent - ran))
        elif direction == 'above':
            flag_bin = y_bin > (cent + ran)
        elif direction == 'below':
            flag_bin = y_bin < (cent - ran)

        # Record flags in final flag column
        flag.loc[flag_bin.index] = flag_bin

    return flag


def cluster_mahalanobis_2d(data_col1, data_col2, n_clusters=13, dist_thresh=3.):
    """K-means clustering of  data into <n_cluster> clusters; Mahalanobis distance evaluated for each cluster and
    points with distances outside of <dist_thresh> are flagged; distinguishes between asset ids

    Args:
        data_col1(:obj:`pandas.Series`): first data column in 2D cluster analysis
        data_col2(:obj:`pandas.Series`): second data column in 2D cluster analysis
        n_clusters(:obj:`int`):' number of clusters to use
        dist_thresh(:obj:`float`): maximum Mahalanobis distance within each cluster for data to be remain unflagged

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    # Create 2D data frame for input into cluster algorithm
    df = pd.DataFrame({'d1': data_col1, 'd2': data_col2})

    # Run cluster algorithm
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters).fit(df)

    # Define empty flag of 'False' values with indices matching value_col
    flag = pd.Series(index=data_col1.index, data=False)

    # Loop through clusters and flag data that fall outside a threshold distance from cluster center
    for ic in range(n_clusters):
        # Extract data for cluster
        clust_sub = ((kmeans.labels_ == ic))
        cluster = df.loc[clust_sub]

        # Cluster centroid
        centroid = kmeans.cluster_centers_[ic]

        # Cluster covariance and inverse covariance
        covmx = cluster.cov()
        invcovmx = sp.linalg.inv(covmx)

        # Compute mahalnobis distance of each point in cluster
        mahalanobis_dist = cluster.apply(lambda r: sp.spatial.distance.mahalanobis(r.values, centroid, invcovmx),
                                         axis=1)

        # Flag data outside the distance threshold
        flag_bin = (mahalanobis_dist > dist_thresh)

        # Record flags in final flag column
        flag.loc[flag_bin.index] = flag_bin

    return flag
