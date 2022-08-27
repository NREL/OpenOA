"""
This module provides functions for flagging pandas data series based on a range of criteria. The functions are largely
intended for application in wind plant operational energy analysis, particularly wind speed vs. power curves.
"""

from __future__ import annotations

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.cluster import KMeans


def range_flag(
    data: pd.DataFrame | pd.Series,
    lower: float | list[float],
    upper: float | list[float],
    col: list[str] | None = None,
) -> pd.Series | pd.DataFrame:
    """Flag data for which the specified data is outside the provided range of [lower, upper].

    Args:
        data (:obj:`pandas.Series` | `pandas.DataFrame`): data frame containing the column to be flagged;
            can either be a `pandas.Series` or `pandas.DataFrame`. If a `pandas.DataFrame`, a list of
            threshold values and columns (if checking a subset of the columns) must be provided.
        col (:obj:`list[str]`): column(s) in `data` to be flagged, by default None. Only required when
            the `data` is a `pandas.DataFrame` and a subset of the columns will be checked. Must be
            the same length as `lower` and `upper`.
        lower (:obj:`float` | `list[float]`): lower threshold (inclusive) for each element of `data`,
            if it's a `pd.Series`, or the list of lower thresholds for each column in `col`. Must be
            the same length as `col` and `upper`.
        upper (:obj:`float` | `list[float]`): upper threshold (inclusive) for each element of `data`,
            if it's a `pd.Series`, or the list of upper thresholds for each column in `col`. Must be
            the same length as `lower` and `col`.

    Returns:
        :obj:`pandas.DataFrame(bool)`: Data frame with boolean entries.
    """
    # If data is `pd.Series`, convert the arguments appropriately
    if to_series := isinstance(data, pd.Series):
        data = data.to_frame()
        upper = [upper]
        lower = [lower]

    if col is None:
        col = data.columns.tolist()

    # Check for invalid inputs to col, upper, and lower
    if not isinstance(lower, list):
        raise ValueError("The input to `lower` must be a list of numbers.")

    if not isinstance(upper, list):
        raise ValueError("The input to `upper` must be a list of numbers.")

    if len(col) != len(lower) != len(upper):
        raise ValueError("The inputs to `col`, `above`, and `below` must be the same length.")

    # Only flag the desired columns
    subset = data.loc[:, col].copy()
    flag = ~(subset.ge(lower) & subset.le(upper))

    # Return back a pd.Series if one was provided, else a pd.DataFrame
    return flag[col[0]] if to_series else flag


def unresponsive_flag(data_col: pd.Series, threshold: int = 3) -> pd.Series:
    """Flag time stamps for which the reported data does not change for `threshold` repeated intervals.

    Args:
        data_col(:obj:`pandas.Series`): data to be flagged
        threshold(:obj:`int`): number of intervals over which measurment does not change, by default 3.

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    # Get boolean value of the difference in successive time steps is not equal to zero, and take the
    # rolling sum of the boolean diff column in period lengths defined by threshold
    flag = data_col.diff().ne(0).rolling(threshold - 1).sum()

    # Create boolean series that is True if rolling sum is zero
    flag = flag == 0

    # Need to flag preceding `threshold` values as well
    flag = flag | np.any([flag.shift(-1 - i) for i in range(threshold - 1)], axis=0)
    return flag


def unresponsive_flag_df(data: pd.DataFrame, col: list[str], threshold: list[int]) -> pd.Series:
    """Flag time stamps for which the reported data does not change for `threshold` repeated intervals.

    Args:
        data(:obj:`pandas.DataFrame`): data frame with column(s) to be flagged
        col(:obj:`list[str]`): data column(s) to be flagged
        threshold(:obj:`list[int]`): number of intervals over which measurment does not change.

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """
    if len(col) != len(threshold):
        raise ValueError("Inputs provide to `col` and `threshold` must be the same length.")
    subset = data.loc[:, col].copy()
    flag = subset.diff(axis=0).ne(0).rolling(threshold - 1).sum()
    flag = flag == 0
    flag = flag | np.any([flag.shift(-1 - i, axis=0) for i in range(threshold - 1)], axis=0)
    return flag


def std_range_flag(data_col: pd.Series, threshold: float = 2.0) -> pd.Series:
    """Flag time stamps for which the measurement is outside of the threshold number of standard deviations
     from the mean across the data.

    ... note:: This method does not distinguish between asset IDs.

    Args:
        data_col(:obj:`pandas.Series`): data to be flagged.
        threshold(:obj:`float`): multiplicative factor on standard deviation to use in flagging.

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    data_mean = data_col.mean()  # Get mean of data
    data_std = data_col.std() * threshold  # Get std of data
    flag = (data_col <= data_mean - data_std) | (data_col >= data_mean + data_std)
    return flag


def std_range_flag_df(data: pd.DataFrame, col: list[str], threshold: float = 2.0) -> pd.Series:
    """Flag time stamps for which the measurement is outside of the threshold number of standard deviations
     from the mean across the data.

    ... note:: This method does not distinguish between asset IDs.

    Args:
        data(:obj:`pandas.DataFrame`): data frame with column(s) to be flagged
        col(:obj:`list[str]`): data column(s) to be flagged
        threshold(:obj:`[float]`): multiplicative factor on standard deviation to use in flagging.

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """
    if len(col) != len(threshold):
        raise ValueError("Inputs provided to `col` and `threshold` must be the same length.")

    subset = data.loc[col].copy()
    data_mean = subset.mean(axis=0)  # Get mean of data
    data_std = subset.std(axis=0) * np.array(threshold)  # Get std of data
    flag = (subset <= data_mean - data_std) | (subset >= data_mean + data_std)
    return flag


def window_range_flag(
    window_col: pd.Series,
    window_start: float,
    window_end: float,
    value_col: pd.Series,
    value_min: float = -np.inf,
    value_max: float = np.inf,
) -> pd.Series:
    """Flag time stamps for which measurement in `window_col` are within the range: [`window_start`, `window_end`], and
    the measurements in `value_col` are outside of the range [`value_min`, `value_max`].

    Args:
        window_col(:obj:`pandas.Series`): data used to define the window
        window_start(:obj:`float`): minimum value for the inclusive window
        window_end(:obj:`float`): maximum value for the inclusive window
        value_col(:obj:`pandas.Series`): data to be flagged
        value_max(:obj:`float`): upper threshold for the inclusive data range; default np.inf
        value_min(:obj:`float`): lower threshold for the inclusive data range; default -np.inf

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    flag = (window_col >= window_start) & (window_col <= window_end)
    flag &= (value_col < value_min) | (value_col > value_max)
    return flag


def bin_filter(
    bin_col: pd.Series,
    value_col: pd.Series,
    bin_width: float,
    threshold: float = 2,
    center_type: str = "mean",
    bin_min: float = None,
    bin_max: float = None,
    threshold_type: str = "std",
    direction: str = "all",
):
    """Flag time stamps for which data in `value_col` when binned by data in `bin_col` into bins of
    width `bin_width` are outside the `threhsold` bin. The `center_type` of each bin can be either the
    median or mean, and flagging can be applied directionally (i.e. above or below the center, or both)

    Args:
        bin_col(:obj:`pandas.Series`): data to be used for binning
        value_col(:obj:`pandas.Series`): data to be flagged
        bin_width(:obj:`float`): width of bin in units of `bin_col`
        threshold(:obj:`float`): outlier threshold (multiplicative factor of std of `value_col` in bin)
        bin_min(:obj:`float`): minimum bin value below which flag should not be applied
        bin_max(:obj:`float`): maximum bin value above which flag should not be applied
        threshold_type(:obj:`str`): option to apply a 'std' or 'scalar' based threshold
        center_type(:obj:`str`): option to use a 'mean' or 'median' center for each bin
        direction(:obj:`str`): option to apply flag only to data 'above' or 'below' the mean, by default 'all'

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    if center_type not in ("mean", "median"):
        raise ValueError("Incorrect `center_type` specified; must be one of 'mean' or 'median'.")
    if threshold_type not in ("std", "scalar"):
        raise ValueError("Incorrect `threshold_type` specified; must be one of 'std' or 'scalar'.")
    if direction not in ("all", "above", "below"):
        raise ValueError(
            "Incorrect `direction` specified; must be one of 'all', 'above', or 'below'."
        )

    # Set bin min and max values if not passed to function
    if bin_min is None:
        bin_min = bin_col.min()
    if bin_max is None:
        bin_max = bin_col.max()

    # Define bin edges
    bin_edges = np.arange(bin_min, bin_max, bin_width)

    # Ensure the last bin edge value is bin_max
    bin_edges = np.unique(np.clip(np.append(bin_edges, bin_max), bin_min, bin_max))

    # Define empty flag of 'False' values with indices matching value_col
    flag = pd.Series(index=value_col.index, data=False)

    # Loop through bins and applying flagging
    nbins = len(bin_edges)
    for i in range(nbins - 1):
        # Get data that fall wihtin bin
        y_bin = value_col.loc[(bin_col <= bin_edges[i + 1]) & (bin_col > bin_edges[i])]

        # Get center of binned data
        center = y_bin.mean() if center_type == "mean" else y_bin.median()

        # Define threshold of data flag
        deviation = y_bin.std() * threshold if threshold_type == "std" else threshold

        # Perform flagging depending on specfied direction
        if direction == "above":
            flag_bin = y_bin > (center + deviation)
        elif direction == "below":
            flag_bin = y_bin < (center - deviation)
        else:
            flag_bin = (y_bin > (center + deviation)) | (y_bin < (center - deviation))

        # Record flags in final flag column
        flag.loc[flag_bin.index] = flag_bin

    return flag


def cluster_mahalanobis_2d(
    data_col1: pd.Series, data_col2: pd.Series, n_clusters: int = 13, dist_thresh: float = 3.0
) -> pd.Series:
    """K-means clustering of  data into `n_cluster` clusters; Mahalanobis distance evaluated for each cluster and
    points with distances outside of `dist_thresh` are flagged; distinguishes between asset IDs.

    Args:
        data_col1(:obj:`pandas.Series`): first data column in 2D cluster analysis
        data_col2(:obj:`pandas.Series`): second data column in 2D cluster analysis
        n_clusters(:obj:`int`):' number of clusters to use
        dist_thresh(:obj:`float`): maximum Mahalanobis distance within each cluster for data to be remain unflagged

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """

    # Create 2D data frame for input into cluster algorithm
    df = pd.DataFrame({"d1": data_col1, "d2": data_col2})

    kmeans = KMeans(n_clusters=n_clusters).fit(df)

    # Define empty flag of 'False' values with indices matching value_col
    flag = pd.Series(index=data_col1.index, data=False)

    # Loop through clusters and flag data that fall outside a threshold distance from cluster center
    for i in range(n_clusters):
        # Extract data for cluster
        clust_sub = kmeans.labels_ == i
        cluster = df.loc[clust_sub]

        # Cluster centroid
        centroid = kmeans.cluster_centers_[i]

        # Cluster covariance and inverse covariance
        covmx = cluster.cov()
        invcovmx = sp.linalg.inv(covmx)

        # Compute mahalnobis distance of each point in cluster
        mahalanobis_dist = cluster.apply(
            lambda r: sp.spatial.distance.mahalanobis(r.values, centroid, invcovmx), axis=1
        )

        # Flag data outside the distance threshold
        flag_bin = mahalanobis_dist > dist_thresh

        # Record flags in final flag column
        flag.loc[flag_bin.index] = flag_bin

    return flag
