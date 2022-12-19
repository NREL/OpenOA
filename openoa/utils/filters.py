"""
This module provides functions for flagging pandas data series based on a range of criteria. The functions are largely
intended for application in wind plant operational energy analysis, particularly wind speed vs. power curves.
"""

from __future__ import annotations

from typing import Any, Type

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.cluster import KMeans

from openoa.utils._converters import (
    series_to_df,
    series_method,
    dataframe_method,
    convert_args_to_lists,
)


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
            if it's a `pd.Series`, or the list of lower thresholds for each column in `col`. If the same
            threshold is applied to each column, then pass the single value, otherwise, it must be
            the same length as `col` and `upper`.
        upper (:obj:`float` | `list[float]`): upper threshold (inclusive) for each element of `data`,
            if it's a `pd.Series`, or the list of upper thresholds for each column in `col`. If the same
            threshold is applied to each column, then pass the single value, otherwise, it must be
            the same length as `lower` and `col`.

    Returns:
        :obj:`pandas.Series` | `pandas.DataFrame`: Series or DataFrame (depending on `data` type) with
            boolean entries.
    """
    # Prepare the inputs to be standardized for use with DataFrames
    if to_series := isinstance(data, pd.Series):
        data, col = series_to_df(data)
    if col is None:
        col = data.columns.tolist()

    upper, lower = convert_args_to_lists(len(col), upper, lower)
    if len(col) != len(lower) != len(upper):
        raise ValueError("The inputs to `col`, `above`, and `below` must be the same length.")

    # Only flag the desired columns
    subset = data.loc[:, col].copy()
    flag = ~(subset.ge(lower) & subset.le(upper))

    # Return back a pd.Series if one was provided, else a pd.DataFrame
    return flag[col[0]] if to_series else flag


def unresponsive_flag(
    data: pd.DataFrame | pd.Series,
    threshold: int = 3,
    col: list[str] | None = None,
) -> pd.Series | pd.DataFrame:
    """Flag time stamps for which the reported data does not change for `threshold` repeated intervals.

    Args:
        data (:obj:`pandas.Series` | `pandas.DataFrame`): data frame containing the column to be flagged;
            can either be a `pandas.Series` or `pandas.DataFrame`. If a `pandas.DataFrame`, a list of
            threshold values and columns (if checking a subset of the columns) must be provided.
        col (:obj:`list[str]`): column(s) in `data` to be flagged, by default None. Only required when
            the `data` is a `pandas.DataFrame` and a subset of the columns will be checked. Must be
            the same length as `lower` and `upper`.
        threshold (:obj:`int`): number of intervals over which measurment does not change for each
            element of `data`, regardless if it's a `pd.Series` or `pd.DataFrame`, by default 3.

    Returns:
        :obj:`pandas.Series` | `pandas.DataFrame`: Series or DataFrame (depending on `data` type) with
            boolean entries.
    """
    # Prepare the inputs to be standardized for use with DataFrames
    if to_series := isinstance(data, pd.Series):
        data, col = series_to_df(data)
    if col is None:
        col = data.columns.tolist()
    if not isinstance(threshold, int):
        raise TypeError("The input to `threshold` must be an integer.")

    # Get boolean value of the difference in successive time steps is not equal to zero, and take the
    # rolling sum of the boolean diff column in period lengths defined by threshold
    subset = data.loc[:, col].copy()
    flag = subset.diff(axis=0).ne(0).rolling(threshold - 1).sum()

    # Create boolean series that is True if rolling sum is zero
    flag = flag == 0

    # Need to flag preceding `threshold` values as well
    flag = flag | np.any([flag.shift(-1 - i, axis=0) for i in range(threshold - 1)], axis=0)

    # Return back a pd.Series if one was provided, else a pd.DataFrame
    return flag[col[0]] if to_series else flag


def std_range_flag(
    data: pd.DataFrame | pd.Series,
    threshold: float | list[float] = 2.0,
    col: list[str] | None = None,
) -> pd.Series | pd.DataFrame:
    """Flag time stamps for which the measurement is outside of the threshold number of standard deviations
        from the mean across the data.

    ... note:: This method does not distinguish between asset IDs.

    Args:
        data (:obj:`pandas.Series` | `pandas.DataFrame`): data frame containing the column to be flagged;
            can either be a `pandas.Series` or `pandas.DataFrame`. If a `pandas.DataFrame`, a list of
            threshold values and columns (if checking a subset of the columns) must be provided.
        col (:obj:`list[str]`): column(s) in `data` to be flagged, by default None. Only required when
            the `data` is a `pandas.DataFrame` and a subset of the columns will be checked. Must be
            the same length as `lower` and `upper`.
        threshold (:obj:`float` | `list[float]`): multiplicative factor on the standard deviation of `data`,
            if it's a `pd.Series`, or the list of multiplicative factors on the standard deviation for
            each column in `col`. If the same factor is applied to each column, then pass the single
            value, otherwise, it must be the same length as `col` and `upper`.

    Returns:
        :obj:`pandas.Series` | `pandas.DataFrame`: Series or DataFrame (depending on `data` type) with
            boolean entries.
    """
    # Prepare the inputs to be standardized for use with DataFrames
    if to_series := isinstance(data, pd.Series):
        data, col = series_to_df(data)
    if col is None:
        col = data.columns.tolist()

    threshold, *_ = convert_args_to_lists(len(col), threshold)
    if len(col) != len(threshold):
        raise ValueError("The inputs to `col` and `threshold` must be the same length.")

    subset = data.loc[:, col].copy()
    data_mean = subset.mean(axis=0)
    data_std = subset.std(axis=0) * np.array(threshold)
    flag = subset.le(data_mean - data_std) | subset.ge(data_mean + data_std)

    # Return back a pd.Series if one was provided, else a pd.DataFrame
    return flag[col[0]] if to_series else flag


@series_method(data_cols=["window_col", "value_col"])
def window_range_flag(
    window_col: str | pd.Series = None,
    window_start: float = -np.inf,
    window_end: float = np.inf,
    value_col: str | pd.Series = None,
    value_min: float = -np.inf,
    value_max: float = np.inf,
    data: pd.DataFrame = None,
) -> pd.Series:
    """Flag time stamps for which measurement in `window_col` are within the range: [`window_start`, `window_end`], and
    the measurements in `value_col` are outside of the range [`value_min`, `value_max`].

    Args:
        data (:obj:`pandas.DataFrame`): data frame containing the columns `window_col` and
            `value_col`, by default None.
        window_col (:obj:`str` | `pandas.Series`): Name of the column or  used to define the window
            range or the data as a pandas Series, by default None.
        window_start(:obj:`float`): minimum value for the inclusive window, by default -np.inf.
        window_end(:obj:`float`): maximum value for the inclusive window, by default np.inf.
        value_col (:obj:`str` | `pandas.Series`): Name of the column used to define the value range
            or the data as a pandas Series, by default None.
        value_max(:obj:`float`): upper threshold for the inclusive data range; default np.inf
        value_min(:obj:`float`): lower threshold for the inclusive data range; default -np.inf

    Returns:
        :obj:`pandas.Series`: Series with boolean entries.
    """
    flag = window_col.between(window_start, window_end) & ~value_col.between(value_min, value_max)
    return flag


@series_method(data_cols=["bin_col", "value_col"])
def bin_filter(
    bin_col: pd.Series | str,
    value_col: pd.Series | str,
    bin_width: float,
    threshold: float = 2,
    center_type: str = "mean",
    bin_min: float = None,
    bin_max: float = None,
    threshold_type: str = "std",
    direction: str = "all",
    data: pd.DataFrame = None,
):
    """Flag time stamps for which data in `value_col` when binned by data in `bin_col` into bins of
    width `bin_width` are outside the `threhsold` bin. The `center_type` of each bin can be either the
    median or mean, and flagging can be applied directionally (i.e. above or below the center, or both)

    Args:
        bin_col(:obj:`pandas.Series` | `str`): The Series or column in `data` to be used for binning.
        value_col(:obj:`pandas.Series`): The Series or column in `data` to be flagged.
        bin_width(:obj:`float`): Width of bin in units of `bin_col`
        threshold(:obj:`float`): Outlier threshold (multiplicative factor of std of `value_col` in bin)
        bin_min(:obj:`float`): Minimum bin value below which flag should not be applied
        bin_max(:obj:`float`): Maximum bin value above which flag should not be applied
        threshold_type(:obj:`str`): Option to apply a 'std', 'scalar', or 'mad' (median absolute deviation)
            based threshold
        center_type(:obj:`str`): Option to use a 'mean' or 'median' center for each bin
        direction(:obj:`str`): Option to apply flag only to data 'above' or 'below' the mean, by default 'all'
        data(:obj:`pd.DataFrame`): DataFrame containing both `bin_col` and `value_col`, if data
            are part of the same DataFrame, by default None.

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """
    if center_type not in ("mean", "median"):
        raise ValueError("Incorrect `center_type` specified; must be one of 'mean' or 'median'.")
    if threshold_type not in ("std", "scalar", "mad"):
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
        if threshold_type == "std":
            deviation = y_bin.std() * threshold
        elif threshold_type == "scalar":
            deviation = threshold
        else: # median absolute deviation (mad)
            deviation = (y_bin - center).abs().median() * threshold

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


@dataframe_method(data_cols=["data_col1", "data_col2"])
def cluster_mahalanobis_2d(
    data_col1: pd.Series | str,
    data_col2: pd.Series | str,
    n_clusters: int = 13,
    dist_thresh: float = 3.0,
    data: pd.DataFrame = None,
) -> pd.Series:
    """K-means clustering of  data into `n_cluster` clusters; Mahalanobis distance evaluated for each cluster and
    points with distances outside of `dist_thresh` are flagged; distinguishes between asset IDs.

    Args:
        data_col1(:obj:`pandas.Series` | `str`): Series or column `data` corresponding to the first
            data column in a 2D cluster analysis
        data_col2(:obj:`pandas.Series` | `str`): Series or column `data` corresponding to the second
            data column in a 2D cluster analysis
        n_clusters(:obj:`int`):' number of clusters to use
        dist_thresh(:obj:`float`): maximum Mahalanobis distance within each cluster for data to be remain unflagged
        data(:obj:`pd.DataFrame`): DataFrame containing both `data_col1` and `data_col2`, if data
            are part of the same DataFrame, by default None.

    Returns:
        :obj:`pandas.Series(bool)`: Array-like object with boolean entries.
    """
    data = data.loc[:, [data_col1, data_col2]].copy()
    kmeans = KMeans(n_clusters=n_clusters).fit(data)

    # Define empty flag of 'False' values with indices matching value_col
    flag = pd.Series(index=data.index, data=False)

    # Loop through clusters and flag data that fall outside a threshold distance from cluster center
    for i in range(n_clusters):
        # Extract data for cluster
        clust_sub = kmeans.labels_ == i
        cluster = data.loc[clust_sub]

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
