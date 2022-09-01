"""
This module provides methods for filling in null data with interpolated (imputed) values.
"""

from hashlib import algorithms_available

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.polynomial import Polynomial


def asset_correlation_matrix(data: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Create a correlation matrix on a MultiIndex `DataFrame` with time (or a different
    alignment value) and ID values as its indices, respectively.

    Args:
        data(:obj:`pandas.DataFrame`): input data frame such as `Plant.scada` that uses a
            MultiIndex with a timestamp and ID column for indices, in that order.
        value_col(:obj:`str`): the column containing the data values to be used when
            assessing correlation

    Returns:
        :obj:`pandas.DataFrame`: Correlation matrix with <id_col> as index and column names
    """
    corr_df = data.loc[:, [value_col]].unstack().corr(min_periods=2)
    corr_df = corr_df.droplevel(0).droplevel(0, axis=1)  # drop the added axes
    corr_df.index = corr_df.index.set_names(None)
    corr_df.columns = corr_df.index.set_names(None)
    np.fill_diagonal(corr_df.values, np.nan)
    return corr_df


def impute_data(
    target_col: str,
    reference_col: str,
    target_data: pd.DataFrame = None,
    reference_data: pd.DataFrame = None,
    align_col: str = None,
    method: str = "linear",
    degree: int = 1,
    data: pd.DataFrame = None,
) -> pd.Series:  # ADD LINEAR FUNCTIONALITY AS DEFAULT, expection otherwise
    """Replaces NaN data in a target Pandas series with imputed data from a reference Panda series based on a linear
    regression relationship.

    Steps include:

    1. Merge the target and reference data frames on <align_col>, which is shared between the two
    2. Determine the linear regression relationship between the target and reference data series
    3. Apply that relationship to NaN data in the target series for which there is finite data in the reference series
    4. Return the imputed results as well as the index matching the target data frame

    Args:
        target_data(:obj:`pandas.DataFrame`): the data frame containing NaN data to be imputed
        target_col(:obj:`str`): the name of the column in <target_data> to be imputed
        ref_data(:obj:`pandas.DataFrame`): the data frame containg data to be used in imputation
        reference_col(:obj:`str`): the name of the column in <target_data> to be used in imputation
        align_col(:obj:`str`): the name of the column in <data> on which different assets are to be merged

    Returns:
        :obj:`pandas.Series`: Copy of target_data_col series with NaN occurrences imputed where possible.
    """
    if data is None:
        if any((not isinstance(x, pd.DataFrame) for x in (target_data, reference_data))):
            raise TypeError(
                "If `data` is not provided, then `ref_data` and `target_data` must be provided as pandas DataFrames."
            )
        if target_col not in target_data:
            raise ValueError("The input `target_col` is not a column of `target_data`.")
        if reference_col not in target_data:
            raise ValueError("The input `reference_col` is not a column of `ref_data`.")
        if align_col not in target_data and align_col not in target_data.index.names:
            raise ValueError(
                "The input `align_col` is not a column or index of one of `target_data`."
            )
        if align_col not in reference_data and align_col not in reference_data.index.names:
            raise ValueError(
                "The input `align_col` is not a column or index of one of `reference_data`."
            )

        # Unify the data, if the target and reference data are provided separately
        data = target_data.merge(reference_data, on=align_col, how="left")

        # If the input and reference series are names the same, adjust their names to match the
        # result from merging
        if target_col == reference_col:  # same data field used for imputing
            target_col = target_col + "_x"  # Match the merged column name
            reference_col = reference_col + "_y"  # Match the merged column name

    if target_col not in data:
        raise ValueError("The input `target_col` is not a column of `data`.")
    if reference_col not in data:
        raise ValueError("The input `reference_col` is not a column of `data`.")

    data = data.loc[:, [reference_col, target_col]]
    data_reg = data.dropna()
    if data_reg.empty:
        raise ValueError("Not enough data to create a curve fit.")

    # Ensure old method call will work here
    if method == "linear":
        method = "polynomial"
        degree = 1
    if method == "polynomial":
        curve_fit = Polynomial.fit(data_reg[reference_col], data_reg[target_col], degree)
    else:
        raise NotImplementedError(
            "Only 'linear' (1-degree polynomial) and 'polynomial' fits are implemented at this time."
        )

    imputed = data.loc[
        (data[target_col].isnull() & np.isfinite(data[reference_col])), [reference_col]
    ]
    return curve_fit(imputed[reference_col])


def impute_all_assets_by_correlation(
    data: pd.DataFrame,
    impute_col: str,
    ref_col: str,
    r2_threshold: float = 0.7,
    method: str = "linear",
):
    """Imputes NaN data in a Pandas data frame to the best extent possible by considering available data
    across different assets in the data frame. Highest correlated assets are prioritized in the imputation process.

    Steps include:

    1. Establish correlation matrix of specified data between different assets
    2. For each asset in the data frame, sort neighboring assets by correlation strength
    3. Then impute asset data based on available data in the highest correlated neighbor
    4. If NaN data still remains in asset, move on to next highest correlated neighbor, etc.
    5. Continue until either:
        a. There are no NaN data remaining in asset data
        b. There are no more neighbors to consider
        c. The neighboring asset does not meet the specified correlation threshold, <r2_threshold>

    Args:
        data(:obj:`pandas.DataFrame`): input data frame such as `Plant.scada` that uses a
            MultiIndex with a timestamp and ID column for indices, in that order.
        impute_col(:obj:`str`): the name of the column in <data> to be imputed.
        ref_col(:obj:`str`): the name of the column in <data> to be used in imputation.
        r2_threshold(:obj:`float`): the correlation threshold for a neighboring assets to be considered valid
            for use in imputation.
        method(:obj:`str`): The imputation method.

    Returns:
        :obj:`pandas.Series`: The imputation results

    """
    # Create correlation matrix between different assets
    corr_df = asset_correlation_matrix(data, impute_col)

    # For efficiency, sort <data> by <id_col> into different dictionary entries immediately
    assets = corr_df.columns
    asset_dict = {}
    for a in assets:
        asset_dict[a] = data.loc[data[id_col] == a]

    # Create imputation series in <data> to be filled, which by default is equal to the original data series
    ret = data[[impute_col]]
    ret = ret.rename(columns={impute_col: "imputed_" + impute_col})

    # Loop through assets and impute missing data where possible
    for target_id, target_data in tqdm(
        iter(asset_dict.items())
    ):  # Target asset refers to data requiring imputation

        # List neighboring assets by correlation strength
        corr_list = corr_df[target_id].sort_values(ascending=False)

        # Define some parameters we'll need as we loop through different assets to be used in imputaiton
        num_nan = target_data.loc[target_data[impute_col].isnull()].shape[
            0
        ]  # Number of NaN data in target asset
        num_neighbors = (
            corr_df.shape[0] - 1
        )  # Number of neighboring assets available for imputation
        r2_neighbor = corr_list.values[0]  # R2 value of target and neighbor data
        id_neighbor = corr_list.index[0]  # Name of neighbor

        # For each target asset, loop through neighboring assets and impute where possible
        # Continue until all NaN data are imputed, or we run out of neighbors, or the correlation threshold
        # is no longer met
        while (num_nan > 0) & (num_neighbors > 0) & (r2_neighbor > r2_threshold):

            # Consider highest correlated neighbor remaining and impute target data using that neighbor
            neighbor_data = asset_dict[id_neighbor]
            imputed_data = impute_data(
                target_data, impute_col, neighbor_data, ref_col, align_col, method
            )

            # Find indices that were imputed (i.e. NaN in <data>, finite in <imputed_data>)
            imputed_bool = ret.loc[
                imputed_data.index, "imputed_" + impute_col
            ].isnull() & np.isfinite(imputed_data)
            imputed_ind = imputed_bool[imputed_bool].index

            # Assign imputed values for those indices to the input data column
            if len(imputed_ind) > 0:  # There is imputed data to update
                ret.loc[imputed_ind, "imputed_" + impute_col] = imputed_data

            # Update conditional parameters
            num_neighbors = num_neighbors - 1  # One less neighbor
            r2_neighbor = corr_list.values[
                len(corr_list) - num_neighbors - 1
            ]  # Next highest correlation
            id_neighbor = corr_list.index[
                len(corr_list) - num_neighbors - 1
            ]  # Name of next highest correlated neighbor
            num_nan = ret.loc[imputed_data.index, "imputed_" + impute_col].isnull().shape[0]

    return ret["imputed_" + impute_col]
