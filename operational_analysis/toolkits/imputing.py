"""
This module provides methods for filling in null data with interpolated (imputed) values.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def correlation_matrix_by_id_column(df, align_col, id_col, value_col):
    """Create a correlation matrix between different assets in a data frame

    Args:
        df(:obj:`pandas.DataFrame`): input data frame
        align_col(:obj:`str`): name of column in <df> on which different assets are to be aligned
        id_col(:obj:`str`): the column distinguishing the different assets
        value_col(:obj:`str`): the column containing the data values to be used when assessing correlation

    Returns:
        :obj:`pandas.DataFrame`: Correlation matrix with <id_col> as index and column names
    """
    assets = df[id_col].unique()  # List different assets in <id_col>
    corr_df = pd.DataFrame(index=assets, columns=assets, dtype=float,
                           data=np.nan)  # Define correlation matrix that is set to NaN by default

    for t in assets:  # Loop through assets
        assets = assets[assets != t]  # Remove the reference asset so we don't loop over it again
        for s in assets:  # Loop through remaining assets
            x_df = df.loc[df[id_col] == t, [align_col, value_col]]  # Asset 't'
            y_df = df.loc[df[id_col] == s, [align_col, value_col]]  # Asset 's'

            merged_df = x_df.merge(y_df, on=align_col).dropna()  # Merge the two on <align_col>, drop any rows with NaN

            # If merged data frame is empty or has only 1 entry, assign NaN correlation
            if (merged_df.empty) | (merged_df.shape[0] < 2):
                corr_df.loc[t, s] = np.nan
            else:  # Now assign correlations
                corr_df.loc[t, s] = np.corrcoef(merged_df[value_col + '_x'], merged_df[value_col + '_y'])[0, 1]

            corr_df.loc[s, t] = corr_df.loc[t, s]  # entry (t,s) is the same as (s,t)

    return corr_df


def impute_data(target_data, target_value_col, ref_data, ref_value_col, align_col,
                method='linear'):  # ADD LINEAR FUNCTIONALITY AS DEFAULT, expection otherwise
    """Replaces NaN data in a target Pandas series with imputed data from a reference Panda series based on a linear
    regression relationship.

    Steps include:

    1. Merge the target and reference data frames on <align_col>, which is shared between the two
    2. Determine the linear regression relationship between the target and reference data series
    3. Apply that relationship to NaN data in the target series for which there is finite data in the reference series
    4. Return the imputed results as well as the index matching the target data frame

    Args:
        target_data(:obj:`pandas.DataFrame`): the data frame containing NaN data to be imputed
        target_value_col(:obj:`str`): the name of the column in <target_data> to be imputed
        ref_data(:obj:`pandas.DataFrame`): the data frame containg data to be used in imputation
        ref_value_col(:obj:`str`): the name of the column in <target_data> to be used in imputation
        align_col(:obj:`str`): the name of the column in <data> on which different assets are to be merged

    Returns:
        :obj:`pandas.Series`: Copy of target_data_col series with NaN occurrences imputed where possible.
    """
    # Merge the input and reference data frames, keeping the input data indices
    merge_df = pd.merge(target_data[[target_value_col, align_col]],
                        ref_data[[ref_value_col, align_col]],
                        on=align_col, how='left')
    merge_df.index = target_data.index

    # If the input and reference series are names the same, adjust their names to match the
    # result from merging
    if target_value_col == ref_value_col:  # same data field used for imputing
        target_value_col = target_value_col + '_x'  # Match the merged column name
        ref_value_col = ref_value_col + '_y'  # Match the merged column name

    # First establish a linear regression relationship for overlapping data
    reg_df = merge_df.dropna()  # Drop NA values so regression works

    # Make sure there is data to perform regression
    if reg_df.empty:
        raise Exception("No valid data to build regression relationship")

    if method == 'linear':
        reg = np.polyfit(reg_df[ref_value_col], reg_df[target_value_col], 1)  # Linear regression relationship
        slope = reg[0]  # Slope
        intercept = reg[1]  # Intercept
    else:
        raise Exception('Only linear regression is currently supported.')

    # Find timestamps for which input data is NaN and imputing data is real
    impute_df = merge_df.loc[(merge_df[target_value_col].isnull()) & np.isfinite(merge_df[ref_value_col])]
    imputed_data = slope * impute_df[ref_value_col] + intercept

    # Apply imputation at those timestamps
    merge_df.loc[impute_df.index, target_value_col] = imputed_data

    # Return target data result after imputation
    return merge_df[target_value_col]


def impute_all_assets_by_correlation(data, input_col, ref_col, align_col, id_col, r2_threshold=0.7, method='linear'):
    '''Imputes NaN data in a Pandas data frame to the best extent possible by considering available data
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
        data(:obj:`pandas.DataFrame`): the data frame subject to imputation
        input_col(:obj:`str`): the name of the column in <data> to be imputed
        ref_col(:obj:`str`): the name of the column in <data> to be used in imputation
        align_col(:obj:`str`): the name of the column in <data> on which different assets are to be merged
        id_col(:obj:`str`): the name of the column in <data> distinguishing different assets
        r2_threshold(:obj:`float`): the correlation threshold for a neighboring assets to be considered valid
                                   for use in imputation

    Returns:
        :obj:`pandas.Series`: The imputation results

    '''
    # Create correlation matrix between different assets
    corr_df = correlation_matrix_by_id_column(data, align_col, id_col, input_col)

    # For efficiency, sort <data> by <id_col> into different dictionary entries immediately
    assets = corr_df.columns
    asset_dict = {}
    for a in assets:
        asset_dict[a] = data.loc[data[id_col] == a]

        # Create imputation series in <data> to be filled, which by default is equal to the original data series
    ret = data[[input_col]]
    ret = ret.rename(columns={input_col: 'imputed_' + input_col})

    # Loop through assets and impute missing data where possible
    for target_id, target_data in tqdm(iter(asset_dict.items())):  # Target asset refers to data requiring imputation

        # List neighboring assets by correlation strength
        corr_list = corr_df[target_id].sort_values(ascending=False)

        # Define some parameters we'll need as we loop through different assets to be used in imputaiton
        num_nan = target_data.loc[target_data[input_col].isnull()].shape[0]  # Number of NaN data in target asset
        num_neighbors = corr_df.shape[0] - 1  # Number of neighboring assets available for imputation
        r2_neighbor = corr_list.values[0]  # R2 value of target and neighbor data
        id_neighbor = corr_list.index[0]  # Name of neighbor

        # For each target asset, loop through neighboring assets and impute where possible
        # Continue until all NaN data are imputed, or we run out of neighbors, or the correlation threshold
        # is no longer met
        while (num_nan > 0) & (num_neighbors > 0) & (r2_neighbor > r2_threshold):

            # Consider highest correlated neighbor remaining and impute target data using that neighbor
            neighbor_data = asset_dict[id_neighbor]
            imputed_data = impute_data(target_data, input_col, neighbor_data, ref_col, align_col, method)

            # Find indices that were imputed (i.e. NaN in <data>, finite in <imputed_data>)
            imputed_bool = ret.loc[imputed_data.index, 'imputed_' + input_col].isnull() & np.isfinite(imputed_data)
            imputed_ind = imputed_bool[imputed_bool].index

            # Assign imputed values for those indices to the input data column
            if len(imputed_ind) > 0:  # There is imputed data to update
                ret.loc[imputed_ind, 'imputed_' + input_col] = imputed_data

            # Update conditional parameters
            num_neighbors = num_neighbors - 1  # One less neighbor
            r2_neighbor = corr_list.values[len(corr_list) - num_neighbors - 1]  # Next highest correlation
            id_neighbor = corr_list.index[
                len(corr_list) - num_neighbors - 1]  # Name of next highest correlated neighbor
            num_nan = ret.loc[imputed_data.index, 'imputed_' + input_col].isnull().shape[0]

    return ret['imputed_' + input_col]
