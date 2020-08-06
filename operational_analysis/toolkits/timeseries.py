"""
This module provides useful functions for processing timeseries data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pytz import timezone


def convert_local_to_utc(d, tz_string):
    """
    Convert timestamps in local time to UTC. The function can only act on a single timestamp at a time, so
    for example use the .apply function in Pandas:

        date_utc = df['time'].apply(convert_local_to_utc, args = ('US/Pacific',))

    Also note that this function doesn't solve the end of DST when times between 1:00-2:00 are repeated
    in November. Those dates are left repeated in UTC time and need to be shifted manually.

    The function does address the missing 2:00-3:00 times at the start of DST in March

    Args:
        d(:obj:`datetime.datetime`): the local date, tzinfo must not be set
        tz_string(:obj:`str`): the local timezone

    Returns:
        :obj:`datetime.datetime`: the local date converted to UTC time

    """
    if d.tzinfo:
        raise Exception("d parameter must not have a timezone")

    d_obj = datetime(d.year, d.month, d.day, d.hour, d.minute)  # Convert datetime object into simple integer form
    tz = timezone(tz_string)  # define timezone
    d_local = tz.localize(d_obj, is_dst=True)  # localize the date object
    utc = timezone('UTC')  # define UTC timezone
    d_utc = d_local.astimezone(utc)  # calculate UTC time

    return d_utc


def find_time_gaps(t_series, freq):
    """
    Find data gaps in input data and report them

    Args:
        t_series(:obj:`pandas.Series`): Pandas series of datetime objects
        freq(:obj:`string`): time series frequency

    Returns:
        :obj:`pandas.Series`: Series of missing time stamps in datetime format
    """
    
    # Convert 't_series' to Pandas series in case a time index is passed
    t_series = pd.Series(t_series)

    if t_series.size == 0:
        return t_series

    range_dt = pd.Series(data=pd.date_range(t_series.min(),
                                            end=t_series.max(), freq=freq))  # Full range of timestamps

    # Find missing time stamps by concatenating full timestamps and actual and removing duplicates
    # What remains is those timestamps not found in the data
    missing_dt = (pd.concat([range_dt, t_series])).drop_duplicates(keep=False)

    return missing_dt


def find_duplicate_times(t_series, freq):
    """
    Find duplicate input data and report them. The first duplicated item is not reported, only subsequent duplicates.

    Args:
        t_series(:obj:`pandas.Series`): Pandas series of datetime objects
        freq(:obj:`string`): time series frequency

    Returns:
        :obj:`pandas.Series`: Duplicates from input data
    """
    
    # Convert 't_series' to Pandas series in case a time index is passed
    t_series = pd.Series(t_series)
    
    repeated_steps = t_series[t_series.duplicated()]

    return repeated_steps


def gap_fill_data_frame(df, time_col, freq):
    """
    Find missing timestamps in the input data frame and add rows with NaN values for those missing rows.
    Return a new data frame that has no missing timestamps and that is sorted by time.

    Args:
        df(:obj:`pandas.DataFrame`): the input data frame
        time_col(:obj:`str`): name of the column in 'df' with time data

    Returns:
        :obj:`pandas.DataFrame`: output data frame with NaN data for the data gaps

    """
    # If the dataframe is empty, just return it.
    if df[time_col].size == 0:
        return df

    timestamp_gaps = find_time_gaps(df[time_col], freq)  # Find gaps in timestep
    gap_df = pd.DataFrame(columns=df.columns)
    gap_df[time_col] = timestamp_gaps

    return df.append(gap_df).sort_values(time_col)


def percent_nan(s):
    """
    Return percentage of data that are Nan or 1 if the series is empty.

    Args:
        s(:obj:`pandas.Series`): The data to be checked for 'na' values
    
    Returns:
        :obj:`float`: Percentage of NaN data in the data series
    """
    if len(s) > 0:
        perc = np.float((s.isnull().sum())) / np.float(len(s))
    else:
        perc = 1
    return perc


def num_days(s):
    """
    Return number of days in 's'

    Args:
        s(:obj:`pandas.Series`): The data to be checked for number of days.
    
    Returns:
        :obj:`int`: Number of days in the data
    """
    n_days = len(s.resample('D'))

    return n_days


def num_hours(s):
    """
    Return number of data points in 's'

    Args:
        s(:obj:`pandas.Series`): The data to be checked for number of data points
    Returns:
        :obj:`int`: Number of hours in the data
    """
    n_hours = len(s.resample('H'))

    return n_hours
