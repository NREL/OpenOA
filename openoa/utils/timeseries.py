"""
This module provides useful functions for processing timeseries data
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
from pytz import utc, timezone
from dateutil.parser import parse

from openoa.utils._converters import df_to_series


def convert_local_to_utc(d: str | datetime.datetime, tz_string: str) -> datetime.datetime:
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
    # TODO: Make a second copy of this method that aligns with the QA.convert_datetime_column method
    if isinstance(d, str):
        d = parse(d)
    if not isinstance(d, datetime.datetime):
        raise TypeError(
            "The input to `d` must be a `datetime.datetime` object or a string that can be converted to one."
        )

    # Define the timezone, and convert to a the localized timestamp as needed
    tz = timezone(tz_string)
    # TODO: Figure out why a datetime object with tzinfo encoded is different than localizing with pytz
    d_local = d if d.tzinfo else tz.localize(d, is_dst=True)
    return d_local.astimezone(utc)  # calculate UTC time


def convert_dt_to_utc(
    dt_col: pd.Series | str, tz_string: str, data: pd.DataFrame = None
) -> pd.Series:
    """Converts a pandas `Series` of timestamps, string-formatted or `datetime.datetime` objects
        that are in a local timezone `tz_string` to a UTC encoded pandas `Series`.

    Args:
        dt_col (:obj:`pandas.Series` | `str`): A pandas `Series` of datetime objects or
            string-encoded timestamps, or a the name of the column in `data`.
        tz_string (str): The string name for the expected timezone of the provided timestamps in `dt_col`.
        data (:obj:`pandas.DataFrame`, optional): The pandas `DataFrame` containing the timestamp
            column: `dt_col`. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    if isinstance(data, pd.DataFrame):
        dt_col = df_to_series(data, dt_col)
    if isinstance(dt_col[0], str):
        dt_col = dt_col.apply(parse)

    # If the timezone information is already encoded, then convert it to a UTC-converted
    # pandas datetime object automatically, otherwise, localize it, then convert it
    if dt_col[0].tzinfo is not None:
        return pd.to_datetime(dt_col, utc=True)
    return dt_col.dt.tz_localize(tz_string, ambiguous=True).dt.tz_convert(utc)


def find_time_gaps(dt_col: pd.Series | str, freq: str, data: pd.DataFrame = None) -> pd.Series:
    """
    Finds gaps in `dt_col` based on the expected frequency, `freq`, and returns them.

    Args:
        dt_col(:obj:`pandas.Series`): Pandas `Series` of `datetime.datetime` objects or the name
            of the column in `data`.
        freq(:obj:`string`): The expected frequency of the timestamps, which should align with
            the pandas timestamp conventions (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases).
        data (:obj:`pandas.DataFrame`, optional): The pandas `DataFrame` containing the timestamp
            column: `dt_col`. Defaults to None.

    Returns:
        :obj:`pandas.Series`: Series of missing time stamps in datetime.datetime format
    """
    if isinstance(data, pd.DataFrame):
        dt_col = df_to_series(data, dt_col)

    if isinstance(dt_col, pd.DatetimeIndex):
        dt_col = dt_col.to_series()

    # If the difference for all of the timestamps is the expected frequency, 0 (duplicate), or a NaT
    # (first element of `diff`), then return an empty series
    if np.all(dt_col.diff().isin([pd.Timedelta(freq), pd.Timedelta("0"), pd.NaT])):
        return pd.Series([], name=dt_col.name, dtype="object")

    # Create a date range object and return a Series of the set difference of both objects
    range_dt = pd.Series(data=pd.date_range(dt_col.min(), end=dt_col.max(), freq=freq))
    return pd.Series(tuple(set(range_dt).difference(dt_col)), name=dt_col.name)


def find_duplicate_times(dt_col: pd.Series | str, data: pd.DataFrame = None):
    """
    Find duplicate input data and report them. The first duplicated item is not reported, only subsequent duplicates.

    Args:
        dt_col(:obj:`pandas.Series` | `str`): Pandas series of datetime.datetime objects or the name of the
            column in `data`.
        data (:obj:`pandas.DataFrame`, optional): The pandas `DataFrame` containing the timestamp
            column: `dt_col`. Defaults to None.

    Returns:
        :obj:`pandas.Series`: Duplicates from input data
    """
    if isinstance(data, pd.DataFrame):
        dt_col = df_to_series(data, dt_col)

    if isinstance(dt_col, pd.DatetimeIndex):
        dt_col = dt_col.to_series()

    return dt_col[dt_col.duplicated()]


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
        perc = np.float64((s.isnull().sum())) / np.float64(len(s))
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
    n_days = len(s.resample("D"))

    return n_days


def num_hours(s):
    """
    Return number of data points in 's'

    Args:
        s(:obj:`pandas.Series`): The data to be checked for number of data points
    Returns:
        :obj:`int`: Number of hours in the data
    """
    n_hours = len(s.resample("H"))

    return n_hours
