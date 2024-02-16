"""Provides the Quality Assurance (QA) methods for SCADA data checking."""

from __future__ import annotations

from typing import Tuple, Union
from datetime import datetime

import pytz
import h5pyd
import numpy as np
import pandas as pd
import dateutil
import matplotlib.pyplot as plt
from pyproj import Proj
from dateutil import tz

from openoa.utils import timeseries as ts
from openoa.logging import logging, logged_method_call
from openoa.utils.plot import set_styling


Number = Union[int, float]
logger = logging.getLogger(__name__)
set_styling()


def _remove_tz(df: pd.DataFrame, t_local_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Identify the non-timestamp elements in the DataFrame timestamp column and return
    a truth array for filtering the values and the timezone-naive timestamps.

    This function should be used after all data has been converted to timestamps, and will
    therefore only be checking for `float` data as invalid because this is the standard
    fault data-type in the conversion to datetime data.

    Args:
        df (:obj:`pandas.DataFrame`): The DataFrame of interest.
        t_local_column (:obj:`str`): The name of the timestamp column.

    Returns:
        :obj:`numpy.ndarray`: Truth array that can be used to filter the timestamps and subsequent values.
        :obj:`numpy.ndarray`: Array of timezone-naive python `datetime` objects.
    """
    arr = np.array(
        [
            [True, pd.to_datetime(el).tz_localize(None).to_pydatetime()]
            if not isinstance(el, float)
            else [False, np.nan]
            for ix, el in enumerate(df.loc[:, t_local_column])
        ]
    )
    ix_filter = arr[:, 0].astype(bool)
    time_stamps = arr[:, 1]
    return ix_filter, time_stamps


def _get_time_window(df, ix, hour_window, time_col, local_time_col, utc_time_col):
    """Retrieves the time window in a DataFrame with likely confusing
    implementation of timezones.

    Args:
        df (:obj:`pandas.DataFrame`): The DataFrame of interest.
        ix (:obj:`pandas._libs.tslibs.timestamps.Timestamp`]): The starting
            Timestamp on which to base the time window.
        hour_window (:obj:`pandas._libs.tslibs.timedeltas.Timedelta`): The number
            length of the window, in hours.
        time_col (:obj:`str`): The original input datetime column.
        local_time_col (:obj:`str`): The local timezone resolved datetime column.
        utc_time_col (:obj:`str`): The UTC resolved datetime column.

    Returns:
        (:obj:`pandas.DataFrame`): The filtered DataFrame object
    """
    if ix.tz is None:
        col = time_col
    elif str(ix.tz) == "UTC":
        col = utc_time_col
    else:
        col = local_time_col
    start = np.where(df[col] == ix - hour_window)[0][0]
    end = np.where(df[col] == ix + hour_window)[0][0]
    return df.iloc[start:end]


def determine_offset_dst(df: pd.DataFrame, local_tz: str) -> pd.DataFrames:
    """Creates a column of "utc_offset" and "is_dst".

    Args:
        df(:obj:`pd.DataFrame`): The dataframe object to manipulate with a tz-aware ``pandas.DatetimeIndex``.
        local_tz(:obj: 'String'): The ``pytz``-compatible timezone for the input `time_field`, by
            default UTC. This should be in the format of "Country/City" or "Region/City" such as
            "America/Denver" or "Europe/Paris".

    Returns:
        (:obj:`pd.DataFrame`): The updated dataframe with "utc_offset" and "is_dst" columns created.
    """
    # The new column names
    _offset = "utc_offset"
    _dst = "is_dst"

    # Get the daylight savings time offset for a non-DST timestamp for comparison
    _non_dst_offset = pytz.timezone(local_tz).localize(datetime(2021, 1, 1)).utcoffset()

    dt = df.copy().tz_convert(local_tz)
    dt_col = dt.index.to_pydatetime()

    # Determine the Daylight Savings Time status and UTC offset
    dt[_offset] = [el.utcoffset() for el in dt_col]
    dt[_dst] = (dt[_offset] != _non_dst_offset).astype(bool)

    # Convert back to UTC
    dt = dt.tz_convert("UTC")
    return dt


def convert_datetime_column(
    df: pd.DataFrame, time_col: str, local_tz: str, tz_aware: bool
) -> pd.DataFrame:
    """Converts the passed timestamp data to a pandas-encoded Datetime, and creates a
    corresponding localized and UTC timestamp using the :py:attr:`time_field` column name with either
    "localized" or "utc", respectively. The ``_df`` object then uses the local timezone
    timestamp for its index.

    Args:
        df(:obj: `pd.DataFrame`): The SCADA ``pd.DataFrame``
        time_col(:obj: `string`): The string name of datetime stamp column in ``df``.
        local_tz(:obj: 'string'): The ``pytz``-compatible timezone for the input :py:attr:`time_field`, by
            default UTC. This should be in the format of "Country/City" or "Region/City" such as
            "America/Denver" or "Europe/Paris".
        tz_aware(:obj: `bool`): Indicator for if the provided data in :py:attr:`time_col` has the timezone
            information embedded (``True``), or not (``False``).

    Returns:
        (:obj: `pd.DataFrame`): The updated ``pd.DataFrame`` with an index of ``pd.DatetimeIndex`` with
            UTC time-encoding, and the following new columns:
            - :py:attr:`time_col`_utc: A UTC-converted timestamp column
            - :py:attr:`time_col`_localized: The fully converted and localized timestamp column
            - utc_offset: The difference, in hours between the localized and UTC time
            - is_dst: Indicator for whether of not the timestamp is considered to be DST (``True``) or not (``False``)
    """
    # Create the necessary columns for processing
    t_utc = f"{time_col}_utc"
    t_local = f"{time_col}_localized"

    # Convert the timestamps to datetime.datetime objects
    dt_col = df[time_col].values

    # Check for raw timestamp inputs or pre-formatted
    if isinstance(dt_col[0], str):
        dt_col = [dateutil.parser.parse(el) for el in dt_col]

    # Read the timestamps as UTC, then convert to the local timezone if the data are
    # timezone-aware, otherwise localize the timestamp to the local timezone
    if tz_aware:
        pd_dt_col = pd.to_datetime(dt_col, utc=True).tz_convert(local_tz)
        df[t_local] = pd_dt_col
    else:
        pd_dt_col = pd.to_datetime(dt_col)
        df[t_local] = pd_dt_col.tz_localize(local_tz, ambiguous=True)

    df[time_col] = pd_dt_col
    df = df.set_index(pd.DatetimeIndex(df[t_local]))

    # Create the UTC-converted time-stamp
    try:
        utc = tz.tzutc()
        df[t_utc] = pd.to_datetime([el.astimezone(utc) for el in df.index]).tz_convert("UTC")
    except AttributeError:  # catches numpy datetime error for astimezone() not existing
        df = df.tz_convert("UTC")
        df[t_utc] = df.index

    # Adjust the index name to reflect the change to a UTC-based timestamp
    df.index.name = t_utc

    df = determine_offset_dst(df, local_tz=local_tz)
    return df


def duplicate_time_identification(
    df: pd.DataFrame, time_col: str, id_col: str
) -> tuple[pd.Series, None | pd.Series, None | pd.Series]:
    """Identifies the time duplications on the modified SCADA data frame to highlight the
    duplications from the original time data (:py:attr:`time_col`), the UTC timestamps, and the localized
    timestamps, if the latter are available.

    Args:
        df (:obj: `pd.DataFrame`): The resulting SCADA dataframe from :py:meth:`convert_datetime_column()`, otherwise
            the UTC and localized column checks will return ``None``.
        time_col (:obj: `str`): The string name of the timestamp column.
        id_col (:obj: `str`): The string name of the turbine asset_id column, to ensure that duplicates
            aren't based off multiple turbine's data.

    Returns:
        (tuple[pd.Series, None | pd.Series, None | pd.Series]): The dataframe subsets with duplicate
            timestamps based on the original timestamp column, the localized timestamp column (``None``
            if the column does not exist), and the UTC-converted timestamp column (``None`` if the
            column does not exist).
    """
    # Create the necessary columns for processing
    t_utc = f"{time_col}_utc"
    t_local = f"{time_col}_localized"

    time_dups = df.loc[df.duplicated(subset=[id_col, time_col]), time_col]
    time_dups_utc = None
    time_dups_local = None

    if t_utc in df.columns:
        time_dups_utc = df.loc[df.duplicated(subset=[id_col, t_utc]), t_utc]

    if t_local in df.columns:
        time_dups_local = df.loc[df.duplicated(subset=[id_col, t_local]), t_local]

    return time_dups, time_dups_local, time_dups_utc


def gap_time_identification(
    df: pd.DataFrame, time_col: str, freq: str
) -> tuple[pd.Series, None | pd.Series, None | pd.Series]:
    """Identifies the time gaps on the modified SCADA data frame to highlight the missing timestamps
    from the original time data (`time_col`), the UTC timestamps, and the localized timestamps, if
    the latter are available.

    Args:
        df (:obj: `pd.DataFrame`): The resulting SCADA dataframe from :py:meth:`convert_datetime_column()`, otherwise
            the UTC and localized column checks will return `1`.
        time_col (:obj: `str`): The string name of the timestamp column.
        freq (:obj: `str`): The expected frequency of the timestamps, which should align with
            the pandas timestamp conventions (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases).

    Returns:
        (tuple[pd.Series, None | pd.Series, None | pd.Series]): The dataframe subsets with duplicate
            timestamps based on the original timestamp column, the localized timestamp column (``None``
            if the column does not exist), and the UTC-converted timestamp column (``None`` if the
            column does not exist).
    """
    # Create the necessary columns for processing
    t_utc = f"{time_col}_utc"
    t_local = f"{time_col}_localized"

    time_gaps = ts.find_time_gaps(df[time_col], freq=freq)
    time_gaps_utc = None
    time_gaps_local = None

    if t_utc in df.columns:
        time_gaps_utc = ts.find_time_gaps(df[t_utc], freq=freq)

    if t_local in df.columns:
        time_gaps_local = ts.find_time_gaps(df[t_local], freq=freq)

    return time_gaps, time_gaps_local, time_gaps_utc


def describe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Thin wrapper for ``pd.DataFrame.describe()``, but transposes the results to be easier to read.

    Args:
        df (:obj: `pd.DataFrame`): The resulting SCADA dataframe from :py:meth:`convert_datetime_column()`, otherwise
            the UTC and localized column checks will return ``None``.
        kwargs (:obj: `dict`): Dictionary of additional arguments to pass to ``df.describe()``.

    Returns:
        pd.DataFrame: The results of ``df.describe().T``.
    """
    return df.describe(**kwargs).T


def daylight_savings_plot(
    df: pd.DataFrame,
    local_tz: str,
    id_col: str,
    time_col: str,
    power_col: str,
    freq: str,
    hour_window: int = 3,
):
    """Produce a timeseries plot showing daylight savings events for each year of the SCADA data frame,
    highlighting potential duplications and gaps with the original timestamps compared against the
    UTC-converted timestamps.

    Args:
        df (:obj: `pd.DataFrame`): The resulting SCADA dataframe from :py:meth:`convert_datetime_column()`.
        local_tz(:obj: 'String'): The ``pytz``-compatible timezone for the input :py:attr:`time_field`, by
            default UTC. This should be in the format of "Country/City" or "Region/City" such as
            "America/Denver" or "Europe/Paris".
        id_col (:obj: `str`): The string name of the turbine asset_id column in :py:attr:`df`, to ensure that
            duplicates aren't based off multiple turbine's data.
        time_col (:obj: `str`): The string name of the timestamp column in :py:attr:`df`.
        power_col(:obj: 'str'): String name of the power column in :py:attr:`df`.
        freq (:obj: `str`): The expected frequency of the timestamps, which should align with
            the pandas timestamp conventions (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases).
        hour_window(:obj: 'int'): number of hours, before and after the Daylight Savings Time
            transitions to view in the plot, by default 3.
    """
    # Create the necessary columns for processing
    _dst = "is_dst"
    t_utc = f"{time_col}_utc"
    t_local = f"{time_col}_localized"

    # Get data for one of the turbines
    df_dst = df.loc[df[id_col] == df[id_col].unique()[0]]
    df_full = df_dst.copy()
    try:
        df_full = df_full.tz_convert(local_tz)
    except TypeError:
        pass

    time_duplications, time_duplications_utc, _ = duplicate_time_identification(
        df, time_col, id_col
    )
    years = df_full[time_col].dt.year.unique().astype(int)  # Years in data record
    num_years = len(years)
    hour_window = pd.Timedelta(hours=hour_window)

    j = 0
    fig = plt.figure(figsize=(20, 24))
    axes = axes = fig.subplots(num_years, 2, gridspec_kw=dict(wspace=0.15, hspace=0.3))
    for i, year in enumerate(years):
        year_data = df_full.loc[df_full[time_col].dt.year == year]
        dst_dates = np.where(year_data[_dst].values)[0]

        # Break the plotting loop if there is a partial year without DST in the data
        if dst_dates.size == 0:
            break

        # Get the start and end DatetimeIndex values
        start_ix = year_data.iloc[dst_dates[0]].name
        end_ix = year_data.iloc[dst_dates[-1] + 1].name

        # Create the data subsets for plotting the appropriate window
        data_spring = _get_time_window(year_data, start_ix, hour_window, time_col, t_local, t_utc)
        data_fall = _get_time_window(year_data, end_ix, hour_window, time_col, t_local, t_utc)

        data_spring = data_spring.sort_values(
            [time_col, power_col], na_position="first"
        ).drop_duplicates(subset=time_col, keep="last")
        data_fall = data_fall.sort_values(
            [time_col, power_col], na_position="first"
        ).drop_duplicates(subset=time_col, keep="last")

        # Plot each as side-by-side subplots
        ax = axes[i, 0]
        if np.sum(~np.isnan(data_spring[power_col])) > 0:
            j += 1
            # For localized time, we want to ensure we're capturing the DST switch as missing data
            ix_filter, time_stamps = _remove_tz(data_spring, time_col)
            time_stamps = pd.Series(time_stamps[ix_filter])
            power_data = data_spring.loc[ix_filter, power_col].tolist()

            # Find the missing data points on the timezone stripped data and append
            # it to the time stamps, then identify where to insert NaN in the power data
            missing = ts.find_time_gaps(time_stamps, freq)
            missing = pd.to_datetime(missing.values)
            time_stamps = np.append(time_stamps, missing)
            time_stamps.sort()
            nan_ix = sorted([np.where(el == time_stamps)[0][0] for el in missing])
            for ix in nan_ix:
                power_data.insert(ix, float("nan"))

            ax.plot(
                time_stamps,
                power_data,
                label="Original Timestamp",
                c="tab:blue",
                lw=1.5,
            )

            # Plot the duplicated time stamps as scatter points
            duplications = data_spring.loc[data_spring[time_col].isin(time_duplications)]
            if duplications.shape[0] > 0:
                ix_filter, time_stamps = _remove_tz(duplications, time_col)
                ax.scatter(
                    time_stamps[ix_filter],
                    duplications.loc[ix_filter, power_col],
                    c="tab:blue",
                    label="Original Timestamp Duplicates",
                )

            # Find bad timestamps, then fill in any potential UTC time gaps due the focus on the input time field
            ix_filter, time_stamps = _remove_tz(data_spring, t_utc)
            data_spring = ts.gap_fill_data_frame(data_spring[ix_filter], t_utc, freq)
            ix_filter, time_stamps = _remove_tz(data_spring, t_utc)
            ax.plot(
                time_stamps[ix_filter],
                data_spring.loc[ix_filter, power_col],
                label="UTC Timestamp",
                c="tab:orange",
                linestyle="--",
            )

            # Plot the duplicated time stamps as scatter points
            duplications = data_spring.loc[data_spring[t_utc].isin(time_duplications_utc)]
            if duplications.shape[0] > 0:
                ix_filter, time_stamps = _remove_tz(duplications, t_utc)
                ax.scatter(
                    time_stamps[ix_filter],
                    duplications.loc[ix_filter, power_col],
                    c="tab:orange",
                    label="UTC Timestamp Duplicates",
                )

        ax.set_title(f"{year}, Spring")
        ax.set_ylabel("Power")
        ax.set_xlabel("Date")
        ax.legend(loc="lower left", fontsize=10)
        ax.tick_params(axis="x", rotation=20)

        ax = axes[i, 1]
        if np.sum(~np.isnan(data_fall[power_col])) > 0:
            j += 1
            ix_filter, time_stamps = _remove_tz(data_fall, time_col)
            ax.plot(
                time_stamps[ix_filter],
                data_fall.loc[ix_filter, power_col],
                label="Original Timestamp",
                c="tab:blue",
                lw=1.5,
            )

            # Plot the duplicated time stamps as scatter points
            duplications = data_fall.loc[data_fall[time_col].isin(time_duplications)]
            if duplications.shape[0] > 0:
                ix_filter, time_stamps = _remove_tz(duplications, time_col)
                ax.scatter(
                    time_stamps[ix_filter],
                    duplications.loc[ix_filter, power_col],
                    c="tab:blue",
                    label="Original Timestamp Duplicates",
                )

            # Find bad timestamps, then fill in any potential UTC time gaps due the focus on the input time field
            ix_filter, time_stamps = _remove_tz(data_fall, t_utc)
            data_fall = ts.gap_fill_data_frame(data_fall[ix_filter], t_utc, freq)
            ix_filter, time_stamps = _remove_tz(data_fall, t_utc)
            ax.plot(
                time_stamps[ix_filter],
                data_fall.loc[ix_filter, power_col],
                label="UTC Timestamp",
                c="tab:orange",
                linestyle="--",
            )

            # Plot the duplicated time stamps as scatter points
            duplications = data_fall.loc[data_fall[t_utc].isin(time_duplications_utc)]
            if duplications.shape[0] > 0:
                ix_filter, time_stamps = _remove_tz(duplications, t_utc)
                ax.scatter(
                    time_stamps[ix_filter],
                    duplications.loc[ix_filter, power_col],
                    c="tab:orange",
                    label="UTC Timestamp Duplicates",
                )

        ax.set_title(f"{year}, Fall")
        ax.set_ylabel("Power")
        ax.set_xlabel("Date")
        ax.legend(loc="lower left", fontsize=10)
        ax.tick_params(axis="x", rotation=20)

    if j < (num_axes := axes.size):
        diff = num_axes - j
        for i in range(1, diff + 1):
            fig.delaxes(axes.flatten()[num_axes - i])

    # fig.tight_layout()
    plt.show()


def wtk_coordinate_indices(
    fn: h5pyd.File, latitude: float, longitude: float
) -> tuple[float, float]:
    """Finds the nearest x/y coordinates for a given latitude and longitude using the Proj4 library
    to find the nearest valid point in the Wind Toolkit coordinates database, and converts it to
    an (x, y) pair.

    ... note:: This relies on the Wind Toolkit HSDS API and h5pyd must be installed.

    Args:
        fn (:obj: `h5pyd.File`): The h5pyd file to be used for coordinate extraction.
        latitude (:obj: `float`): The latitude of the wind power plant's center.
        longitude (:obj: `float`): The longitude of the wind power plant's center.

    Returns:
        tuple[float, float]: The nearest valid x and y coordinates to the provided `latitude` and
            `longitude`.
    """
    coordinates = fn["coordinates"]
    project_coord_string = """
        +proj=lcc +lat_1=30 +lat_2=60
        +lat_0=38.47240422490422 +lon_0=-96.0
        +x_0=0 +y_0=0 +ellps=sphere
        +units=m +no_defs
    """
    projectLcc = Proj(project_coord_string)
    origin = projectLcc(*reversed(coordinates[0][0]))  # Grab origin directly from database

    project_coords = projectLcc(longitude, latitude)
    delta = np.subtract(project_coords, origin)
    xy = reversed([int(round(x / 2000)) for x in delta])
    return tuple(xy)


def wtk_diurnal_prep(
    latitude: float,
    longitude: float,
    fn: str = "/nrel/wtk-us.h5",
    start_date: str = "2007-01-01",
    end_date: str = "2013-12-31",
) -> pd.Series:
    """Links to the WIND Toolkit (WTK) data on AWS as a data source to capture the wind speed data
    and calculate the diurnal hourly averages.

    Args:
        latitude (:obj: `float`): The latitude of the wind power plant's center.
        longitude (:obj: `float`): The longitude of the wind power plant's center.
        fn (:obj: `str`, optional): The path and name of the WTK API file. Defaults to "/nrel/wtk-us.h5".
        start_date (:obj: `str`, optional): Starting date for the WTK data. Defaults to "2007-01-01".
        end_date (:obj: `str`, optional): Ending date for the WTK data. Defaults to "2013-12-31".

    Raises:
        IndexError: Raised if the latitude and longitude are not found within the WTK data set.

    Returns:
        pd.Series: The diurnal hourly average wind speed.
    """
    # Startup the API and grab the database
    f = h5pyd.File(fn, "r")
    wtk_coordinates = f["coordinates"]
    wtk_dt = f["datetime"]
    wtk_ws = f["windspeed_80m"]

    # Set up the date and time requirements
    dt = pd.DataFrame({"datetime": wtk_dt}, index=range(wtk_dt.shape[0]))
    dt["datetime"] = dt["datetime"].apply(dateutil.parser.parse)

    project_ix = wtk_coordinate_indices(f, latitude, longitude)
    try:
        _ = wtk_coordinates[project_ix[0]][project_ix[1]]
    except ValueError:
        msg = f"Project Coordinates (lat, long) = ({latitude}, {longitude}) are outside the WIND Toolkit domain."
        raise IndexError(msg)

    window_ix = dt.loc[(dt.datetime >= start_date) & (dt.datetime <= end_date)].index
    ws = pd.DataFrame(
        wtk_ws[min(window_ix) : max(window_ix) + 1, project_ix[0], project_ix[1]],
        columns=["ws"],
        index=dt.loc[window_ix, "datetime"],
    )
    ws_diurnal = ws.groupby(ws.index.hour).mean()
    return ws_diurnal


def wtk_diurnal_plot(
    wtk_df: pd.DataFrame | None,
    scada_df: pd.DataFrame,
    time_col: str,
    power_col: str,
    *,
    latitude: float = 0,
    longitude: float = 0,
    fn: str = "/nrel/wtk-us.h5",
    start_date: str = "2007-01-01",
    end_date: str = "2013-12-31",
    return_fig: bool = False,
) -> None:
    """Plots the WTK diurnal wind profile alongside the hourly power averages from the :py:attr:`scada_df`

    Args:
        wtk_df (:obj: `pd.DataFrame` | `None`): The WTK diurnal profile data produced in
            `wtk_diurnal_prep`. If `None`, then this method will be run internally as the following
            keyword arguments are provided: :py:attr:`latitude`, :py:attr:`longitude`, :py:attr:`fn`,
            :py:attr:`start_date`, and :py:attr:`end_date`.
        scada_df (:obj: `pd.DataFrame` | None): The SCADA data that was produced in :py:meth:`convert_datetime_column`.
        time_col (:obj: `str`): The name of the time column in :py:attr:`scada_df`.
        power_col (:obj: `str`): The name of the power column in :py:attr:`scada_df`
        latitude (:obj: `float`): The latitude of the wind power plant's center.
        longitude (:obj: `float`): The longitude of the wind power plant's center.
        fn (:obj: `str`, optional): WTK API file path and location. Defaults to "/nrel/wtk-us.h5".
        start_date (:obj: `str` | None, optional): Starting date for the WTK data. If None, then it
            uses the starting date of :py:attr:`scada_df`. Defaults to None.
        end_date (:obj: `str` | None, optional): Ending date for the WTK data. If None, then it
            uses the ending date of :py:attr:`scada_df`. Defaults to None.
        return_fig(:obj:`String`): Indicator for if the figure and axes objects should be returned,
            by default False.
    """
    # Get the WTK data if needed
    if wtk_df is None:
        if latitude == longitude == 0:
            raise ValueError(
                "If `wtk_df` is not provided, then the WTK accessor information must "
                "be provided to create the data set"
            )
        else:
            utc_time = f"{time_col}_utc"
            if start_date is None:
                start_date = scada_df[utc_time].min()
            if end_date is None:
                end_date = scada_df[utc_time].max()
            wtk_df = wtk_diurnal_prep(latitude, longitude, fn, start_date, end_date)

    sum_power_df = scada_df.groupby(scada_df.index).sum()[[power_col]]
    power_diurnal_df = sum_power_df.groupby(sum_power_df.index.hour).mean()[[power_col]]

    ws_norm = wtk_df / wtk_df.mean()
    power_norm = power_diurnal_df / power_diurnal_df.mean()

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(ws_norm, label="WTK Wind Speed")
    ax.plot(power_norm, label="Measured Power")

    ax.legend()

    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Normalized Values")
    ax.set_title("WTK and Measured Data Comparison")
    fig.tight_layout()
    plt.show()
    if return_fig:
        return fig, ax
