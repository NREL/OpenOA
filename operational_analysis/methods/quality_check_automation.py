"""Creates the QualityControlDiagnosticSuite and specific data-based subclasses."""

from abc import abstractmethod
from typing import List, Tuple, Union
from datetime import time, datetime

import pytz
import h5pyd
import numpy as np
import pandas as pd
import dateutil
import matplotlib.pyplot as plt
from pyproj import Proj
from dateutil import tz

from operational_analysis import logging, logged_method_call
from operational_analysis.toolkits import timeseries


Number = Union[int, float]
logger = logging.getLogger(__name__)


def _read_data(data: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """Takes the `DataFrame` or file path and returns a `DataFrame`

    Args:
     data(:obj: `Union[pd.DataFrame, str]`): The actual data or a path to the csv data.

    Returns
     (:obj: `pd.DataFrame`): The data fram object.
    """
    if isinstance(data, pd.DataFrame):
        return data
    return pd.read_csv(data)


def _remove_tz(df: pd.DataFrame, t_local_column: str) -> Tuple[np.ndarray, np.ndarray]:
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


class QualityControlDiagnosticSuite:
    """This class defines key analytical procedures in a quality check process for turbine data.
    After analyzing the data for missing and duplicate timestamps, timezones, Daylight Savings Time
    corrections, and extrema values, the user can make informed decisions about how to handle the data.
    """

    @logged_method_call
    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        ws_field: str = "wmet_wdspd_avg",
        power_field: str = "wtur_W_avg",
        time_field: str = "datetime",
        id_field: str = None,
        freq: str = "10T",
        lat_lon: Tuple[Number, Number] = (0, 0),
        local_tz: str = "UTC",
        timezone_aware: bool = False,
    ):
        """
        Initialize QCAuto object with data and parameters.

        Args:
         data(:obj: `Union[pd.DataFrame, str]`): The actual data or a path to the csv data.
         ws_field(:obj: 'String'): String name of the windspeed field to df
         power_field(:obj: 'String'): String name of the power field to df
         time_field(:obj: 'String'): String name of the time field to df
         id_field(:obj: 'String'): String name of the id field to df
         freq(:obj: 'String'): String representation of the resolution for the time field to df
         lat_lon(:obj: 'tuple'): latitude and longitude of farm represented as a tuple; this is
            purely informational.
         local_tz(:obj: 'String'): The `pytz`-compatible timezone for the input `time_field`, by
            default UTC. This should be in the format of "Country/City" or "Region/City" such as
            "America/Denver" or "Europe/Paris".
         timezone_aware(:obj: `bool`): If True, this indicates the `time_field` column has timezone
            information embedded, and if False, then there is no timezone information, by default False.
        """

        logger.info("Initializing QC_Automation Object")

        self._df = _read_data(data)
        self._ws = ws_field
        self._w = power_field
        self._t = time_field
        self._t_utc = f"{time_field}_utc"
        self._t_local = f"{time_field}_localized"
        self._id = id_field
        self._freq = freq
        self._lat_lon = lat_lon
        self._local_tz = local_tz
        self._local_ptz = pytz.timezone(local_tz)
        self._tz_aware = timezone_aware
        self._offset = "utc_offset"
        self._dst = "is_dst"
        self._non_dst_offset = self._local_ptz.localize(datetime(2021, 1, 1)).utcoffset()

        if self._id is None:
            self._id = "ID"
            self._df["ID"] = "Data"

        self._convert_datetime_column()

    def _determine_offset_dst(self, df: pd.DataFrame) -> None:
        """Creates a column of "utc_offset" and "is_dst".

        Args:
         df(:obj:`pd.DataFrame`): The dataframe object to manipulate.

        Returns:
         (:obj:`pd.DataFrame`): The updated dataframe with "utc_offset" and "is_dst" columns created.
        """
        dt = df.copy().tz_convert(self._local_tz)
        dt_col = dt.index.to_pydatetime()

        # Determine the Daylight Savings Time status and UTC offset
        dt[self._offset] = [el.utcoffset() for el in dt_col]
        dt[self._dst] = (dt[self._offset] != self._non_dst_offset).astype(bool)

        # Convert back to UTC
        dt = dt.tz_convert("UTC")
        return dt

    def _convert_datetime_column(self) -> None:
        """Converts the passed timestamp data to a pandas-encoded Datetime, and creates a
        corresponding localized and UTC timestamp using the `time_field` column name with either
        "localized" or "utc", respectively. The `_df` object then uses the local timezone
        timestamp for its index.
        """
        # Convert the timestamps to datetime.datetime objects
        dt_col = self._df[self._t].values

        # Check for raw timestamp inputs or pre-formatted
        if isinstance(dt_col[0], str):
            dt_col = [dateutil.parser.parse(el) for el in dt_col]

        # Read the timestamps as UTC, then convert to the local timezone if the data are
        # timezone-aware, otherwise localize the timestamp to the local timezone
        if self._tz_aware:
            pd_dt_col = pd.to_datetime(dt_col, utc=True).tz_convert(self._local_tz)
            self._df[self._t_local] = pd_dt_col
        else:
            pd_dt_col = pd.to_datetime(dt_col)
            self._df[self._t_local] = pd_dt_col.tz_localize(self._local_tz, ambiguous=True)

        self._df[self._t] = pd_dt_col
        self._df = self._df.set_index(pd.DatetimeIndex(self._df[self._t_local]))

        # Create the UTC-converted time-stamp
        try:
            self._df[self._t_utc] = pd.to_datetime(
                [el.astimezone(tz.tzutc()) for el in dt_col]
            ).tz_convert("UTC")
        except AttributeError:  # catches numpy datetime error for astimezone() not existing
            self._df = self._df.tz_convert("UTC")
            self._df[self._t_utc] = self._df.index

        # Adjust the index name to reflect the change to a UTC-based timestamp
        self._df.index.name = self._t_utc

        self._df = self._determine_offset_dst(self._df)

    @abstractmethod
    @logged_method_call
    def run(self):
        """
        Run the QC analysis functions in order by calling this function. In the base
        class, this operates as a basic demonstration of features, so it is encouraged
        that users devise a method to enhance their data checking.
        """
        logger.info("Identifying Time Duplications")
        self.dup_time_identification()
        logger.info("Identifying Time Gaps")
        self.gap_time_identification()
        logger.info("Isolating Extrema Values")
        self.max_min()
        logger.info("QC Diagnostic Complete")

    def dup_time_identification(self):
        """
        This function identifies any time duplications in the dataset for original, localized, and UTC timestamps.
        """
        self._time_duplications = self._df.loc[
            self._df.duplicated(subset=[self._id, self._t]), self._t
        ]

        self._time_duplications_local = self._df.loc[
            self._df.duplicated(subset=[self._id, self._t_local]), self._t_local
        ]

        self._time_duplications_utc = self._df.loc[
            self._df.duplicated(subset=[self._id, self._t_utc]), self._t_utc
        ]

    def gap_time_identification(self):
        """
        This function identifies any time gaps in the dataset for original, localized, and UTC timestamps.
        """
        self._time_gaps = timeseries.find_time_gaps(self._df[self._t], freq=self._freq)
        self._time_gaps_local = timeseries.find_time_gaps(self._df[self._t_local], freq=self._freq)
        self._time_gaps_utc = timeseries.find_time_gaps(self._df[self._t_utc], freq=self._freq)

    def max_min(self):

        """
        This function creates a DataFrame that contains the max and min values for each column

        Args:
        (None)

        Returns:
        (None)
        """

        self._max_min = pd.DataFrame(index=self._df.columns, columns={"max", "min"})
        self._max_min["max"] = self._df.max()
        self._max_min["min"] = self._df.min()

    def daylight_savings_plot(self, hour_window=3):

        """
        Produce a timeseries plot showing daylight savings events for each year using the passed data.

        Args:
            hour_window(:obj: 'int'): number of hours, before and after the Daylight Savings Time
            transitions to view in the plot, by default 3.

        Returns:
            (None)
        """
        # Get data for one of the turbines
        self._df_dst = self._df.loc[self._df[self._id] == self._df[self._id].unique()[0]]
        df_full = self._df_dst.copy()

        # Locate the missing timestamps, convert to UTC, and recreate DST and UTC-offset columns
        missing_original = timeseries.find_time_gaps(self._df_dst[self._t], self._freq)
        missing_local = timeseries.find_time_gaps(self._df_dst[self._t_local], self._freq)
        missing_utc = timeseries.find_time_gaps(self._df_dst[self._t_utc], self._freq)

        missing_df = pd.DataFrame([], columns=self._df.columns)
        missing_df.loc[:, self._t_utc] = missing_utc
        missing_df.loc[:, self._t_local] = missing_local
        missing_df.loc[:, self._t] = missing_original
        missing_df = missing_df.set_index(self._t_utc, drop=False)

        # Append and resort the missing timestamps, then convert to local time
        df_full = df_full.append(missing_df).sort_values(self._t)
        try:
            df_full = df_full.tz_convert(self._local_tz)
        except TypeError:
            pass
        self._df_full = df_full

        years = df_full[self._t].dt.year.unique().astype(int)  # Years in data record
        num_years = len(years)
        hour_window = pd.Timedelta(hours=hour_window)
        plt.figure(figsize=(14, 20))

        for i, year in enumerate(years):
            year_data = df_full.loc[df_full[self._t].dt.year == year]
            dst_dates = np.where(year_data[self._dst].values)[0]

            # Break the plotting loop if there is a partial year without DST in the data
            if dst_dates.size == 0:
                break

            # Get the start and end DatetimeIndex values
            start_ix = year_data.iloc[dst_dates[0]].name
            end_ix = year_data.iloc[dst_dates[-1] + 1].name

            # Create the data subsets for plotting the appropriate window
            data_spring = _get_time_window(
                year_data, start_ix, hour_window, self._t, self._t_local, self._t_utc
            )
            data_fall = _get_time_window(
                year_data, end_ix, hour_window, self._t, self._t_local, self._t_utc
            )

            data_spring = data_spring.sort_values(
                [self._t, self._w], na_position="first"
            ).drop_duplicates(subset=self._t, keep="last")
            data_fall = data_fall.sort_values(
                [self._t, self._w], na_position="first"
            ).drop_duplicates(subset=self._t, keep="last")

            # Plot each as side-by-side subplots
            plt.subplot(num_years, 2, 2 * i + 1)
            if np.sum(~np.isnan(data_spring[self._w])) > 0:
                # For localized time, we want to ensure we're capturing the DST switch as missing data
                ix_filter, time_stamps = _remove_tz(data_spring, self._t)
                time_stamps = time_stamps[ix_filter]
                power_data = data_spring.loc[ix_filter, self._w].tolist()

                # Find the missing data points on the timezone stripped data and append
                # it to the time stamps, then identify where to insert NaN in the power data
                missing = timeseries.find_time_gaps(time_stamps, self._freq)
                missing = pd.to_datetime(missing.values).to_pydatetime()
                time_stamps = np.append(time_stamps, missing)
                time_stamps.sort()
                nan_ix = sorted([np.where(el == time_stamps)[0][0] for el in missing])
                for ix in nan_ix:
                    power_data.insert(ix, float("nan"))

                plt.plot(
                    time_stamps,
                    power_data,
                    label="Original Timestamp",
                    c="tab:blue",
                    lw=1.5,
                )

                # Plot the duplicated time stamps as scatter points
                duplications = data_spring.loc[data_spring[self._t].isin(self._time_duplications)]
                if duplications.shape[0] > 0:
                    ix_filter, time_stamps = _remove_tz(duplications, self._t)
                    plt.scatter(
                        time_stamps[ix_filter],
                        duplications.loc[ix_filter, self._w],
                        c="tab:blue",
                        label="Original Timestamp Duplicates",
                    )

                # Find bad timestamps, then fill in any potential UTC time gaps due the focus on the input time field
                ix_filter, time_stamps = _remove_tz(data_spring, self._t_utc)
                data_spring = timeseries.gap_fill_data_frame(
                    data_spring[ix_filter], self._t_utc, self._freq
                )
                ix_filter, time_stamps = _remove_tz(data_spring, self._t_utc)
                plt.plot(
                    time_stamps[ix_filter],
                    data_spring.loc[ix_filter, self._w],
                    label="UTC Timestamp",
                    c="tab:orange",
                    linestyle="--",
                )

                # Plot the duplicated time stamps as scatter points
                duplications = data_spring.loc[
                    data_spring[self._t_utc].isin(self._time_duplications_utc)
                ]
                if duplications.shape[0] > 0:
                    ix_filter, time_stamps = _remove_tz(duplications, self._t_utc)
                    plt.scatter(
                        time_stamps[ix_filter],
                        duplications.loc[ix_filter, self._w],
                        c="tab:orange",
                        label="UTC Timestamp Duplicates",
                    )

            plt.title(f"{year}, Spring")
            plt.ylabel("Power")
            plt.xlabel("Date")
            plt.legend(loc="lower left")

            plt.subplot(num_years, 2, 2 * i + 2)
            if np.sum(~np.isnan(data_fall[self._w])) > 0:
                ix_filter, time_stamps = _remove_tz(data_fall, self._t)
                plt.plot(
                    time_stamps[ix_filter],
                    data_fall.loc[ix_filter, self._w],
                    label="Original Timestamp",
                    c="tab:blue",
                    lw=1.5,
                )

                # Plot the duplicated time stamps as scatter points
                duplications = data_fall.loc[data_fall[self._t].isin(self._time_duplications)]
                if duplications.shape[0] > 0:
                    ix_filter, time_stamps = _remove_tz(duplications, self._t)
                    plt.scatter(
                        time_stamps[ix_filter],
                        duplications.loc[ix_filter, self._w],
                        c="tab:blue",
                        label="Original Timestamp Duplicates",
                    )

                # Find bad timestamps, then fill in any potential UTC time gaps due the focus on the input time field
                ix_filter, time_stamps = _remove_tz(data_fall, self._t_utc)
                data_fall = timeseries.gap_fill_data_frame(
                    data_fall[ix_filter], self._t_utc, self._freq
                )
                ix_filter, time_stamps = _remove_tz(data_fall, self._t_utc)
                plt.plot(
                    time_stamps[ix_filter],
                    data_fall.loc[ix_filter, self._w],
                    label="UTC Timestamp",
                    c="tab:orange",
                    linestyle="--",
                )

                # Plot the duplicated time stamps as scatter points
                duplications = data_fall.loc[
                    data_fall[self._t_utc].isin(self._time_duplications_utc)
                ]
                if duplications.shape[0] > 0:
                    ix_filter, time_stamps = _remove_tz(duplications, self._t_utc)
                    plt.scatter(
                        time_stamps[ix_filter],
                        duplications.loc[ix_filter, self._w],
                        c="tab:orange",
                        label="UTC Timestamp Duplicates",
                    )

            plt.title(f"{year}, Fall")
            plt.ylabel("Power")
            plt.xlabel("Date")
            plt.legend(loc="lower left")

        plt.tight_layout()
        plt.show()

    def plot_by_id(self, x_axis=None, y_axis=None):

        """
        This is generalized function that allows the user to plot any two fields against each other
        with unique plots for each unique ID. For scada data, this function produces turbine plots
        and for meter data, this will return a single plot.

        Args:
            x_axis(:obj:'String'): Independent variable to plot (default is windspeed field)
            y_axis(:obj:'String'): Dependent variable to plot (default is power field)

        Returns:
            (None)
        """
        if x_axis is None:
            x_axis = self._ws

        if y_axis is None:
            y_axis = self._w

        turbs = self._df[self._id].unique()
        num_turbs = len(turbs)
        num_rows = np.ceil(num_turbs / 4.0)

        plt.figure(figsize=(15, num_rows * 5))
        n = 1
        for t in turbs:
            plt.subplot(num_rows, 4, n)
            scada_sub = self._df.loc[self._df[self._id] == t, :]
            plt.scatter(scada_sub[x_axis], scada_sub[y_axis], s=5)
            n = n + 1
            plt.title(t)
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
        plt.tight_layout()
        plt.show()

    def column_histograms(self):
        """
        Produces histogram plot for each numeric column.

        Args:
            (None)

        Returns:
            (None)
        """

        for c in self._df.columns:
            if (self._df[c].dtype == float) | (self._df[c].dtype == int):
                # plt.subplot(2,2,n)
                plt.figure(figsize=(8, 6))
                plt.hist(self._df[c].dropna(), 40)
                # n = n + 1
                plt.title(c)
                plt.ylabel("Count")
                plt.show()


class WindToolKitQualityControlDiagnosticSuite(QualityControlDiagnosticSuite):
    """This is a child class of the `QualityControlDiagnosticSuite` that adds in
    WindToolKit (WTK) specific functionality for checking datetime data, and the
    diurnality of the input power data.
    """

    @logged_method_call
    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        ws_field: str = "wmet_wdspd_avg",
        power_field: str = "wtur_W_avg",
        time_field: str = "datetime",
        id_field: str = None,
        freq: str = "10T",
        lat_lon: Tuple[Number, Number] = (0, 0),
        local_tz: str = "UTC",
        timezone_aware: bool = False,
        check_tz: bool = False,
    ):
        """
        Initialize QCAuto object with data and parameters.

        Args:
         data(:obj: `Union[pd.DataFrame, str]`): The actual data or a path to the csv data.
         ws_field(:obj: 'String'): String name of the windspeed field to df
         power_field(:obj: 'String'): String name of the power field to df
         time_field(:obj: 'String'): String name of the time field to df
         id_field(:obj: 'String'): String name of the id field to df
         freq(:obj: 'String'): String representation of the resolution for the time field to df
         lat_lon(:obj: 'tuple'): latitude and longitude of farm represented as a tuple; this is
            purely informational.
         local_tz(:obj: 'String'): The `pytz`-compatible timezone for local time reference, by
            default UTC. This should be in the format of "Country/City" or "Region/City" such as
            "America/Denver" or "Europe/Paris".
         timezone_aware(:obj: `bool`): If True, this indicates the `time_field` column has timezone
            information embedded, and if False, then there is no timezone information, by default False.
         check_tz(:obj: 'bool'): Boolean on whether to use WIND Toolkit data to assess timezone of
            data, by default False. This should only be set to `True` when the `lat_lon` fall within
            the WIND Toolkit parameters.
        """
        super().__init__(
            data=data,
            ws_field=ws_field,
            power_field=power_field,
            time_field=time_field,
            id_field=id_field,
            freq=freq,
            lat_lon=lat_lon,
            local_tz=local_tz,
            timezone_aware=timezone_aware,
        )
        self._check_tz = check_tz

    @logged_method_call
    def run(self):
        """
        Run the QC analysis functions in order by calling this function.

        Args:
            (None)

        Returns:
            (None)
        """

        logger.info("Identifying Time Duplications")
        self.dup_time_identification()
        logger.info("Identifying Time Gaps")
        self.gap_time_identification()

        if self._check_tz:
            logger.info("Evaluating timezone deviation from UTC")
            try:
                self.ws_diurnal_prep()
                self.corr_df_calc()
            except IndexError as e:
                logger.info(str(e))

        logger.info("Isolating Extrema Values")
        self.max_min()
        logger.info("QC Diagnostic Complete")

    def indicesForCoord(self, f):

        """
        This function finds the nearest x/y indices for a given lat/lon.
        Rather than fetching the entire coordinates database, which is 500+ MB, this
        uses the Proj4 library to find a nearby point and then converts to x/y indices.
        This function relies on the Wind Toolkit HSDS API.

        Args:
            f (h5 file): file to be read in

        Returns:
            x and y coordinates corresponding to a given lat/lon as a tuple
        """

        dset_coords = f["coordinates"]
        projstring = """+proj=lcc +lat_1=30 +lat_2=60
                    +lat_0=38.47240422490422 +lon_0=-96.0
                    +x_0=0 +y_0=0 +ellps=sphere
                    +units=m +no_defs """
        projectLcc = Proj(projstring)
        origin_ll = reversed(dset_coords[0][0])  # Grab origin directly from database
        origin = projectLcc(*origin_ll)

        lat, lon = self._lat_lon
        coords = (lon, lat)
        coords = projectLcc(*coords)
        delta = np.subtract(coords, origin)
        ij = [int(round(x / 2000)) for x in delta]
        return tuple(reversed(ij))

    def ws_diurnal_prep(self, start_date="2007-01-01", end_date="2013-12-31"):

        """
        This method links into Wind Toolkit data on AWS as a data source, grabs wind speed data,
        and calculates diurnal hourly averages. These diurnal hourly averages are returned as a
        Pandas series.

        Args:
            start_date(:obj:'String'): start date to diurnal analysis (optional)
            end_date(:obj:'String'): end date to diurnal analysis (optional)


        Returns:
            ws_diurnal (Pandas Series): Series where each index corresponds to a different hour of
            the day and each value corresponds to the average windspeed
        """

        f = h5pyd.File("/nrel/wtk-us.h5", "r")

        # Setup date and time
        dt = f["datetime"]
        dt = pd.DataFrame({"datetime": dt[:]}, index=range(0, dt.shape[0]))
        dt["datetime"] = dt["datetime"].apply(dateutil.parser.parse)

        project_idx = self.indicesForCoord(f)

        try:
            coordinates = f["coordinates"][project_idx[0]][project_idx[1]]
        except ValueError:
            message = "Project coordinates are outside of the WIND Toolkit domain; aborting diurnal methods!"
            raise IndexError(message)

        print("y,x indices for project: \t\t {}".format(project_idx))
        print("Coordinates of project: \t {}".format(self._lat_lon))
        print("Coordinates of project: \t {}".format(coordinates))

        # Get wind speed at 80m from the specified lat/lon
        ws = f["windspeed_80m"]
        t_range = dt.loc[(dt.datetime >= start_date) & (dt.datetime < end_date)].index

        # Convert to dataframe
        ws_tseries = ws[min(t_range) : max(t_range) + 1, project_idx[0], project_idx[1]]
        ws_df = pd.DataFrame(index=dt.loc[t_range, "datetime"], data={"ws": ws_tseries})

        # Calculate diurnal profile of wind speed
        ws_diurnal = ws_df.groupby(ws_df.index.hour).mean()

        self._wtk_ws_diurnal = ws_diurnal

    def wtk_diurnal_plot(self):

        """
        This method plots the WTK diurnal plot alongisde the hourly power averages of the df across all turbines

        Args:
            (None)

        Returns:
            (None)
        """

        sum_df = self._df.groupby(self._df[self._t])[self._w].sum().to_frame()
        # df_temp = sum_df.copy()
        # df_temp[self._t] = df_temp.index

        # df_diurnal = df_temp.groupby(df_temp[self._t].dt.hour)[self._w].mean()
        df_diurnal = sum_df.groupby(sum_df.index.hour)[self._w].mean()

        ws_norm = self._wtk_ws_diurnal / self._wtk_ws_diurnal.mean()
        df_norm = df_diurnal / df_diurnal.mean()

        plt.figure(figsize=(8, 5))
        plt.plot(ws_norm, label="WTK wind speed (UTC)")
        plt.plot(df_norm, label="QC power")
        plt.grid()
        plt.xlabel("Hour of day")
        plt.ylabel("Normalized values")
        plt.title("WTK and QC Timezone Comparison")
        plt.legend()
        plt.show()

    def corr_df_calc(self):
        """
        This method creates a correlation series that compares the current power data (with
        different shift thresholds) to wind speed data from the WTK with hourly resolution.

        Args:
            (None)

        Returns:
            (None)
        """

        self._df_diurnal = self._df.groupby(self._df[self._t].dt.hour)[self._w].mean()
        return_corr = np.empty((24))

        for i in np.arange(24):
            return_corr[i] = np.corrcoef(self._wtk_ws_diurnal["ws"], np.roll(self._df_diurnal, i))[
                0, 1
            ]

        self._hour_shift = pd.DataFrame(index=np.arange(24), data={"corr_by_hour": return_corr})
