# TODO:
# - Finish refactoring from Timeseries Table to PlantData (test and verify)
# - Rename API to openoa.analysis.aep.long_term_monte_carlo_method
# - Make AnalysisResult into an attrs dataclass, and Analysis object.
# - Refactor the class to be more modular and performant. Add QMC method.

from __future__ import annotations

import random
import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.markers import MarkerStyle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from openoa import PlantData, logging, logged_method_call
from openoa.utils import plot, filters
from openoa.utils import timeseries as tm
from openoa.utils import unit_conversion as un
from openoa.utils import met_data_processing as mt
from openoa.utils.machine_learning_setup import MachineLearningSetup


logger = logging.getLogger(__name__)


def get_annual_values(data):
    """
    This function returns annual summations of values in a pandas Series (or each column of a pandas DataFrame) with a
    DatetimeIndex index starting from the first row. The purpose of the function is to correctly resample to annual
    values when the first index does not fall on the beginning of the month.

    Args:
        data(:obj:`pandas.Series` or :obj:`pandas.DataFrame`): Input data with a DatetimeIndex index.

    Returns:
        :obj:`numpy.ndarray`: Array containing annual summations for each column of the input data.
    """

    # shift time index to beginning of first month so resampling by 'MS' groups the data into full years
    # starting from the beginning of the time series
    ix_start = data.index[0]
    month_start = ix_start.floor("d") + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)
    data.index = data.index - pd.Timedelta(ix_start - month_start)

    return data.resample("12MS").sum().values


class MonteCarloAEPResult(object):
    """
    Result object of a MonteCarlo AEP Analysis
    """

    pass


# Long Term AEP
class MonteCarloAEP(object):
    """
    A serial (Pandas-driven) implementation of the benchmark PRUF operational
    analysis implementation. This module collects standard processing and
    analysis methods for estimating plant level operational AEP and uncertainty.

    The preprocessing should run in this order:
        1. Process revenue meter energy - creates monthly/daily data frame, gets revenue meter on monthly/daily basis, and adds
           data flag
        2. Process loss estimates - add monthly/daily curtailment and availabilty losses to monthly/daily data frame

        3. Process reanalysis data - add monthly/daily density-corrected wind speeds, temperature (if used) and wind direction (if used)
           from several reanalysis products to the monthly data frame

        4. Set up Monte Carlo - create the necessary Monte Carlo inputs to the OA process

        5. Run AEP Monte Carlo - run the OA process iteratively to get distribution of AEP results

    The end result is a distribution of AEP results which we use to assess expected AEP and associated uncertainty
    """

    @logged_method_call
    def __init__(
        self,
        plant: PlantData,
        reanal_products: list[str] = ["merra2", "ncep2", "erai", "era5"],
        uncertainty_meter: float = 0.005,
        uncertainty_losses: float = 0.05,
        uncertainty_windiness: tuple[int, int] = (10, 20),
        uncertainty_loss_max: tuple[int, int] = (10, 20),
        outlier_detection: bool = False,
        uncertainty_outlier: tuple[float] = (1, 3),
        uncertainty_nan_energy: float = 0.01,
        time_resolution: str = "M",
        end_date_lt: str | pd.Timestamp = None,
        reg_model: str = "lin",
        ml_setup_kwargs: dict = {},
        reg_temperature: bool = False,
        reg_winddirection: bool = False,
    ):
        """
        Initialize APE_MC analysis with data and parameters.

        Args:
         plant(:obj:`PlantData object`): PlantData object from which PlantAnalysis should draw data.
         reanal_products(obj:`list`) : List of reanalysis products to use for Monte Carlo sampling. Defaults to ["merra2", "ncep2", "erai"].
         uncertainty_meter(:obj:`float`): uncertainty on revenue meter data
         uncertainty_losses(:obj:`float`): uncertainty on long-term losses
         uncertainty_windiness(:obj:`tuple`): number of years to use for the windiness correction
         uncertainty_loss_max(:obj:`tuple`): threshold for the combined availabilty and curtailment monthly loss threshold
         outlier_detection(:obj:`bool`): whether to perform (True) or not (False - default) outlier detection filtering
         uncertainty_outlier(:obj:`tuple`): min and max thresholds (Monte-Carlo sampled) for the outlier detection filter. At monthly resolution, this is the tuning constant for Huber’s t function for a robust linear regression. At daily/hourly resolution, this is the number of stdev of wind speed used as threshold for the bin filter.
         uncertainty_nan_energy(:obj:`float`): threshold to flag days/months based on NaNs
         time_resolution(:obj:`string`): whether to perform the AEP calculation at monthly ('M'), daily ('D') or hourly ('H') time resolution
         end_date_lt(:obj:`string` or :obj:`pandas.Timestamp`): The last date to use for the long-term correction. Note that only the component of the date corresponding to the time_resolution argument is considered. If None, the end of the last complete month of reanalysis data will be used.
         reg_model(:obj:`string`): which model to use for the regression ('lin' for linear, 'gam', 'gbm', 'etr'). At monthly time resolution only linear regression is allowed because of the reduced number of data points.
         ml_setup_kwargs(:obj:`kwargs`): keyword arguments to MachineLearningSetup class
         reg_temperature(:obj:`bool`): whether to include temperature (True) or not (False) as regression input
         reg_winddirection(:obj:`bool`): whether to include wind direction (True) or not (False) as regression input
        """
        logger.info("Initializing MonteCarloAEP Analysis Object")

        self._aggregate = pd.DataFrame()
        self._plant = plant  # defined at runtime
        self._reanal_products = reanal_products  # set of reanalysis products to use

        # Memo dictionaries help speed up computation
        self.outlier_filtering = {}  # Combinations of outlier filter results
        self.long_term_sampling = {}  # Combinations of long-term reanalysis data sampling
        self.opt_model = {}  # Optimized ML model hyperparameters for each reanalysis product

        # Define relevant uncertainties, data ranges and max thresholds to be applied in Monte Carlo sampling
        self.uncertainty_meter = np.float64(uncertainty_meter)
        self.uncertainty_losses = np.float64(uncertainty_losses)
        self.uncertainty_windiness = np.array(uncertainty_windiness, dtype=np.float64)
        self.uncertainty_outlier = np.array(uncertainty_outlier, dtype=np.float64)
        self.uncertainty_loss_max = np.array(uncertainty_loss_max, dtype=np.float64)
        self.uncertainty_nan_energy = np.float64(uncertainty_nan_energy)
        self.outlier_detection = outlier_detection

        # Check that selected time resolution is allowed
        if time_resolution not in ["M", "D", "H"]:
            raise ValueError(
                "time_res has to either be M (monthly, default) or D (daily) or H (hourly)"
            )
        self.time_resolution = time_resolution
        self._resample_freq = {"M": "MS", "D": "D", "H": "H"}[self.time_resolution]
        self._hours_in_res = {"M": 30 * 24, "D": 1 * 24, "H": 1}[self.time_resolution]
        self._calendar_samples = {"M": 12, "D": 365, "H": 365 * 24}[self.time_resolution]
        self.num_days_lt = (31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

        if end_date_lt is not None:
            self.end_date_lt = pd.to_datetime(end_date_lt).replace(minute=0)  # drop minute field
        else:
            self.end_date_lt = end_date_lt

        # Check that choices for regression inputs are allowed
        if reg_temperature not in [True, False]:
            raise ValueError(
                "reg_temperature has to either be True (if temperature is considered in the regression), or False (if temperature is omitted"
            )
        if reg_winddirection not in [True, False]:
            raise ValueError(
                "reg_winddirection has to either be True (if wind direction is considered in the regression), or False (if wind direction is omitted"
            )
        self.reg_winddirection = reg_winddirection
        self.reg_temperature = reg_temperature

        # Build list of regression variables
        self._rean_vars = []
        if self.reg_temperature:
            self._rean_vars += ["temperature"]
        if self.reg_winddirection:
            self._rean_vars += ["windspeed_u", "windspeed_v"]

        # Check that selected regression model is allowed
        if reg_model not in ["lin", "gbm", "etr", "gam"]:
            raise ValueError(
                "reg_model has to either be lin (Linear regression, default), gbm (Gradient boosting model), etr (Extra trees regressor) or gam (Generalized additive model)"
            )
        self.reg_model = reg_model
        self.ml_setup_kwargs = ml_setup_kwargs

        # Monthly data can only use robust linear regression because of limited number of data
        if (time_resolution == "M") & (reg_model != "lin"):
            raise ValueError("For monthly time resolution, only linear regression is allowed!")

        # Run preprocessing step
        self.calculate_aggregate_dataframe()

        # Store start and end of period of record
        self._start_por = self._aggregate.index.min()
        self._end_por = self._aggregate.index.max()

        # Create a data frame to store monthly/daily reanalysis data over plant period of record
        self._reanalysis_por = self._aggregate.loc[
            (self._aggregate.index >= self._start_por) & (self._aggregate.index <= self._end_por)
        ]

    @logged_method_call
    def run(self, num_sim, reanal_subset=None):
        """
        Perform pre-processing of data into an internal representation for which the analysis can run more quickly.

        Args:
            reanal_subset(:obj:`list`): list of str data indicating which reanalysis products to use in OA
            num_sim(:obj:`int`): number of simulations to perform

        Returns:
            None
        """
        self.num_sim = num_sim

        if reanal_subset is None:
            self.reanal_subset = self._reanal_products
        else:
            self.reanal_subset = reanal_subset

        # Write parameters of run to the log file
        logged_self_params = [
            "uncertainty_meter",
            "uncertainty_losses",
            "uncertainty_loss_max",
            "uncertainty_windiness",
            "uncertainty_nan_energy",
            "num_sim",
            "reanal_subset",
        ]
        logged_params = {name: getattr(self, name) for name in logged_self_params}
        logger.info("Running with parameters: {}".format(logged_params))

        # Start the computation
        self.calculate_long_term_losses()

        self.setup_monte_carlo_inputs()

        self.results = self.run_AEP_monte_carlo()

        # Log the completion of the run
        logger.info("Run completed")

    @logged_method_call
    def groupby_time_res(self, df):
        """
        Group pandas dataframe based on the time resolution chosen in the calculation.

        Args:
            df(:obj:`dataframe`): dataframe that needs to be grouped based on time resolution used

        Returns:
            None
        """

        if self.time_resolution == "M":
            df_grouped = df.groupby(df.index.month).mean()
        elif self.time_resolution == "D":
            df_grouped = df.groupby([(df.index.month), (df.index.day)]).mean()
        elif self.time_resolution == "H":
            df_grouped = df.groupby([(df.index.month), (df.index.day), (df.index.hour)]).mean()

        return df_grouped

    @logged_method_call
    def calculate_aggregate_dataframe(self):
        """
        Perform pre-processing of the plant data to produce a monthly/daily data frame to be used in AEP analysis.

        Args:
            (None)

        Returns:
            (None)
        """

        # Average to monthly/daily, quantify NaN data
        self.process_revenue_meter_energy()

        # Average to monthly/daily, quantify NaN data, merge with revenue meter energy data
        self.process_loss_estimates()

        # Density correct wind speeds, process temperature and wind direction, average to monthly/daily
        self.process_reanalysis_data()

        # Remove first and last reporting months if only partial month reported
        # (only for monthly time resolution calculations)
        if self.time_resolution == "M":
            self.trim_monthly_df()

        # Drop any data that have NaN gross energy values or NaN reanalysis data
        self._aggregate = self._aggregate.dropna(
            subset=["gross_energy_gwh"] + [product for product in self._reanal_products]
        )

    @logged_method_call
    def process_revenue_meter_energy(self):
        """
        Initial creation of monthly data frame:
            1. Populate monthly/daily data frame with energy data summed from 10-min QC'd data
            2. For each monthly/daily value, find percentage of NaN data used in creating it and flag if percentage is
               greater than 0

        Args:
            (None)
        Returns:
            (None)
        """
        df = self._plant.meter  # Get the meter data frame

        # Create the monthly/daily data frame by summing meter energy
        self._aggregate = (
            df.resample(self._resample_freq)["energy"].sum() / 1e6
        ).to_frame()  # Get monthly energy values in GWh
        self._aggregate.rename(columns={"energy": "energy_gwh"}, inplace=True)  # Rename kWh to MWh

        # Determine how much 10-min data was missing for each year-month/daily energy value. Flag accordigly if any is missing
        self._aggregate["energy_nan_perc"] = df.resample(self._resample_freq)["energy"].apply(
            tm.percent_nan
        )  # Get percentage of meter data that were NaN when summing to monthly/daily

        if self.time_resolution == "M":
            # Create a column with expected number of days per month (to be used when normalizing to 30-days for regression)
            days_per_month = (pd.Series(self._aggregate.index)).dt.daysinmonth
            days_per_month.index = self._aggregate.index
            self._aggregate["num_days_expected"] = days_per_month

            # Get actual number of days per month in the raw data
            # (used when trimming beginning and end of monthly data frame)
            # If meter data has higher resolution than monthly
            if (self._plant.metadata.meter.frequency == "1MS") | (
                self._plant.metadata.meter.frequency == "1M"
            ):
                self._aggregate["num_days_actual"] = self._aggregate["num_days_expected"]
            else:
                self._aggregate["num_days_actual"] = df.resample("MS")["energy"].apply(tm.num_days)

    @logged_method_call
    def process_loss_estimates(self):
        """
        Append availability and curtailment losses to monthly data frame

        Args:
            (None)

        Returns:
            (None)
        """
        df = self._plant.curtail

        curt_aggregate = np.divide(
            df.resample(self._resample_freq)[["availability", "curtailment"]].sum(), 1e6
        )  # Get sum of avail and curt losses in GWh

        curt_aggregate.rename(
            columns={"availability": "availability_gwh", "curtailment": "curtailment_gwh"},
            inplace=True,
        )
        # Merge with revenue meter monthly/daily data
        self._aggregate = self._aggregate.join(curt_aggregate)

        # Add gross energy field
        self._aggregate["gross_energy_gwh"] = un.compute_gross_energy(
            self._aggregate["energy_gwh"],
            self._aggregate["availability_gwh"],
            self._aggregate["curtailment_gwh"],
            "energy",
            "energy",
        )

        # Calculate percentage-based losses
        self._aggregate["availability_pct"] = np.divide(
            self._aggregate["availability_gwh"], self._aggregate["gross_energy_gwh"]
        )
        self._aggregate["curtailment_pct"] = np.divide(
            self._aggregate["curtailment_gwh"], self._aggregate["gross_energy_gwh"]
        )

        self._aggregate["avail_nan_perc"] = df.resample(self._resample_freq)["availability"].apply(
            tm.percent_nan
        )  # Get percentage of 10-min meter data that were NaN when summing to monthly/daily
        self._aggregate["curt_nan_perc"] = df.resample(self._resample_freq)["curtailment"].apply(
            tm.percent_nan
        )  # Get percentage of 10-min meter data that were NaN when summing to monthly/daily

        self._aggregate["nan_flag"] = False  # Set flag to false by default
        self._aggregate.loc[
            (self._aggregate["energy_nan_perc"] > self.uncertainty_nan_energy)
            | (self._aggregate["avail_nan_perc"] > self.uncertainty_nan_energy)
            | (self._aggregate["curt_nan_perc"] > self.uncertainty_nan_energy),
            "nan_flag",
        ] = True  # If more than 1% of data are NaN, set flag to True

        # By default, assume all reported losses are representative of long-term operational
        self._aggregate["availability_typical"] = True
        self._aggregate["curtailment_typical"] = True

        # By default, assume combined availability and curtailment losses are below the threshold to be considered valid
        self._aggregate["combined_loss_valid"] = True

    @logged_method_call
    def process_reanalysis_data(self):
        """
        Process reanalysis data for use in PRUF plant analysis:
            - calculate density-corrected wind speed and wind components
            - get monthly/daily average wind speeds and components
            - calculate monthly/daily average wind direction
            - calculate monthly/daily average temperature
            - append monthly/daily averages to monthly/daily energy data frame

        Args:
            (None)

        Returns:
            (None)
        """

        # Identify start and end dates for long-term correction
        # First find date range common to all reanalysis products and drop minute field of start date
        start_date = max(
            [self._plant.reanalysis[key].index.min() for key in self._reanal_products]
        ).replace(minute=0)
        end_date = min([self._plant.reanalysis[key].index.max() for key in self._reanal_products])

        # Next, update the start date to make sure it corresponds to a full time period
        start_date_minus = start_date - pd.DateOffset(hours=1)
        if (self.time_resolution == "M") & (start_date.month == start_date_minus.month):
            # If not at the beginning of a month, use the beginning of the next month as the start date
            start_date = start_date.replace(day=1, hour=0, minute=0) + pd.DateOffset(months=1)
        elif (self.time_resolution == "D") & (start_date.day == start_date_minus.day):
            # If not at the beginning of a day, use the beginning of the next day as the start date
            start_date = start_date.replace(hour=0, minute=0) + pd.DateOffset(days=1)

        # Now determine the end date based on either the user-defined end date or the end of the last full month
        if self.end_date_lt is not None:
            # If valid (before the last full time period in the reanalysis data), use the specified end date
            end_date_lt_plus = self.end_date_lt + pd.DateOffset(hours=1)
            if (self.time_resolution == "M") & (self.end_date_lt.month == end_date_lt_plus.month):
                # If not at the end of a month, use the end of the month as the new end date
                self.end_date_lt = (
                    self.end_date_lt.replace(day=1, hour=0, minute=0)
                    + pd.DateOffset(months=1)
                    - pd.DateOffset(hours=1)
                )
            elif (self.time_resolution == "D") & (self.end_date_lt.day == end_date_lt_plus.day):
                # If not at the end of a day, use the end of the day as the new end date
                self.end_date_lt = self.end_date_lt.replace(hour=23, minute=0)

            if self.end_date_lt > end_date:
                raise ValueError(
                    "Invalid end date for long-term correction. The end date cannot exceed the last full time period (defined by the time resolution) in the provided reanalysis data."
                )
            else:
                # replace end date
                end_date = self.end_date_lt
        else:
            # If not at the end of a month, use the end of the previous month as the end date
            if end_date.month == (end_date + pd.DateOffset(hours=1)).month:
                end_date = end_date.replace(day=1, hour=0, minute=0) - pd.DateOffset(hours=1)

        # Define empty data frame that spans our period of interest
        self._reanalysis_aggregate = pd.DataFrame(
            index=pd.date_range(
                start=start_date, end=end_date, freq=self._resample_freq
            ),  # tz="UTC"),
            dtype=float,
        )

        # Check if the date range covers the maximum number of years needed for the windiness correction
        start_date_required = (
            self._reanalysis_aggregate.index[-1]
            + self._reanalysis_aggregate.index.freq
            - pd.offsets.DateOffset(years=self.uncertainty_windiness[1])
        )
        if self._reanalysis_aggregate.index[0] > start_date_required:
            if self.end_date_lt is not None:
                raise ValueError(
                    "Invalid end date argument for long-term correction. This end date does not provide enough reanalysis data for the long-term correction."
                )
            else:
                raise ValueError(
                    "The date range of the provided reanalysis data is not long enough to perform the long-term correction."
                )

        # Now loop through the different reanalysis products, density-correct wind speeds, and take monthly averages
        for key in self._reanal_products:
            rean_df = self._plant.reanalysis[key]
            # rean_df = rean_df.rename(self._plant.metadata[key].col_map)
            rean_df["ws_dens_corr"] = mt.air_density_adjusted_wind_speed(
                rean_df["windspeed"], rean_df["density"]
            )  # Density correct wind speeds
            self._reanalysis_aggregate[key] = rean_df.resample(self._resample_freq)[
                "ws_dens_corr"
            ].mean()  # .to_frame() # Get average wind speed by year-month

            if self.reg_winddirection | self.reg_temperature:
                namescol = [key + "_" + var for var in self._rean_vars]
                self._reanalysis_aggregate[namescol] = (
                    rean_df[self._rean_vars].resample(self._resample_freq).mean()
                )

            if self.reg_winddirection:  # if wind direction is considered as regression variable
                self._reanalysis_aggregate[key + "_winddirection"] = np.rad2deg(
                    np.pi
                    - (
                        np.arctan2(
                            -self._reanalysis_aggregate[key + "_windspeed_u"],
                            self._reanalysis_aggregate[key + "_windspeed_v"],
                        )
                    )
                )  # Calculate wind direction

        # TODO JP: Had to localize the timezone after V3 update. Is there a better way to do this?
        # self._aggregate.index = self._aggregate.index.tz_localize("UTC")
        self._aggregate = self._aggregate.join(
            self._reanalysis_aggregate
        )  # Merge monthly reanalysis data to monthly energy data frame

    @logged_method_call
    def trim_monthly_df(self):
        """
        Remove first and/or last month of data if the raw data had an incomplete number of days

        Args:
            (None)

        Returns:
            (None)
        """
        for p in self._aggregate.index[[0, -1]]:  # Loop through 1st and last data entry
            if (
                self._aggregate.loc[p, "num_days_expected"]
                != self._aggregate.loc[p, "num_days_actual"]
            ):
                self._aggregate.drop(p, inplace=True)  # Drop the row from data frame

    @logged_method_call
    def calculate_long_term_losses(self):
        """
        This function calculates long-term availability and curtailment losses based on the reported data grouped by the time resolution,
        filtering for those data that are deemed representative of average plant performance.

        Args:
            (None)

        Returns:
            (None)
        """
        df = self._aggregate

        # isolate availabilty and curtailment values that are representative of average plant performance
        avail_valid = df.loc[df["availability_typical"], "availability_pct"].to_frame()
        curt_valid = df.loc[df["curtailment_typical"], "curtailment_pct"].to_frame()

        # Now get average percentage losses by month or day
        avail_long_term = self.groupby_time_res(avail_valid)["availability_pct"]
        curt_long_term = self.groupby_time_res(curt_valid)["curtailment_pct"]

        # Ensure there are 12 or 365 data points in long-term average. If not, throw an exception:
        if avail_long_term.shape[0] < self._calendar_samples:
            raise Exception(
                "Not all calendar days/months represented in long-term availability calculation"
            )
        if curt_long_term.shape[0] < self._calendar_samples:
            raise Exception(
                "Not all calendar days/months represented in long-term curtailment calculation"
            )

        self.long_term_losses = (avail_long_term, curt_long_term)

    def setup_monte_carlo_inputs(self):
        """
        Create and populate the data frame defining the simulation parameters.
        This data frame is stored as self._inputs

        Args:
            (None)

        Returns:
            (None)
        """

        # Create extra long list of renanalysis product names to sample from
        reanal_list = list(np.repeat(self.reanal_subset, self.num_sim))

        inputs = {
            "reanalysis_product": np.asarray(random.sample(reanal_list, self.num_sim)),
            "metered_energy_fraction": np.random.normal(1, self.uncertainty_meter, self.num_sim),
            "loss_fraction": np.random.normal(1, self.uncertainty_losses, self.num_sim),
            "num_years_windiness": np.random.randint(
                self.uncertainty_windiness[0], self.uncertainty_windiness[1] + 1, self.num_sim
            ),
            "loss_threshold": np.random.randint(
                self.uncertainty_loss_max[0], self.uncertainty_loss_max[1] + 1, self.num_sim
            )
            / 100.0,
        }
        if self.outlier_detection:
            inputs["outlier_threshold"] = (
                np.random.randint(
                    self.uncertainty_outlier[0] * 10,
                    (self.uncertainty_outlier[1] + 0.1) * 10,
                    self.num_sim,
                )
                / 10.0
            )

        self._inputs = pd.DataFrame(inputs)

    @logged_method_call
    def filter_outliers(self, n):
        """
        This function filters outliers based on a combination of range filter, unresponsive sensor filter,
        and window filter.
        We use a memoized funciton to store the regression data in a dictionary for each combination as it
        comes up in the Monte Carlo simulation. This saves significant computational time in not having to run
        robust linear regression for each Monte Carlo iteration

        Args:
            n(:obj:`float`): Monte Carlo iteration

        Returns:
            :obj:`pandas.DataFrame`: Filtered monthly/daily data ready for linear regression
        """

        reanal = self._run.reanalysis_product

        # Check if valid data has already been calculated and stored. If so, just return it
        if (reanal, self._run.loss_threshold) in self.outlier_filtering:
            valid_data = self.outlier_filtering[(reanal, self._run.loss_threshold)]
            return valid_data

        # If valid data hasn't yet been stored in dictionary, determine the valid data
        df = self._aggregate

        # First set of filters checking combined losses and if the Nan data flag was on
        df_sub = df.loc[
            ((df["availability_pct"] + df["curtailment_pct"]) < self._run.loss_threshold)
            & (~df["nan_flag"]),
            :,
        ]

        # Set maximum range for using bin-filter, convert from MW to GWh
        plant_capac = self._plant.metadata.capacity / 1000.0 * self._hours_in_res

        # Apply range filter to wind speed
        df_sub = df_sub.assign(flag_range=filters.range_flag(df_sub[reanal], lower=0, upper=40))
        if self.reg_temperature:
            # Apply range filter to temperatre
            df_sub = df_sub.assign(
                flag_range_T=filters.range_flag(
                    df_sub[reanal + "_temperature"], lower=200, upper=320
                )  # Temperature is in Kelvin
            )
        # Apply window range filter
        df_sub.loc[:, "flag_window"] = filters.window_range_flag(
            window_col=df_sub[reanal],
            window_start=5.0,
            window_end=40,
            value_col=df_sub["energy_gwh"],
            value_min=0.02 * plant_capac,
            value_max=1.2 * plant_capac,
        )

        if self.outlier_detection:
            if self.time_resolution == "M":
                # Monthly linear regression (i.e., few data points):
                # filter outliers based on robust linear regression
                # using Huber algorithm to flag outliers
                X = sm.add_constant(df_sub[reanal])  # Reanalysis data with constant column
                y = (
                    df_sub["gross_energy_gwh"] * 30 / df_sub["num_days_expected"]
                )  # Energy data (normalized to 30-days)

                # Perform robust linear regression
                rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT(self._run.outlier_threshold))
                rlm_results = rlm.fit()

                # Define valid data as points in which the Huber algorithm returned a value of 1
                df_sub.loc[:, "flag_outliers"] = rlm_results.weights != 1

            else:
                # Daily regressions (i.e., higher number of data points):
                # Apply bin filter to catch outliers
                df_sub.loc[:, "flag_outliers"] = filters.bin_filter(
                    data=df_sub,
                    bin_col="gross_energy_gwh",
                    value_col=reanal,
                    bin_width=0.06 * plant_capac,
                    threshold=self._run.outlier_threshold,  # wind bin threshold (multiplicative factor of std of <value_col> in bin)
                    center_type="median",
                    bin_min=0.01 * plant_capac,
                    bin_max=0.85 * plant_capac,
                    threshold_type="std",
                    direction="all",  # both left and right (from the median)
                )
        else:
            df_sub.loc[:, "flag_outliers"] = False

        # Create a 'final' flag which is true if any of the previous flags are true
        df_sub.loc[:, "flag_final"] = (
            (df_sub.loc[:, "flag_range"])
            | (df_sub.loc[:, "flag_window"])
            | (df_sub.loc[:, "flag_outliers"])
        )
        if self.reg_temperature:
            df_sub.loc[:, "flag_final"] = (df_sub.loc[:, "flag_final"]) | (
                df_sub.loc[:, "flag_range_T"]
            )

        # Define valid data
        valid_data = df_sub.loc[
            ~df_sub.loc[:, "flag_final"],
            [reanal, "energy_gwh", "availability_gwh", "curtailment_gwh"],
        ]
        if self.reg_winddirection:
            valid_data_to_add = df_sub.loc[
                ~df_sub.loc[:, "flag_final"],
                [reanal + "_winddirection", reanal + "_windspeed_u", reanal + "_windspeed_v"],
            ]
            valid_data = pd.concat([valid_data, valid_data_to_add], axis=1)

        if self.reg_temperature:
            valid_data_to_add = df_sub.loc[~df_sub.loc[:, "flag_final"], [reanal + "_temperature"]]
            valid_data = pd.concat([valid_data, valid_data_to_add], axis=1)

        if self.time_resolution == "M":
            valid_data_to_add = df_sub.loc[~df_sub.loc[:, "flag_final"], ["num_days_expected"]]
            valid_data = pd.concat([valid_data, valid_data_to_add], axis=1)

        # Update the dictionary
        self.outlier_filtering[(reanal, self._run.loss_threshold)] = valid_data

        # Return result
        return valid_data

    @logged_method_call
    def set_regression_data(self, n):
        """
        This will be called for each iteration of the Monte Carlo simulation and will do the following:

            1. Randomly sample monthly/daily revenue meter, availabilty, and curtailment data based on specified uncertainties
               and correlations
            2. Randomly choose one reanalysis product
            3. Calculate gross energy from randomzied energy data
            4. Normalize gross energy to 30-day months
            5. Filter results to remove months/days with NaN data and with combined losses that exceed the Monte Carlo
               sampled max threhold
            6. Return the wind speed and normalized gross energy to be used in the regression relationship

        Args:
            n(:obj:`int`): The Monte Carlo iteration number

        Returns:
            :obj:`pandas.Series`: Monte-Carlo sampled wind speeds and other variables (temperature, wind direction) if used in the regression
            :obj:`pandas.Series`: Monte-Carlo sampled normalized gross energy
        """
        # Get data to use in regression based on filtering result
        reg_data = self.filter_outliers(n)

        # Now monte carlo sample the data
        mc_energy = (
            reg_data["energy_gwh"] * self._run.metered_energy_fraction
        )  # Create new Monte-Carlo sampled data frame and sample energy data
        mc_availability = (
            reg_data["availability_gwh"] * self._run.loss_fraction
        )  # Calculate MC-generated availability
        mc_curtailment = (
            reg_data["curtailment_gwh"] * self._run.loss_fraction
        )  # Calculate MC-generated curtailment

        # Calculate gorss energy and normalize to 30-days
        mc_gross_energy = mc_energy + mc_availability + mc_curtailment
        if self.time_resolution == "M":
            num_days_expected = reg_data["num_days_expected"]
            mc_gross_norm = (
                mc_gross_energy * 30 / num_days_expected
            )  # Normalize gross energy to 30-day months
        else:
            mc_gross_norm = mc_gross_energy

        # Set reanalysis product
        reg_inputs = reg_data[
            self._run.reanalysis_product
        ]  # Copy wind speed data to Monte Carlo data frame

        if self.reg_temperature:  # if temperature is considered as regression variable
            mc_temperature = reg_data[
                self._run.reanalysis_product + "_temperature"
            ]  # Copy temperature data to Monte Carlo data frame
            reg_inputs = pd.concat([reg_inputs, mc_temperature], axis=1)

        if self.reg_winddirection:  # if wind direction is considered as regression variable
            mc_wind_direction = reg_data[
                self._run.reanalysis_product + "_winddirection"
            ]  # Copy wind direction data to Monte Carlo data frame
            reg_inputs = pd.concat([reg_inputs, np.sin(np.deg2rad(mc_wind_direction))], axis=1)
            reg_inputs = pd.concat([reg_inputs, np.cos(np.deg2rad(mc_wind_direction))], axis=1)

        reg_inputs = pd.concat([reg_inputs, mc_gross_norm], axis=1)
        # Return values needed for regression
        return reg_inputs  # Return randomly sampled wind speed, wind direction, temperature and normalized gross energy

    @logged_method_call
    def run_regression(self, n):
        """
        Run robust linear regression between Monte-Carlo generated monthly/daily gross energy,
        wind speed, temperature and wind direction (if used)

        Args:
            n(:obj:`int`): The Monte Carlo iteration number

        Returns:
            :obj:`?`: trained regression model
        """
        reg_data = self.set_regression_data(n)  # Get regression data

        # Bootstrap input data to incorporate some regression uncertainty
        reg_data = np.array(reg_data.sample(frac=1.0, replace=True))

        # Update Monte Carlo tracker fields
        self._mc_num_points[n] = np.shape(reg_data)[0]

        # Run regression. Note, the last column of reg_data is the target variable for the regression
        # Linear regression
        if self.reg_model == "lin":
            reg = LinearRegression().fit(np.array(reg_data[:, 0:-1]), reg_data[:, -1])
            predicted_y = reg.predict(np.array(reg_data[:, 0:-1]))

            self._mc_slope[n, :] = reg.coef_
            self._mc_intercept[n] = np.float64(reg.intercept_)

            self._r2_score[n] = r2_score(reg_data[:, -1], predicted_y)
            self._mse_score[n] = mean_squared_error(reg_data[:, -1], predicted_y)
            return reg
        # Machine learning models
        else:
            ml = MachineLearningSetup(algorithm=self.reg_model, **self.ml_setup_kwargs)
            # Memoized approach for optimized hyperparameters
            if self._run.reanalysis_product in self.opt_model:
                self.opt_model[(self._run.reanalysis_product)].fit(
                    np.array(reg_data[:, 0:-1]), reg_data[:, -1]
                )
            else:  # optimize hyperparameters once for each reanalysis product
                ml.hyper_optimize(
                    np.array(reg_data[:, 0:-1]),
                    reg_data[:, -1],
                    n_iter_search=20,
                    report=False,
                    cv=KFold(n_splits=5),
                )
                # Store optimized hyperparameters for each reanalysis product
                self.opt_model[(self._run.reanalysis_product)] = ml.opt_model

            predicted_y = self.opt_model[(self._run.reanalysis_product)].predict(
                np.array(reg_data[:, 0:-1])
            )

            self._r2_score[n] = r2_score(reg_data[:, -1], predicted_y)
            self._mse_score[n] = mean_squared_error(reg_data[:, -1], predicted_y)
            return self.opt_model[(self._run.reanalysis_product)]

    @logged_method_call
    def run_AEP_monte_carlo(self):
        """
        Loop through OA process a number of times and return array of AEP results each time

        Returns:
            :obj:`numpy.ndarray` Array of AEP, long-term avail, long-term curtailment calculations
        """

        num_sim = self.num_sim

        # Initialize arrays to store metrics and results
        self._mc_num_points = np.empty(num_sim, dtype=np.float64)
        self._r2_score = np.empty(num_sim, dtype=np.float64)
        self._mse_score = np.empty(num_sim, dtype=np.float64)

        num_vars = 1
        if self.reg_winddirection:
            num_vars = num_vars + 2
        if self.reg_temperature:
            num_vars = num_vars + 1

        if self.reg_model == "lin":
            self._mc_intercept = np.empty(num_sim, dtype=np.float64)
            self._mc_slope = np.empty([num_sim, num_vars], dtype=np.float64)

        aep_GWh = np.empty(num_sim)
        avail_pct = np.empty(num_sim)
        curt_pct = np.empty(num_sim)
        lt_por_ratio = np.empty(num_sim)
        iav = np.empty(num_sim)

        # Loop through number of simulations, run regression each time, store AEP results
        for n in tqdm(np.arange(num_sim)):

            self._run = self._inputs.loc[n]

            # Run regression
            fitted_model = self.run_regression(n)

            # Get long-term regression inputs
            reg_inputs_lt = self.sample_long_term_reanalysis()

            # Get long-term normalized gross energy by applying regression result to long-term monthly wind speeds
            inputs = np.array(reg_inputs_lt)
            if num_vars == 1:
                inputs = inputs.reshape(-1, 1)
            gross_lt = fitted_model.predict(inputs)

            # Get POR gross energy by applying regression result to POR regression inputs
            reg_inputs_por = [self._reanalysis_por[self._run.reanalysis_product]]
            if self.reg_temperature:
                reg_inputs_por += [
                    self._reanalysis_por[self._run.reanalysis_product + "_temperature"]
                ]
            if self.reg_winddirection:
                reg_inputs_por += [
                    np.sin(
                        np.deg2rad(
                            self._reanalysis_por[self._run.reanalysis_product + "_winddirection"]
                        )
                    )
                ]
                reg_inputs_por += [
                    np.cos(
                        np.deg2rad(
                            self._reanalysis_por[self._run.reanalysis_product + "_winddirection"]
                        )
                    )
                ]
            gross_por = fitted_model.predict(np.array(pd.concat(reg_inputs_por, axis=1)))

            # Create padans dataframe for gross_por and group by calendar date to have a single full year
            gross_por = self.groupby_time_res(
                pd.DataFrame(
                    data=gross_por, index=self._reanalysis_por[self._run.reanalysis_product].index
                )
            )

            if self.time_resolution == "M":  # Undo normalization to 30-day months
                # Shift the list of number of days per month to align with the reanalysis data
                last_month = self._reanalysis_aggregate.index[-1].month
                gross_lt = (
                    gross_lt
                    * np.tile(
                        np.roll(self.num_days_lt, 12 - last_month), self._run.num_years_windiness
                    )
                    / 30
                )
                gross_por = np.array(gross_por).flatten() * self.num_days_lt / 30

            # Annual values of lt gross energy, needed for IAV
            reg_inputs_lt["gross_lt"] = gross_lt

            # Annual resample starting on the first day in reg_inputs_lt
            gross_lt_annual = get_annual_values(reg_inputs_lt["gross_lt"])

            # Get long-term availability and curtailment losses, using gross_lt to weight individual monthly losses
            [avail_lt_losses, curt_lt_losses] = self.sample_long_term_losses(
                reg_inputs_lt["gross_lt"]
            )

            # Assign AEP, IAV, long-term availability, and long-term curtailment to output data frame
            aep_GWh[n] = gross_lt.sum() / self._run.num_years_windiness * (1 - avail_lt_losses)
            iav[n] = gross_lt_annual.std() / gross_lt_annual.mean()
            avail_pct[n] = avail_lt_losses
            curt_pct[n] = curt_lt_losses
            lt_por_ratio[n] = (gross_lt.sum() / self._run.num_years_windiness) / gross_por.sum()

        # Calculate mean IAV for gross energy
        iav_avg = iav.mean()

        # Apply IAV to AEP from single MC iterations
        iav_nsim = np.random.normal(1, iav_avg, self.num_sim)
        aep_GWh = aep_GWh * iav_nsim
        lt_por_ratio = lt_por_ratio * iav_nsim

        # Return final output
        sim_results = pd.DataFrame(
            index=np.arange(num_sim),
            data={
                "aep_GWh": aep_GWh,
                "avail_pct": avail_pct,
                "curt_pct": curt_pct,
                "lt_por_ratio": lt_por_ratio,
                "r2": self._r2_score,
                "mse": self._mse_score,
                "n_points": self._mc_num_points,
                "iav": iav,
            },
        )
        return sim_results

    @logged_method_call
    def sample_long_term_reanalysis(self):
        """
        This function returns the long-term monthly/daily wind speeds based on the Monte-Carlo generated sample of:

            1. The reanalysis product
            2. The number of years to use in the long-term correction

        Args:
           (None)
        Returns:
           :obj:`pandas.DataFrame`: the windiness-corrected or 'long-term' monthly/daily wind speeds
        """
        # Check if valid data has already been calculated and stored. If so, just return it
        if (self._run.reanalysis_product, self._run.num_years_windiness) in self.long_term_sampling:
            long_term_reg_inputs = self.long_term_sampling[
                (self._run.reanalysis_product, self._run.num_years_windiness)
            ]
            return long_term_reg_inputs.copy()

        # Sample long-term wind speed values
        ws_df = (
            self._reanalysis_aggregate[self._run.reanalysis_product].to_frame().dropna()
        )  # Drop NA values from monthly/daily reanalysis data series
        long_term_reg_inputs = ws_df[
            ws_df.index[-1]
            + ws_df.index.freq
            - pd.offsets.DateOffset(years=self._run.num_years_windiness) :
        ]  # Get last 'x' years of data from reanalysis product

        # Temperature and wind direction
        namescol = [self._run.reanalysis_product + "_" + var for var in self._rean_vars]
        long_term_temp = self._reanalysis_aggregate[namescol].dropna()[
            ws_df.index[-1]
            + ws_df.index.freq
            - pd.offsets.DateOffset(years=self._run.num_years_windiness) :
        ]
        if self.reg_temperature:
            long_term_reg_inputs = pd.concat(
                [
                    long_term_reg_inputs,
                    long_term_temp[self._run.reanalysis_product + "_temperature"],
                ],
                axis=1,
            )
        if self.reg_winddirection:
            wd_aggregate = np.rad2deg(
                np.pi
                - np.arctan2(
                    -long_term_temp[self._run.reanalysis_product + "_windspeed_u"],
                    long_term_temp[self._run.reanalysis_product + "_windspeed_v"],
                )
            )  # Calculate wind direction
            long_term_reg_inputs = pd.concat(
                [
                    long_term_reg_inputs,
                    np.sin(np.deg2rad(wd_aggregate)),
                    np.cos(np.deg2rad(wd_aggregate)),
                ],
                axis=1,
            )

        # Store result in dictionary
        self.long_term_sampling[
            (self._run.reanalysis_product, self._run.num_years_windiness)
        ] = long_term_reg_inputs

        # Return result
        return long_term_reg_inputs.copy()

    @logged_method_call
    def sample_long_term_losses(self, gross_lt):
        """
        This function calculates long-term availability and curtailment losses based on the Monte Carlo sampled
        historical availability and curtailment data. To estimate long-term losses, average percentage monthly losses
        are weighted by monthly long-term gross energy.

        Args:
            gross_lt(:obj:`pandas.Series`): Time series of long-term gross energy

        Returns:
            :obj:`float`: long-term availability loss expressed as fraction
            :obj:`float`: long-term curtailment loss expressed as fraction
        """
        mc_avail = self.long_term_losses[0] * self._run.loss_fraction
        mc_curt = self.long_term_losses[1] * self._run.loss_fraction

        # Calculate annualized monthly average long-term gross energy
        # Rename axis to time to be consistent with mc_avail and mc_curt when combining variables
        gross_lt_avg = self.groupby_time_res(gross_lt.rename_axis("time"))

        # Estimate long-term losses by weighting monthly losses by long-term monthly gross energy
        mc_avail_lt = (gross_lt_avg * mc_avail).sum() / gross_lt_avg.sum()
        mc_curt_lt = (gross_lt_avg * mc_curt).sum() / gross_lt_avg.sum()

        # Return long-term availabilty and curtailment
        return mc_avail_lt, mc_curt_lt

    # Plotting Routines

    def plot_normalized_monthly_reanalysis_windspeed(
        self,
        xlim: tuple[datetime.datetime, datetime.datetime] = (None, None),
        ylim: tuple[float, float] = (None, None),
        return_fig: bool = False,
        figure_kwargs: dict = {},
        plot_kwargs: dict = {},
        legend_kwargs: dict = {},
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Make a plot of the normalized annual average wind speeds from reanalysis data to show general
        trends for each, and highlighting the period of record for the plant data.

        Args:
            aep (:obj:`openoa.analysis.MonteCarloAEP`): An initialized MonteCarloAEP object.
            xlim (:obj:`tuple[datetime.datetime, datetime.datetime]`, optional): A tuple of datetimes
                representing the x-axis plotting display limits. Defaults to (None, None).
            ylim (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display limits.
                Defaults to (None, None).
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to `plt.figure()`. Defaults to {}.
            plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                `ax.plot()`. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                `ax.legend()`. Defaults to {}.

        Returns:
            None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]: If `return_fig` is True, then
                the figure and axes objects are returned for further tinkering/saving.
        """
        return plot.plot_monthly_reanalysis_windspeed(
            data=self._plant.reanalysis,
            windspeed_col="ws_dens_corr",
            plant_por=(self._aggregate.index[0], self._aggregate.index[-1]),
            xlim=xlim,
            ylim=ylim,
            return_fig=return_fig,
            figure_kwargs=figure_kwargs,
            plot_kwargs=plot_kwargs,
            legend_kwargs=legend_kwargs,
        )

    def plot_reanalysis_gross_energy_data(
        self,
        outlier_threshold: int,
        xlim: tuple[float, float] = (None, None),
        ylim: tuple[float, float] = (None, None),
        return_fig: bool = False,
        figure_kwargs: dict = {},
        plot_kwargs: dict = {},
        legend_kwargs: dict = {},
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """
        Makes a plot of the gross energy vs wind speed for each reanalysis product, with outliers
        highlighted in a contrasting color and separate marker. For

        Args:
            reanalysis (:obj:`dict[str, pandas.DataFrame]`): `PlantData.reanalysis` dictionary of reanalysis
                `DataFrame`s.
            outlier_thres (:obj:`float`): outlier threshold (typical range of 1 to 4) which adjusts
                outlier sensitivity detection.
            xlim (:obj:`tuple[float, float]`, optional): A tuple of datetimes
                representing the x-axis plotting display limits. Defaults to (None, None).
            ylim (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display limits.
                Defaults to (None, None).
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to `plt.figure()`. Defaults to {}.
            plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                `ax.scatter()`. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                `ax.legend()`. Defaults to {}.

        Returns:
            None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]: If `return_fig` is True, then
                the figure and axes objects are returned for further tinkering/saving.
        """

        figure_kwargs.setdefault("figsize", (9, 9))
        figure_kwargs.setdefault("dpi", 200)
        fig = plt.figure(**figure_kwargs)
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(
            color=[
                "tab:blue",
                "tab:orange",
                "tab:green",
                "tab:red",
                "tab:brown",
                "tab:pink",
                "tab:gray",
                "tab:olive",
            ]
        )

        valid_aggregate = self._aggregate

        # Monthly case: apply robust linear regression for outliers detection
        if self.time_resolution == "M":
            for name, df in self._plant.reanalysis.items():
                x = sm.add_constant(valid_aggregate[name])
                y = valid_aggregate["gross_energy_gwh"] * 30 / valid_aggregate["num_days_expected"]
                rlm = sm.RLM(y, x, M=sm.robust.norms.HuberT(t=outlier_threshold))
                rlm_results = rlm.fit()
                ix_outlier = rlm_results.weights != 1
                r2 = np.corrcoef(x.loc[~ix_outlier, name], y[~ix_outlier])[0, 1]
                ax.scatter(
                    x.loc[~ix_outlier, name],
                    y[~ix_outlier],
                    marker=MarkerStyle(marker="o", fillstyle="none"),
                    label=f"Valid {name} Data (R2={r2:.3f})",
                    **plot_kwargs,
                )
                ax.scatter(
                    x.loc[ix_outlier, name],
                    y[ix_outlier],
                    marker="x",
                    label=f"{name} Outlier",
                    **plot_kwargs,
                )

            ax.set_ylabel("30-day normalized gross energy (GWh)")

        # Daily/hourly case: apply bin filter for outliers detection
        else:
            for name, df in self._plant.reanalysis.items():
                x = valid_aggregate[name]
                y = valid_aggregate["gross_energy_gwh"]
                plant_capac = self._plant.metadata.capacity / 1000.0 * self._hours_in_res

                # Apply bin filter
                flag = filters.bin_filter(
                    bin_col=y,
                    value_col=x,
                    bin_width=0.06 * plant_capac,
                    threshold=outlier_threshold,  # wind bin threshold (stdev outside the median)
                    center_type="median",
                    bin_min=0.01 * plant_capac,
                    bin_max=0.85 * plant_capac,
                    threshold_type="std",
                    direction="all",  # both left and right (from the median)
                )

                # Continue plotting
                ax.scatter(x.loc[flag], y[flag], "x", label=f"{name} Outlier", **plot_kwargs)
                ax.scatter(x.loc[~flag], y[~flag], ".", label=f"Valid {name} data", **plot_kwargs)

            if self.time_resolution == "D":
                ax.set_ylabel("Daily gross energy (GWh)")
            elif self.time_resolution == "H":
                ax.set_ylabel("Hourly gross energy (GWh)")

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend(**legend_kwargs)
        ax.set_xlabel("Wind speed (m/s)")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        fig.tight_layout()
        plt.show()

        if return_fig:
            return fig, ax

    def plot_aggregate_plant_data_timeseries(
        self,
        xlim: tuple[datetime.datetime, datetime.datetime] = (None, None),
        ylim_energy: tuple[float, float] = (None, None),
        ylim_loss: tuple[float, float] = (None, None),
        return_fig: bool = False,
        figure_kwargs: dict = {},
        plot_kwargs: dict = {},
        legend_kwargs: dict = {},
    ):
        """
        Plot timeseries of monthly/daily gross energy, availability and curtailment.

        Args:
            data(:obj:`pandas.DataFrame`): A pandas DataFrame containing energy production and losses.
            energy_col(:obj:`str`): The name of the column in :py:attr:`data` containing the energy production.
            loss_cols(:obj:`list[str]`): The name(s) of the column(s) in :py:attr:`data` containing the loss data.
            energy_label(:obj:`str`): The legend label and y-axis label for the energy plot.
            loss_labels(:obj:`list[str]`): The legend labels losses plot.
            xlim (:obj:`tuple[datetime.datetime, datetime.datetime]`, optional): A tuple of datetimes
                representing the x-axis plotting display limits. Defaults to None.
            ylim_energy (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display
                limits for the gross energy plot (top figure). Defaults to None.
            ylim_loss (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display
                limits for the loss plot (bottom figure). Defaults to (None, None).
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to `plt.figure()`. Defaults to {}.
            plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                `ax.scatter()`. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                `ax.legend()`. Defaults to {}.

        Returns:
            None | tuple[matplotlib.pyplot.Figure, tuple[matplotlib.pyplot.Axes, matplotlib.pyplot.Axes]]:
                If `return_fig` is True, then the figure and axes objects are returned for further
                tinkering/saving.
        """
        return plot.plot_plant_energy_losses_timeseries(
            data=self._aggregate,
            energy_col="gross_energy_gwh",
            loss_cols=["availability_pct", "curtailment_pct"],
            energy_label="Gross Energy (GWh/yr)",
            loss_labels=["Availability", "Curtailment"],
            xlim=xlim,
            ylim_energy=ylim_energy,
            ylim_loss=ylim_loss,
            return_fig=return_fig,
            figure_kwargs=figure_kwargs,
            plot_kwargs=plot_kwargs,
            legend_kwargs=legend_kwargs,
        )

    def plot_result_aep_distributions(
        self,
        xlim_aep: tuple[float, float] = (None, None),
        xlim_availability: tuple[float, float] = (None, None),
        xlim_curtail: tuple[float, float] = (None, None),
        ylim_aep: tuple[float, float] = (None, None),
        ylim_availability: tuple[float, float] = (None, None),
        ylim_curtail: tuple[float, float] = (None, None),
        return_fig: bool = False,
        figure_kwargs: dict = {},
        plot_kwargs: dict = {},
        annotate_kwargs: dict = {},
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """
        Plot a distribution of AEP values from the Monte-Carlo OA method

        Args:
            xlim_aep (:obj:`tuple[float, float]`, optional): A tuple of floats representing the x-axis plotting display
                limits for the AEP subplot. Defaults to (None, None).
            xlim_availability (:obj:`tuple[float, float]`, optional): A tuple of floats representing the x-axis plotting
                display limits for the availability subplot. Defaults to (None, None).
            xlim_curtail (:obj:`tuple[float, float]`, optional): A tuple of floats representing the
                x-axis plotting display limits for the curtailment subplot. Defaults to (None, None).
            ylim_aep (:obj:`tuple[float, float]`, optional): A tuple of floats representing the y-axis plotting display
                limits for the AEP subplot. Defaults to (None, None).
            ylim_availability (:obj:`tuple[float, float]`, optional): A tuple of floats representing the y-axis plotting
                display limits for the availability subplot. Defaults to (None, None).
            ylim_curtail (:obj:`tuple[float, float]`, optional): A tuple of floats representing the
                y-axis plotting display limits for the curtailment subplot. Defaults to (None, None).
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to `plt.figure()`. Defaults to {}.
            plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                `ax.hist()`. Defaults to {}.
            annotate_kwargs (:obj:`dict`, optional): Additional annotation keyword arguments that are
                passed to `ax.annotate()`. Defaults to {}.

        Returns:
            None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]: If `return_fig` is True, then
                the figure and axes objects are returned for further tinkering/saving.
        """
        plot_results = self.results.copy()
        plot_results[["avail_pct", "curt_pct"]] = plot_results[["avail_pct", "curt_pct"]] * 100
        return plot.plot_distributions(
            data=plot_results,
            which=["aep_GWh", "avail_pct", "curt_pct"],
            xlabels=["AEP (GWh/yr)", "Availability Loss (%)", "Curtailment Loss (%)"],
            xlim=(xlim_aep, xlim_availability, xlim_curtail),
            ylim=(ylim_aep, ylim_availability, ylim_curtail),
            return_fig=return_fig,
            figure_kwargs=figure_kwargs,
            plot_kwargs=plot_kwargs,
            annotate_kwargs=annotate_kwargs,
        )

    def plot_aep_boxplot(
        self,
        x: pd.Series,
        xlabel: str,
        ylim: tuple[float, float] = (None, None),
        with_points: bool = False,
        return_fig: bool = False,
        figure_kwargs: dict = {},
        plot_kwargs_box: dict = {},
        plot_kwargs_points: dict = {},
        legend_kwargs: dict = {},
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plot box plots of AEP results sliced by a specified Monte Carlo parameter

        Args:
            x(:obj:`pandas.Series`): The data that splits the results in y.
            xlabel(:obj:`str`): The x-axis label.
            ylim (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display limits.
                Defaults to None.
            with_points (:obj:`bool`, optional): Flag to plot the individual points like a seaborn `swarmplot`. Defaults to False.
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to `plt.figure()`. Defaults to {}.
            plot_kwargs_box (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                `ax.boxplot()`. Defahults to {}.
            plot_kwargs_points (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                `ax.boxplot()`. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                `ax.legend()`. Defaults to {}.

        Returns:
            None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes, dict]: If `return_fig` is
                True, then the figure object, axes object, and a dictionary of the boxplot objects are
                returned for further tinkering/saving.
        """
        return plot.plot_boxplot(
            x=x,
            xlabel=xlabel,
            y=self.results.aep_GWh,
            ylabel="AEP (GWh/yr)",
            ylim=ylim,
            with_points=with_points,
            return_fig=return_fig,
            figure_kwargs=figure_kwargs,
            plot_kwargs_box=plot_kwargs_box,
            plot_kwargs_points=plot_kwargs_points,
            legend_kwargs=legend_kwargs,
        )
