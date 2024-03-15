from __future__ import annotations

import random
import datetime
from copy import deepcopy

import attrs
import numpy as np
import pandas as pd
import numpy.typing as npt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
from attrs import field, define
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.markers import MarkerStyle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from openoa.plant import PlantData, convert_to_list
from openoa.utils import plot, filters
from openoa.utils import timeseries as tm
from openoa.utils import unit_conversion as un
from openoa.utils import met_data_processing as mt
from openoa.schema import FromDictMixin, ResetValuesMixin
from openoa.logging import logging, logged_method_call
from openoa.schema.metadata import convert_frequency
from openoa.utils.machine_learning_setup import MachineLearningSetup
from openoa.analysis._analysis_validators import validate_reanalysis_selections


logger = logging.getLogger(__name__)

NDArrayFloat = npt.NDArray[np.float64]


plot.set_styling()


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


# TODO: Split this into a more generic naming convention to have other AEP methods, such as QMC
# TODO: Create an analysis result class that could be used for better results aggregation
@define(auto_attribs=True)
class MonteCarloAEP(FromDictMixin, ResetValuesMixin):
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

    Args:
        plant(:obj:`PlantData`): PlantData object from which PlantAnalysis should draw data.
        reg_temperature(:obj:`bool`): Indicator to include temperature (True) or not (False) as a
            regression input. Defaults to False.
        reg_wind_direction(:obj:`bool`): Indicator to include wind direction (True) or not (False) as
            a regression input. Defaults to False.
        reanalysis_products(``list[str]``) : List of reanalysis products to use for Monte Carlo
            sampling. Defaults to None, which pulls all the products contained in
            :py:attr:`plant.reanalysis`.
        uncertainty_meter(:obj:`float`): Uncertainty on revenue meter data. Defaults to 0.005.
        uncertainty_losses(:obj:`float`): Uncertainty on long-term losses. Defaults to 0.05.
        uncertainty_windiness(:obj:`tuple[int, int]`): number of years to use for the windiness
            correction. Defaults to (10, 20).
        uncertainty_loss_max(:obj:`tuple[int, int]`): Threshold for the combined availabilty and
            curtailment monthly loss threshold. Defaults to (10, 20).
        outlier_detection(:obj:`bool`): whether to perform (True) or not (False - default) outlier
            detection filtering. Defaults to False.
        uncertainty_outlier(:obj:`tuple[float, float]`): Min and max thresholds (Monte-Carlo
            sampled) for the outlier detection filter. At monthly resolution, this is the tuning
            constant for Huber's t function for a robust linear regression. At daily/hourly
            resolution, this is the number of stdev of wind speed used as threshold for the bin
            filter. Defaults to (1, 3).
        uncertainty_nan_energy(:obj:`float`): Threshold to flag days/months based on NaNs. Defaults
            to 0.01.
        time_resolution(:obj:`string`): whether to perform the AEP calculation at monthly ("ME" or
            "MS"), daily ("D") or hourly ("h") time resolution. Defaults to "MS".
        end_date_lt(:obj:`string` or :obj:`pandas.Timestamp`): The last date to use for the
            long-term correction. Note that only the component of the date corresponding to the
            time_resolution argument is considered. If None, the end of the last complete month of
            reanalysis data will be used. Defaults to None.
        reg_model(:obj:`string`): Which model to use for the regression ("lin" for linear, "gam" for,
            general additive, "gbm" for gradient boosting, or "etr" for extra treees). At monthly
            time resolution only linear regression is allowed because of the reduced number of data
            points. Defaults to "lin".
        ml_setup_kwargs(:obj:`kwargs`): Keyword arguments to
            :py:class:`openoa.utils.machine_learning_setup.MachineLearningSetup` class. Defaults to {}.
    """

    plant: PlantData = field(converter=deepcopy, validator=attrs.validators.instance_of(PlantData))
    reg_temperature: bool = field(default=False, converter=bool)
    reg_wind_direction: bool = field(default=False, converter=bool)
    reanalysis_products: list[str] = field(
        default=None,
        converter=convert_to_list,
        validator=(
            attrs.validators.deep_iterable(
                iterable_validator=attrs.validators.instance_of(list),
                member_validator=attrs.validators.instance_of((str, type(None))),
            ),
            validate_reanalysis_selections,
        ),
    )
    uncertainty_meter: float = field(default=0.005, converter=float)
    uncertainty_losses: float = field(default=0.05, converter=float)
    uncertainty_windiness: NDArrayFloat = field(
        default=(10.0, 20.0),
        converter=np.array,
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(np.ndarray),
            member_validator=attrs.validators.instance_of(float),
        ),
    )
    uncertainty_loss_max: NDArrayFloat = field(
        default=(10.0, 20.0),
        converter=np.array,
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(np.ndarray),
            member_validator=attrs.validators.instance_of(float),
        ),
    )
    outlier_detection: bool = field(default=False, converter=bool)
    uncertainty_outlier: NDArrayFloat = field(
        default=(1.0, 3.0),
        converter=np.array,
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(np.ndarray),
            member_validator=attrs.validators.instance_of(float),
        ),
    )
    uncertainty_nan_energy: float = field(default=0.01, converter=float)
    time_resolution: str = field(
        default="MS",
        converter=convert_frequency,
        validator=attrs.validators.in_(("MS", "ME", "D", "h")),
    )
    end_date_lt: str | pd.Timestamp = field(default=None)
    reg_model: str = field(
        default="lin", converter=str, validator=attrs.validators.in_(("lin", "gbm", "etr", "gam"))
    )
    ml_setup_kwargs: dict = field(default={}, converter=dict)

    # Internally created attributes need to be given a type before usage
    resample_freq: str = field(init=False)
    resample_hours: int = field(init=False)
    calendar_samples: int = field(init=False)
    outlier_filtering: dict = field(factory=dict, init=False)
    long_term_sampling: dict = field(factory=dict, init=False)
    opt_model: dict = field(factory=dict, init=False)
    reanalysis_vars: list[str] = field(factory=list, init=False)
    aggregate: pd.DataFrame = field(init=False)
    start_por: pd.Timestamp = field(init=False)
    end_por: pd.Timestamp = field(init=False)
    reanalysis_por: pd.DataFrame = field(init=False)
    num_days_lt: tuple = field(
        default=(31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
        init=False,
    )
    _reanalysis_aggregate: pd.DataFrame = field(init=False)
    num_sim: int = field(init=False)
    long_term_losses: tuple[pd.Series, pd.Series] = field(init=False)
    mc_inputs: pd.DataFrame = field(init=False)
    _mc_num_points: NDArrayFloat = field(init=False)
    _r2_score: NDArrayFloat = field(init=False)
    _mse_score: NDArrayFloat = field(init=False)
    _mc_intercept: NDArrayFloat = field(init=False)
    _mc_slope: NDArrayFloat = field(init=False)
    _run: pd.DataFrame = field(init=False)
    results: pd.DataFrame = field(init=False)
    run_parameters: list[str] = field(
        init=False,
        default=[
            "num_sim",
            "reg_model",
            "reanalysis_products",
            "uncertainty_meter",
            "uncertainty_losses",
            "uncertainty_windiness",
            "uncertainty_loss_max",
            "outlier_detection",
            "uncertainty_outlier",
            "uncertainty_nan_energy",
            "time_resolution",
            "end_date_lt",
            "ml_setup_kwargs",
        ],
    )

    @logged_method_call
    def __attrs_post_init__(self):
        """
        Initialize the Monte Carlo AEP analysis with data and parameters.
        """

        if self.reg_temperature and self.reg_wind_direction:
            analysis_type = "MonteCarloAEP-temp-wd"
            self.reanalysis_vars.extend(["WMETR_EnvTmp", "WMETR_HorWdSpdU", "WMETR_HorWdSpdV"])
        elif self.reg_temperature:
            analysis_type = "MonteCarloAEP-temp"
            self.reanalysis_vars.append("WMETR_EnvTmp")
        elif self.reg_wind_direction:
            analysis_type = "MonteCarloAEP-wd"
            self.reanalysis_vars.extend(["WMETR_HorWdSpdU", "WMETR_HorWdSpdV"])
        else:
            analysis_type = "MonteCarloAEP"

        if {analysis_type, "all"}.intersection(self.plant.analysis_type) == set():
            self.plant.analysis_type.append(analysis_type)

        # Ensure the data are up to spec before continuing with initialization
        self.plant.validate()

        logger.info("Initializing MonteCarloAEP Analysis Object")

        self.resample_freq = self.time_resolution
        self.resample_hours = {"MS": 30 * 24, "ME": 30 * 24, "D": 1 * 24, "h": 1}[
            self.time_resolution
        ]
        self.calendar_samples = {"MS": 12, "ME": 12, "D": 365, "h": 365 * 24}[self.time_resolution]

        if self.end_date_lt is not None:
            # Set to the bottom of the bottom of the hour
            self.end_date_lt = pd.to_datetime(self.end_date_lt).replace(minute=0)

        # Monthly data can only use robust linear regression because of limited number of data
        if (self.time_resolution in ("ME", "MS")) & (self.reg_model != "lin"):
            raise ValueError("For monthly time resolution, only linear regression is allowed!")

        # Run preprocessing step
        self.aggregate = pd.DataFrame()
        self.calculate_aggregate_dataframe()

        # Store start and end of period of record
        self.start_por = self.aggregate.index.min()
        self.end_por = self.aggregate.index.max()

        # Create a data frame to store monthly/daily reanalysis data over plant period of record
        self.reanalysis_por = self.aggregate.loc[
            (self.aggregate.index >= self.start_por) & (self.aggregate.index <= self.end_por)
        ]

    @logged_method_call
    def run(
        self,
        num_sim: int,
        reg_model: str = None,
        reanalysis_products: list[str] = None,
        uncertainty_meter: float = None,
        uncertainty_losses: float = None,
        uncertainty_windiness: float | tuple[float, float] = None,
        uncertainty_loss_max: float | tuple[float, float] = None,
        outlier_detection: bool = None,
        uncertainty_outlier: float | tuple[float, float] = None,
        uncertainty_nan_energy: float = None,
        time_resolution: str = None,
        end_date_lt: str | pd.Timestamp | None = None,
        ml_setup_kwargs: dict = None,
    ) -> None:
        """
        Process all appropriate data and run the MonteCarlo AEP analysis.

        .. note:: If None is provided to any of the inputs, then the last used input value will be
            used for the analysis, and if no prior values were set, then this is the model's defaults.

        Args:
            num_sim(:obj:`int`): number of simulations to perform
            reanal_products(obj:`list[str]`) : List of reanalysis products to use for Monte Carlo
                sampling. Defaults to None, which pulls all the products contained in
                :py:attr:`plant.reanalysis`.
            uncertainty_meter(:obj:`float`): Uncertainty on revenue meter data. Defaults to 0.005.
            uncertainty_losses(:obj:`float`): Uncertainty on long-term losses. Defaults to 0.05.
            uncertainty_windiness(:obj:`tuple[int, int]`): number of years to use for the windiness
                correction. Defaults to (10, 20).
            uncertainty_loss_max(:obj:`tuple[int, int]`): Threshold for the combined availabilty and
                curtailment monthly loss threshold. Defaults to (10, 20).
            outlier_detection(:obj:`bool`): whether to perform (True) or not (False - default) outlier
                detection filtering. Defaults to False.
            uncertainty_outlier(:obj:`tuple[float, float]`): Min and max thresholds (Monte-Carlo
                sampled) for the outlier detection filter. At monthly resolution, this is the tuning
                constant for Huber's t function for a robust linear regression. At daily/hourly
                resolution, this is the number of stdev of wind speed used as threshold for the bin
                filter. Defaults to (1, 3).
            uncertainty_nan_energy(:obj:`float`): Threshold to flag days/months based on NaNs. Defaults
                to 0.01.
            time_resolution(:obj:`string`): whether to perform the AEP calculation at monthly ("ME" or
                "MS"), daily ("D") or hourly ("h") time resolution. Defaults to "ME".
            end_date_lt(:obj:`string` or :obj:`pandas.Timestamp`): The last date to use for the
                long-term correction. Note that only the component of the date corresponding to the
                time_resolution argument is considered. If None, the end of the last complete month of
                reanalysis data will be used. Defaults to None.
            reg_model(:obj:`string`): Which model to use for the regression ("lin" for linear, "gam" for,
                general additive, "gbm" for gradient boosting, or "etr" for extra treees). At monthly
                time resolution only linear regression is allowed because of the reduced number of data
                points. Defaults to "lin".
            ml_setup_kwargs(:obj:`kwargs`): Keyword arguments to
                :py:class:`openoa.utils.machine_learning_setup.MachineLearningSetup` class. Defaults to {}.

        Returns:
            None
        """
        self.num_sim = num_sim
        initial_parameters = {}
        if reanalysis_products is not None:
            initial_parameters["reanalysis_products"] = self.reanalysis_products
            self.reanalysis_products = reanalysis_products
        if reg_model is not None:
            initial_parameters["reg_model"] = self.reg_model
            self.reg_model = reg_model
        if uncertainty_meter is not None:
            initial_parameters["uncertainty_meter"] = self.uncertainty_meter
            self.uncertainty_meter = uncertainty_meter
        if uncertainty_losses is not None:
            initial_parameters["uncertainty_losses"] = self.uncertainty_losses
            self.uncertainty_losses = uncertainty_losses
        if uncertainty_windiness is not None:
            initial_parameters["uncertainty_windiness"] = self.uncertainty_windiness
            self.uncertainty_windiness = uncertainty_windiness
        if uncertainty_loss_max is not None:
            initial_parameters["uncertainty_loss_max"] = self.uncertainty_loss_max
            self.uncertainty_loss_max = uncertainty_loss_max
        if outlier_detection is not None:
            initial_parameters["outlier_detection"] = self.outlier_detection
            self.outlier_detection = outlier_detection
        if uncertainty_outlier is not None:
            initial_parameters["uncertainty_outlier"] = self.uncertainty_outlier
            self.uncertainty_outlier = uncertainty_outlier
        if uncertainty_nan_energy is not None:
            initial_parameters["uncertainty_nan_energy"] = self.uncertainty_nan_energy
            self.uncertainty_nan_energy = uncertainty_nan_energy
        if time_resolution is not None:
            initial_parameters["time_resolution"] = self.time_resolution
            self.time_resolution = time_resolution
        if end_date_lt is not None:
            initial_parameters["end_date_lt"] = self.end_date_lt
            self.end_date_lt = end_date_lt
        if ml_setup_kwargs is not None:
            initial_parameters["ml_setup_kwargs"] = self.ml_setup_kwargs
            self.ml_setup_kwargs = ml_setup_kwargs

        # Write parameters of run to the log file
        logged_params = dict(
            uncertainty_meter=self.uncertainty_meter,
            uncertainty_losses=self.uncertainty_losses,
            uncertainty_loss_max=self.uncertainty_loss_max,
            uncertainty_windiness=self.uncertainty_windiness,
            uncertainty_nan_energy=self.uncertainty_nan_energy,
            num_sim=self.num_sim,
            reanalysis_products=self.reanalysis_products,
        )
        logger.info(f"Running with parameters: {logged_params}")

        # Start the computation
        self.calculate_long_term_losses()
        self.setup_monte_carlo_inputs()
        self.results = self.run_AEP_monte_carlo()

        # Log the completion of the run
        logger.info("Run completed")

        # Reset the class arguments back to the initialized values
        self.set_values(initial_parameters)

    @logged_method_call
    def groupby_time_res(self, df):
        """
        Group pandas dataframe based on the time resolution chosen in the calculation.

        Args:
            df(:obj:`dataframe`): dataframe that needs to be grouped based on time resolution used

        Returns:
            None
        """

        if self.time_resolution in ("MS", "ME"):
            df_grouped = df.groupby(df.index.month).mean()
        elif self.time_resolution == "D":
            df_grouped = df.groupby([(df.index.month), (df.index.day)]).mean()
        elif self.time_resolution == "h":
            df_grouped = df.groupby([(df.index.month), (df.index.day), (df.index.hour)]).mean()

        return df_grouped

    @logged_method_call
    def calculate_aggregate_dataframe(self):
        """
        Perform pre-processing of the plant data to produce a monthly/daily data frame to be used in AEP analysis.
        """

        # Average to monthly/daily, quantify NaN data
        self.process_revenue_meter_energy()

        # Average to monthly/daily, quantify NaN data, merge with revenue meter energy data
        self.process_loss_estimates()

        # Density correct wind speeds, process temperature and wind direction, average to monthly/daily
        self.process_reanalysis_data()

        # Remove first and last reporting months if only partial month reported
        # (only for monthly time resolution calculations)
        if self.time_resolution in ("MS", "ME"):
            self.trim_monthly_df()

        # Drop any data that have NaN gross energy values or NaN reanalysis data
        self.aggregate = self.aggregate.dropna(
            subset=["gross_energy_gwh"] + [product for product in self.reanalysis_products]
        )

    @logged_method_call
    def process_revenue_meter_energy(self):
        """
        Initial creation of monthly data frame:
            1. Populate monthly/daily data frame with energy data summed from 10-min QC'd data
            2. For each monthly/daily value, find percentage of NaN data used in creating it and flag if percentage is
               greater than 0
        """
        df = self.plant.meter  # Get the meter data frame

        # Create the monthly/daily data frame by summing meter energy, in GWh
        self.aggregate = df.resample(self.resample_freq)["MMTR_SupWh"].sum().to_frame() / 1e6
        self.aggregate.rename(columns={"MMTR_SupWh": "energy_gwh"}, inplace=True)

        # Determine how much 10-min data was missing for each year-month/daily energy value. Flag accordigly if any is missing
        # Get percentage of meter data that were NaN when summing to monthly/daily
        self.aggregate["energy_nan_perc"] = df.resample(self.resample_freq)["MMTR_SupWh"].apply(
            tm.percent_nan
        )

        if self.time_resolution in ("MS", "ME"):
            # Create a column with expected number of days per month (to be used when normalizing to 30-days for regression)
            days_per_month = (pd.Series(self.aggregate.index)).dt.daysinmonth
            days_per_month.index = self.aggregate.index
            self.aggregate["num_days_expected"] = days_per_month

            # Get actual number of days per month in the raw data
            # (used when trimming beginning and end of monthly data frame)
            # If meter data has higher resolution than monthly
            if self.plant.metadata.meter.frequency in ("1M", "1MS"):
                self.aggregate["num_days_actual"] = self.aggregate["num_days_expected"]
            else:
                self.aggregate["num_days_actual"] = df.resample("MS")["MMTR_SupWh"].apply(
                    tm.num_days
                )

    @logged_method_call
    def process_loss_estimates(self):
        """Append availability and curtailment losses to monthly data frame."""
        df = self.plant.curtail.copy()

        curt_aggregate = np.divide(
            df.resample(self.resample_freq)[["IAVL_DnWh", "IAVL_ExtPwrDnWh"]].sum(), 1e6
        )  # Get sum of avail and curt losses in GWh

        curt_aggregate.rename(
            columns={"IAVL_DnWh": "availability_gwh", "IAVL_ExtPwrDnWh": "curtailment_gwh"},
            inplace=True,
        )
        # Merge with revenue meter monthly/daily data
        self.aggregate = self.aggregate.join(curt_aggregate)

        # Add gross energy field
        self.aggregate["gross_energy_gwh"] = un.compute_gross_energy(
            self.aggregate["energy_gwh"],
            self.aggregate["availability_gwh"],
            self.aggregate["curtailment_gwh"],
            "energy",
            "energy",
        )

        # Calculate percentage-based losses
        self.aggregate["availability_pct"] = np.divide(
            self.aggregate["availability_gwh"], self.aggregate["gross_energy_gwh"]
        )
        self.aggregate["curtailment_pct"] = np.divide(
            self.aggregate["curtailment_gwh"], self.aggregate["gross_energy_gwh"]
        )

        # Get percentage of 10-min meter data that were NaN when summing to monthly/daily
        self.aggregate["avail_nan_perc"] = df.resample(self.resample_freq)["IAVL_DnWh"].apply(
            tm.percent_nan
        )
        self.aggregate["curt_nan_perc"] = df.resample(self.resample_freq)["IAVL_ExtPwrDnWh"].apply(
            tm.percent_nan
        )

        # If more than 1% of data are NaN, set flag to True
        self.aggregate["nan_flag"] = False  # Set flag to false by default
        ix_nan = (
            self.aggregate[["energy_nan_perc", "avail_nan_perc", "curt_nan_perc"]]
            > self.uncertainty_nan_energy
        ).any(axis=1)
        self.aggregate.loc[ix_nan, "nan_flag"] = True

        # By default, assume all reported losses are representative of long-term operational
        self.aggregate["availability_typical"] = True
        self.aggregate["curtailment_typical"] = True

        # By default, assume combined availability and curtailment losses are below the threshold to be considered valid
        self.aggregate["combined_loss_valid"] = True

    @logged_method_call
    def process_reanalysis_data(self):
        """
        Process reanalysis data for use in PRUF plant analysis:
            - calculate density-corrected wind speed and wind components
            - get monthly/daily average wind speeds and components
            - calculate monthly/daily average wind direction
            - calculate monthly/daily average temperature
            - append monthly/daily averages to monthly/daily energy data frame
        """

        # Identify start and end dates for long-term correction
        # First find date range common to all reanalysis products and drop minute field of start date
        start_date = max(
            [self.plant.reanalysis[key].index.min() for key in self.reanalysis_products]
        ).replace(minute=0)
        end_date = min([self.plant.reanalysis[key].index.max() for key in self.reanalysis_products])

        # Next, update the start date to make sure it corresponds to a full time period, by shifting
        # to either the start of the next month, or start of the next day, depending on the frequency
        start_date_minus = start_date - pd.DateOffset(hours=1)
        if (self.time_resolution in ("MS", "ME")) & (start_date.month == start_date_minus.month):
            start_date = start_date.replace(day=1, hour=0, minute=0) + pd.DateOffset(months=1)
        elif (self.time_resolution == "D") & (start_date.day == start_date_minus.day):
            start_date = start_date.replace(hour=0, minute=0) + pd.DateOffset(days=1)

        # Now determine the end date based on either the user-defined end date or the end of the
        # last full month, or last full day
        if self.end_date_lt is not None:
            # If valid (before the last full time period in the data), use the specified end date
            end_date_lt_plus = self.end_date_lt + pd.DateOffset(hours=1)
            if (self.time_resolution in ("MS", "ME")) & (
                self.end_date_lt.month == end_date_lt_plus.month
            ):
                self.end_date_lt = (
                    self.end_date_lt.replace(day=1, hour=0, minute=0)
                    + pd.DateOffset(months=1)
                    - pd.DateOffset(hours=1)
                )
            elif (self.time_resolution == "D") & (self.end_date_lt.day == end_date_lt_plus.day):
                self.end_date_lt = self.end_date_lt.replace(hour=23, minute=0)

            if self.end_date_lt > end_date:
                raise ValueError(
                    "Invalid end date for long-term correction. The end date cannot exceed the "
                    "last full time period (defined by the time resolution) in the provided "
                    "reanalysis data."
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
            index=pd.date_range(start=start_date, end=end_date, freq=self.resample_freq),
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
                    "Invalid end date argument for long-term correction. This end date does not "
                    "provide enough reanalysis data for the long-term correction."
                )
            else:
                raise ValueError(
                    "The date range of the provided reanalysis data is not long enough to "
                    "perform the long-term correction."
                )

        # Correct each reanalysis product, density-correct wind speeds, and take monthly averages
        for key in self.reanalysis_products:
            rean_df = self.plant.reanalysis[key]
            # rean_df = rean_df.rename(self.plant.metadata[key].col_map)
            rean_df["ws_dens_corr"] = mt.air_density_adjusted_wind_speed(
                rean_df["WMETR_HorWdSpd"], rean_df["WMETR_AirDen"]
            )
            self._reanalysis_aggregate[key] = rean_df.resample(self.resample_freq)[
                "ws_dens_corr"
            ].mean()  # .to_frame()

            if self.reg_wind_direction | self.reg_temperature:
                cols = [f"{key}_{var}" for var in self.reanalysis_vars]
                self._reanalysis_aggregate[cols] = (
                    rean_df[self.reanalysis_vars].resample(self.resample_freq).mean()
                )

            if self.reg_wind_direction:
                self._reanalysis_aggregate[key + "_WMETR_HorWdDir"] = np.rad2deg(
                    np.pi
                    - (
                        np.arctan2(
                            -self._reanalysis_aggregate[key + "_WMETR_HorWdSpdU"],
                            self._reanalysis_aggregate[key + "_WMETR_HorWdSpdV"],
                        )
                    )
                )  # Calculate wind direction

        self.aggregate = self.aggregate.join(
            self._reanalysis_aggregate
        )  # Merge monthly reanalysis data to monthly energy data frame

    @logged_method_call
    def trim_monthly_df(self):
        """
        Remove first and/or last month of data if the raw data had an incomplete number of days.
        """
        for p in self.aggregate.index[[0, -1]]:  # Loop through 1st and last data entry
            if (
                self.aggregate.loc[p, "num_days_expected"]
                != self.aggregate.loc[p, "num_days_actual"]
            ):
                self.aggregate.drop(p, inplace=True)  # Drop the row from data frame

    @logged_method_call
    def calculate_long_term_losses(self):
        """
        This function calculates long-term availability and curtailment losses based on the reported
        data grouped by the time resolution, filtering for those data that are deemed representative
        of average plant performance.
        """
        df = self.aggregate

        # isolate availabilty and curtailment values that are representative of average plant performance
        avail_valid = df.loc[df["availability_typical"], "availability_pct"].to_frame()
        curt_valid = df.loc[df["curtailment_typical"], "curtailment_pct"].to_frame()

        # Now get average percentage losses by month or day
        avail_long_term = self.groupby_time_res(avail_valid)["availability_pct"]
        curt_long_term = self.groupby_time_res(curt_valid)["curtailment_pct"]

        # Ensure there are 12 or 365 data points in long-term average. If not, throw an exception:
        if avail_long_term.shape[0] < self.calendar_samples:
            raise Exception(
                "Not all calendar days/months represented in long-term availability calculation"
            )
        if curt_long_term.shape[0] < self.calendar_samples:
            raise Exception(
                "Not all calendar days/months represented in long-term curtailment calculation"
            )

        self.long_term_losses = (avail_long_term, curt_long_term)

    @logged_method_call
    def setup_monte_carlo_inputs(self):
        """
        Create and populate the data frame defining the simulation parameters.
        This data frame is stored as self.mc_inputs
        """

        # Create extra long list of renanalysis product names to sample from
        reanal_list = list(np.repeat(self.reanalysis_products, self.num_sim))

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

        self.mc_inputs = pd.DataFrame(inputs)

    @logged_method_call
    def filter_outliers(self, n):
        """
        This function filters outliers based on a combination of range filter, unresponsive sensor
        filter, and window filter.

        We use a memoized funciton to store the regression data in a dictionary for each
        combination as it comes up in the Monte Carlo simulation. This saves significant
        computational time in not having to run robust linear regression for each Monte Carlo
        iteration.

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
        df = self.aggregate

        # First set of filters checking combined losses and if the Nan data flag was on
        df_sub = df.loc[
            ((df["availability_pct"] + df["curtailment_pct"]) < self._run.loss_threshold)
            & (~df["nan_flag"]),
            :,
        ]

        # Set maximum range for using bin-filter, convert from MW to GWh
        plant_capac = self.plant.metadata.capacity / 1000.0 * self.resample_hours

        # Apply range filter to wind speed
        df_sub = df_sub.assign(flag_range=filters.range_flag(df_sub[reanal], lower=0, upper=40))
        if self.reg_temperature:
            # Apply range filter to temperature, in Kelvin
            df_sub = df_sub.assign(
                flag_range_T=filters.range_flag(
                    df_sub[f"{reanal}_WMETR_EnvTmp"], lower=200, upper=320
                )
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
            if self.time_resolution in ("MS", "ME"):
                # Monthly linear regression (i.e., few data points):
                # flag outliers with robust linear regression using Huber algorithm

                # Reanalysis data with constant column, and energy data normalized to 30 days
                X = sm.add_constant(df_sub[reanal])
                y = df_sub["gross_energy_gwh"] * 30 / df_sub["num_days_expected"]

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
        df_sub.loc[:, "flag_final"] = df_sub[["flag_range", "flag_window", "flag_outliers"]].any(
            axis=1
        )
        if self.reg_temperature:
            df_sub.loc[:, "flag_final"] = df_sub[["flag_final", "flag_range_T"]].any(axis=1)

        # Define valid data
        valid_data = df_sub.loc[
            ~df_sub.loc[:, "flag_final"],
            [reanal, "energy_gwh", "availability_gwh", "curtailment_gwh"],
        ]
        if self.reg_wind_direction:
            add_cols = [
                f"{reanal}_{x}" for x in ("WMETR_HorWdDir", "WMETR_HorWdSpdU", "WMETR_HorWdSpdV")
            ]
            valid_data_to_add = df_sub.loc[~df_sub.loc[:, "flag_final"], add_cols]
            valid_data = pd.concat([valid_data, valid_data_to_add], axis=1)

        if self.reg_temperature:
            valid_data_to_add = df_sub.loc[~df_sub.loc[:, "flag_final"], [f"{reanal}_WMETR_EnvTmp"]]
            valid_data = pd.concat([valid_data, valid_data_to_add], axis=1)

        if self.time_resolution in ("MS", "ME"):
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
        # Create new Monte-Carlo sampled data frame and sample energy data, calculate MC-generated
        # availability and curtailment
        mc_energy = reg_data["energy_gwh"] * self._run.metered_energy_fraction
        mc_availability = reg_data["availability_gwh"] * self._run.loss_fraction
        mc_curtailment = reg_data["curtailment_gwh"] * self._run.loss_fraction

        # Calculate gorss energy and normalize to 30-days
        mc_gross_energy = mc_energy + mc_availability + mc_curtailment
        if self.time_resolution in ("MS", "ME"):
            num_days_expected = reg_data["num_days_expected"]
            mc_gross_norm = mc_gross_energy * 30 / num_days_expected
        else:
            mc_gross_norm = mc_gross_energy

        # Set reanalysis product for MC inputs
        reg_inputs = reg_data[self._run.reanalysis_product]

        if self.reg_temperature:  # if temperature is considered as regression variable
            mc_temperature = reg_data[f"{self._run.reanalysis_product}_WMETR_EnvTmp"]
            reg_inputs = pd.concat([reg_inputs, mc_temperature], axis=1)

        if self.reg_wind_direction:  # if wind direction is considered as regression variable
            mc_wind_direction = reg_data[f"{self._run.reanalysis_product}_WMETR_HorWdDir"]
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
            n(:obj:`int`): The Monte Carlo iteration number.

        Returns:
            A trained regression model.
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
            if self.plant.log_level in ("WARNING", "ERROR", "CRITICAL", "INFO"):
                verbosity = 0
            else:
                verbosity = 2
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
                    verbose=verbosity,
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
        if self.reg_wind_direction:
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
            self._run = self.mc_inputs.loc[n]

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
            reg_inputs_por = [self.reanalysis_por[self._run.reanalysis_product]]
            if self.reg_temperature:
                reg_inputs_por += [
                    self.reanalysis_por[self._run.reanalysis_product + "_WMETR_EnvTmp"]
                ]
            if self.reg_wind_direction:
                reg_inputs_por += [
                    np.sin(
                        np.deg2rad(
                            self.reanalysis_por[self._run.reanalysis_product + "_WMETR_HorWdDir"]
                        )
                    )
                ]
                reg_inputs_por += [
                    np.cos(
                        np.deg2rad(
                            self.reanalysis_por[self._run.reanalysis_product + "_WMETR_HorWdDir"]
                        )
                    )
                ]
            gross_por = fitted_model.predict(np.array(pd.concat(reg_inputs_por, axis=1)))

            # Create padans dataframe for gross_por and group by calendar date to have a single full year
            gross_por = self.groupby_time_res(
                pd.DataFrame(
                    data=gross_por, index=self.reanalysis_por[self._run.reanalysis_product].index
                )
            )

            if self.time_resolution in ("MS", "ME"):  # Undo normalization to 30-day months
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
            gps = (
                gross_por.sum()
                if not isinstance(gross_por, (pd.Series, pd.DataFrame))
                else gross_por.values.sum()
            )
            lt_por_ratio[n] = (gross_lt.sum() / self._run.num_years_windiness) / gps

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
        This function returns the long-term monthly/daily wind speeds based on the Monte-Carlo
        generated sample of:

            1. The reanalysis product
            2. The number of years to use in the long-term correction

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
        namescol = [f"{self._run.reanalysis_product}_{var}" for var in self.reanalysis_vars]
        long_term_temp = self._reanalysis_aggregate[namescol].dropna()[
            ws_df.index[-1]
            + ws_df.index.freq
            - pd.offsets.DateOffset(years=self._run.num_years_windiness) :
        ]
        if self.reg_temperature:
            long_term_reg_inputs = pd.concat(
                [
                    long_term_reg_inputs,
                    long_term_temp[f"{self._run.reanalysis_product}_WMETR_EnvTmp"],
                ],
                axis=1,
            )
        if self.reg_wind_direction:
            wd_aggregate = np.rad2deg(
                np.pi
                - np.arctan2(
                    -long_term_temp[f"{self._run.reanalysis_product}_WMETR_HorWdSpdU"],
                    long_term_temp[f"{self._run.reanalysis_product}_WMETR_HorWdSpdV"],
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
        """Make a plot of the normalized annual average wind speeds from reanalysis data to show
        general trends for each, and highlighting the period of record for the plant data.

        Args:
            aep (:obj:`openoa.analysis.MonteCarloAEP`): An initialized MonteCarloAEP object.
            xlim (:obj:`tuple[datetime.datetime, datetime.datetime]`, optional): A tuple of datetimes
                representing the x-axis plotting display limits. Defaults to (None, None).
            ylim (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display limits.
                Defaults to (None, None).
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to ``plt.figure()``. Defaults to {}.
            plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                ``ax.plot()``. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                ``ax.legend()``. Defaults to {}.

        Returns:
            None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]: If ``return_fig`` is
                True, then the figure and axes objects are returned for further tinkering/saving.
        """
        return plot.plot_monthly_reanalysis_windspeed(
            data=self.plant.reanalysis,
            windspeed_col="ws_dens_corr",
            plant_por=(self.aggregate.index[0], self.aggregate.index[-1]),
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
        highlighted in a contrasting color and separate marker.

        Args:
            reanalysis (:obj:`dict[str, pandas.DataFrame]`): :py:attr:`PlantData.reanalysis`
                dictionary of reanalysis :py:class:`DataFrame`.
            outlier_thres (:obj:`float`): outlier threshold (typical range of 1 to 4) which adjusts
                outlier sensitivity detection.
            xlim (:obj:`tuple[float, float]`, optional): A tuple of datetimes
                representing the x-axis plotting display limits. Defaults to (None, None).
            ylim (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display limits.
                Defaults to (None, None).
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to ``plt.figure()``. Defaults to {}.
            plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                ``ax.scatter()``. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                ``ax.legend()``. Defaults to {}.

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

        valid_aggregate = self.aggregate

        # Monthly case: apply robust linear regression for outliers detection
        if self.time_resolution in ("MS", "ME"):
            for name, df in self.plant.reanalysis.items():
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

            ax.set_ylabel("30-day Normalized Gross Energy (GWh)")

        # Daily/hourly case: apply bin filter for outliers detection
        else:
            for name, df in self.plant.reanalysis.items():
                x = valid_aggregate[name]
                y = valid_aggregate["gross_energy_gwh"]
                plant_capac = self.plant.metadata.capacity / 1000.0 * self._hours_in_res

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
            elif self.time_resolution == "h":
                ax.set_ylabel("Hourly gross energy (GWh)")

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
                that are passed to ``plt.figure()``. Defaults to {}.
            plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                ``ax.scatter()``. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                ``ax.legend()``. Defaults to {}.

        Returns:
            None | tuple[matplotlib.pyplot.Figure, tuple[matplotlib.pyplot.Axes, matplotlib.pyplot.Axes]]:
                If `return_fig` is True, then the figure and axes objects are returned for further
                tinkering/saving.
        """
        return plot.plot_plant_energy_losses_timeseries(
            data=self.aggregate,
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
                that are passed to ``plt.figure()``. Defaults to {}.
            plot_kwargs (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                ``ax.hist()``. Defaults to {}.
            annotate_kwargs (:obj:`dict`, optional): Additional annotation keyword arguments that are
                passed to ``ax.annotate()``. Defaults to {}.

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
        points_label: str = "Individual AEP Estimates",
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
            with_points (:obj:`bool`, optional): Flag to plot the individual points like a seaborn
                ``swarmplot``. Defaults to False.
            points_label(:obj:`bool` | None, optional): Legend label for the points, if plotting.
                Defaults to None.
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to ``plt.figure()``. Defaults to {}.
            plot_kwargs_box (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                ``ax.boxplot()``. Defaults to {}.
            plot_kwargs_points (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                ``ax.boxplot()``. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                ``ax.legend()``. Defaults to {}.

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
            points_label=points_label,
            return_fig=return_fig,
            figure_kwargs=figure_kwargs,
            plot_kwargs_box=plot_kwargs_box,
            plot_kwargs_points=plot_kwargs_points,
            legend_kwargs=legend_kwargs,
        )


__defaults_reanalysis_products = MonteCarloAEP.__attrs_attrs__.reanalysis_products.default
__defaults_uncertainty_meter = MonteCarloAEP.__attrs_attrs__.uncertainty_meter.default
__defaults_uncertainty_losses = MonteCarloAEP.__attrs_attrs__.uncertainty_losses.default
__defaults_uncertainty_windiness = MonteCarloAEP.__attrs_attrs__.uncertainty_windiness.default
__defaults_uncertainty_loss_max = MonteCarloAEP.__attrs_attrs__.uncertainty_loss_max.default
__defaults_outlier_detection = MonteCarloAEP.__attrs_attrs__.outlier_detection.default
__defaults_uncertainty_outlier = MonteCarloAEP.__attrs_attrs__.uncertainty_outlier.default
__defaults_uncertainty_nan_energy = MonteCarloAEP.__attrs_attrs__.uncertainty_nan_energy.default
__defaults_time_resolution = MonteCarloAEP.__attrs_attrs__.time_resolution.default
__defaults_end_date_lt = MonteCarloAEP.__attrs_attrs__.end_date_lt.default
__defaults_reg_model = MonteCarloAEP.__attrs_attrs__.reg_model.default
__defaults_ml_setup_kwargs = MonteCarloAEP.__attrs_attrs__.ml_setup_kwargs.default
__defaults_reg_temperature = MonteCarloAEP.__attrs_attrs__.reg_temperature.default
__defaults_reg_wind_direction = MonteCarloAEP.__attrs_attrs__.reg_wind_direction.default


def create_MonteCarloAEP(
    project: PlantData,
    reanalysis_products: list[str] = __defaults_reanalysis_products,
    uncertainty_meter: float = __defaults_uncertainty_meter,
    uncertainty_losses: float = __defaults_uncertainty_losses,
    uncertainty_windiness: NDArrayFloat = __defaults_uncertainty_windiness,
    uncertainty_loss_max: NDArrayFloat = __defaults_uncertainty_loss_max,
    outlier_detection: bool = __defaults_outlier_detection,
    uncertainty_outlier: NDArrayFloat = __defaults_uncertainty_outlier,
    uncertainty_nan_energy: float = __defaults_uncertainty_nan_energy,
    time_resolution: str = __defaults_time_resolution,
    end_date_lt: str | pd.Timestamp = __defaults_end_date_lt,
    reg_model: str = __defaults_reg_model,
    ml_setup_kwargs: dict = __defaults_ml_setup_kwargs,
    reg_temperature: bool = __defaults_reg_temperature,
    reg_wind_direction: bool = __defaults_reg_wind_direction,
) -> MonteCarloAEP:
    return MonteCarloAEP(
        plant=project,
        reanalysis_products=reanalysis_products,
        uncertainty_meter=uncertainty_meter,
        uncertainty_losses=uncertainty_losses,
        uncertainty_windiness=uncertainty_windiness,
        uncertainty_loss_max=uncertainty_loss_max,
        outlier_detection=outlier_detection,
        uncertainty_outlier=uncertainty_outlier,
        uncertainty_nan_energy=uncertainty_nan_energy,
        time_resolution=time_resolution,
        end_date_lt=end_date_lt,
        reg_model=reg_model,
        ml_setup_kwargs=ml_setup_kwargs,
        reg_temperature=reg_temperature,
        reg_wind_direction=reg_wind_direction,
    )


create_MonteCarloAEP.__doc__ = MonteCarloAEP.__doc__
