# This class defines key analytical routines for estimating wake losses for an operating
# wind plant using SCADA data. At a high level, for each SCADA time step, freestream wind
# turbines are identified using the turbine coordinates and a reference wind direction
# signal. The mean power production for all turbines in the wind plant is summed over all
# time steps and compared to the mean power of the freestream turbines summed over all time
# steps to estimate wake losses during the period of record. Methods for calclating the
# long-term wake losses using reanalaysis data and quantifying uncertainty are provided as well.

from __future__ import annotations

import random

import attrs
import numpy as np
import pandas as pd
import numpy.typing as npt
from tqdm import tqdm
from attrs import field, define
from sklearn.linear_model import LinearRegression

from openoa.plant import PlantData, FromDictMixin
from openoa.utils import filters
from openoa.logging import logging, logged_method_call
from openoa.utils.plot import set_styling
from openoa.analysis._analysis_validators import validate_UQ_input


logger = logging.getLogger(__name__)

NDArrayFloat = npt.NDArray[np.float64]


@define(auto_attribs=True)
class WakeLosses(FromDictMixin):
    """
    A serial implementation of a method for estimating wake losses from SCADA data. Wake losses are
    estimated for the entire wind plant as well as for each individual turbine for a) the period of
    record for which data are available, and b) the estimated long-term wind conditions the wind
    plant will experience based on historical reanalysis wind resource data.

    The method is comprised of the following core steps:
        1. Calculate a representative wind plant-level wind direction at each time step using the
           mean wind direction of the specified of wind turbines or meteorological (met) towers.
           Note that time steps for which any necessary plant-level or turbine-level data are
           missing are discarded.

           a. If :py:attr:`UQ` is selected, wake losses are calculated multiple times using a Monte
              Carlo approach with randomly chosen analysis parameters and randomly sampled, with
              replacement, time steps for each iteration. The remaining steps described below are
              performed for each Monte Carlo iteration. If UQ is not used, wake losses are calculated
              once using the specified analysis parameters for the full set of available time steps.

        2. Identify the set of derated, curtailed, or unavailable turbines (i.e., turbines whose power
           production is limited not by wake losses but by operating mode)for each time step using a
           power curve outlier detection.
        3. Calculate the average wind speed and power production for the set of normally operating
           (i.e., not derated) freestream turbines.

           a. Freestream turbines are those with upstream turbines located within a user-specified
              sector of wind directions centered on the bin center direction.

        4. Calculate the POR losses for the wind plant by comparing the potential energy production
           (sum of the mean freestream power production multiplied by the number of turbines in the
           wind power plant) to the actual energy production (sum of the actual wind plant power
           production at each time step).

           a. If :py:attr:`correct_for_derating` is True, then the potential power production of the
              wind plant is assumed to be the actual power produced by the derated turbines plus the
              mean power production of the freestream turbines for all other turbines in the wind
              plant. This procedure is then used to estimate the wake losses for each individual
              wind turbine.

        5. Finally, estimate the long-term corrected wake losses using the long-term historical
           reanalysis data.

           a. Calculate the long-term occurence frequencies for a set of wind direction and wind
              speed bins based on the hour reanalysis data (typically, 10-20 years).
           b. Next, using a linear regression, compare the mean freestream wind speeds calculated
              from the SCADA data to the wind speeds from the reanalysis data and correct to remove
              biases.
           c. Compute the average potential and actual wind power plant production using the
              representative wind plant wind directions from the SCADA or met tower data in
              conjunction with the corrected freestream wind speeds for each wind direction and wind
              speed bin.
           d. Estimate the long-term corrected wake losses by comparing the long-term
              corrected potential and actual energy production. These are computed by weighting
              the average potential and actual power production for each with condition bin
              with the long-term frequencies.

        6. Repeat to estimate the long-term corrected wake losses for each individual turbine. Note
           that the long-term correction is determined for each reanalysis product specified by the
           user. If UQ is used, a random reanalysis product is selected each iteration. If UQ is not
           selected, the long-term corrected wake losses are calculated as the average wake losses
           determined for all reanalysis products.

    Args:
        plant (:obj:`PlantData`): A :py:attr:`openoa.plant.PlantData` object that has been validated
            with at least `:py:attr:`openoa.plant.PlantData.analysis_type` = "WakeLosses".
        wind_direction_col (:obj:`string`, optional): Column name to use for wind direction.
            Defaults to "wind_direction"
        wind_direction_data_type (:obj:`string`, optional): Data type to use for wind directions
            ("scada" for turbine measurements or "tower" for meteorological tower measurements).
            Defaults to "scada".
        wind_direction_asset_ids (:obj:`list`, optional): List of asset IDs (turbines or met towers)
            used to calculate the average wind direction at each time step. If None, all assets of
            the corresponding data type will be used. Defaults to None.
        UQ (:obj:`bool`, optional): Dertermines whether to perform uncertainty quantification using
            Monte Carlo simulation (True) or provide a single wake loss estimate (False). Defaults
            to True.
        start_date (:obj:`pandas.Timestamp` or :obj:`string`, optional): Start datetime for wake
            loss analysis. If None, the earliest SCADA datetime will be used. Default is None.
        end_date (:obj:`pandas.Timestamp` or :obj:`string`, optional): End datetime for wake loss
            analysis. If None, the latest SCADA datetime will be used. Default is None.
        reanal_products (:obj:`list`, optional): List of reanalysis products to use for long-term
            correction. If UQ = True, a single product will be selected form this list each Monte
            Carlo iteration. Defaults to ["merra2", "era5"].
        end_date_lt (:obj:`string` or :obj:`pandas.Timestamp`): The last date to use for the
            long-term correction. If None, the most recent date common to all reanalysis products
            will be used.
    """

    plant: PlantData = field(validator=attrs.validators.instance_of(PlantData))
    wind_direction_col: str = field(default="wind_direction", converter=str)
    wind_direction_data_type: str = field(
        default="scada", validator=attrs.validators.in_(("scada", "tower"))
    )
    wind_direction_asset_ids: list[str] = field(default=None)
    UQ: bool = field(default=True, converter=bool)
    start_date: str | pd.Timestamp = field(default=None)
    end_date: str | pd.Timestamp = field(default=None)
    reanal_products: list[str] = field(default=["merra2", "era5"])
    end_date_lt: str | pd.Timestamp = field(default=None)

    # Internally created attributes need to be given a type before usage
    turbine_ids: list[str] = field(init=False)
    aggregate_df: pd.DataFrame = field(init=False)
    inputs: pd.DataFrame = field(init=False)
    aggregate_df_sample: pd.DataFrame = field(init=False)
    wake_losses_por: NDArrayFloat = field(init=False)
    turbine_wake_losses_por: NDArrayFloat = field(init=False)
    wake_losses_lt: NDArrayFloat = field(init=False)
    turbine_wake_losses_lt: NDArrayFloat = field(init=False)
    wake_losses_por_wd: NDArrayFloat = field(init=False)
    turbine_wake_losses_por_wd: NDArrayFloat = field(init=False)
    wake_losses_lt_wd: NDArrayFloat = field(init=False)
    turbine_wake_losses_lt_wd: NDArrayFloat = field(init=False)
    energy_por_wd: NDArrayFloat = field(init=False)
    energy_lt_wd: NDArrayFloat = field(init=False)
    wake_losses_por_ws: NDArrayFloat = field(init=False)
    turbine_wake_losses_por_ws: NDArrayFloat = field(init=False)
    wake_losses_lt_ws: NDArrayFloat = field(init=False)
    turbine_wake_losses_lt_ws: NDArrayFloat = field(init=False)
    energy_por_ws: NDArrayFloat = field(init=False)
    energy_lt_ws: NDArrayFloat = field(init=False)
    wake_losses_lt_mean: float = field(init=False)
    turbine_wake_losses_lt_mean: float = field(init=False)
    wake_losses_por_mean: float = field(init=False)
    turbine_wake_losses_por_mean: float = field(init=False)
    wake_losses_lt_std: float = field(init=False)
    turbine_wake_losses_lt_std: float = field(init=False)
    wake_losses_por_std: float = field(init=False)
    turbine_wake_losses_por_std: float = field(init=False)
    _num_sim: int = field(init=False)
    _wd_bin_width_LT_corr: float = field(init=False)
    _ws_bin_width_LT_corr: float = field(init=False)
    _assume_no_wakes_high_ws_LT_corr: bool = field(init=False)
    _no_wakes_ws_thresh_LT_corr: float = field(init=False)
    _freestream_sector_width: float | tuple[float, float] = field(init=False)
    _derating_filter_wind_speed_start: float | tuple[float, float] = field(init=False)
    _max_power_filter: float | tuple[float, float] = field(init=False)
    _wind_bin_mad_thresh: float | tuple[float, float] = field(init=False)
    _num_years_LT: float | tuple[float, float] = field(init=False)
    _run: pd.DataFrame = field(init=False)

    @plant.validator
    def validate_plant_ready_for_anylsis(
        self, attribute: attrs.Attribute, value: PlantData
    ) -> None:
        """Validates that the value has been validated for a wake loss analysis."""
        if set(("WakeLosses", "all")).intersection(value.analysis_type) == set():
            raise TypeError("The input to 'plant' must be validated for at least 'WakeLosses'")

    @logged_method_call
    def __attrs_post_init__(self):
        """
        Initialize logging and post-initialization setup steps.
        """
        logger.info("Initializing WakeLosses analysis object")

        # Check that selected UQ is allowed and reset num_sim if no UQ
        if self.UQ:
            logger.info("Note: uncertainty quantification will be performed in the calculation")
        else:
            logger.info("Note: uncertainty quantification will NOT be performed in the calculation")

        # set default start and end dates if undefined
        if self.start_date is None:
            self.start_date = self.plant.scada.index.get_level_values("time").min()

        if self.end_date is None:
            self.end_date = self.plant.scada.index.get_level_values("time").max()

        self.turbine_ids = list(self.plant.turbine_ids)

        if (self.wind_direction_asset_ids is None) & (self.wind_direction_data_type == "scada"):
            self.wind_direction_asset_ids = self.turbine_ids
        elif (self.wind_direction_asset_ids is None) & (self.wind_direction_data_type == "tower"):
            self.wind_direction_asset_ids = list(self.plant.tower_ids)

        if self.end_date_lt is not None:
            # Set minutes to 30 to handle time indices on the hour and on the half hour
            self.end_date_lt = pd.to_datetime(self.end_date_lt).replace(minute=30)
        else:
            # Find most recent time common to all reanalysis products
            self.end_date_lt = min(
                [self.plant.reanalysis[product].index.max() for product in self.reanal_products]
            ).replace(minute=30)

        # Run preprocessing steps
        self._calculate_aggregate_dataframe()

    @logged_method_call
    def run(
        self,
        num_sim: int = 100,
        wd_bin_width: float = 5.0,
        freestream_sector_width: float = None,
        freestream_power_method: str = "mean",
        freestream_wind_speed_method: str = "mean",
        correct_for_derating: bool = True,
        derating_filter_wind_speed_start: float = None,
        max_power_filter: float = None,
        wind_bin_mad_thresh: float = None,
        wd_bin_width_LT_corr: float = 5.0,
        ws_bin_width_LT_corr: float = 1.0,
        num_years_LT: int = None,
        assume_no_wakes_high_ws_LT_corr: bool = True,
        no_wakes_ws_thresh_LT_corr: float = 13.0,
    ):
        """
        Estimates wake losses by comparing wind plant energy production to energy production of turbines identified as
        operating in freestream conditions. Wake losses are expressed as a fractional loss (e.g., 0.05 indicates a wake
        loss values of 5%).

        Args:
            num_sim (int, optional): Number of Monte Carlo iterations to perform. Only used if UQ = True.
                Defaults to 100.
            wd_bin_width (float, optional): Wind direction bin size when identifying freestream wind turbines
                (degrees). Defaults to 5 degrees.
            freestream_sector_width (tuple | float, optional): Wind direction sector size to use when
                identifying freestream wind turbines (degrees). If no turbines are located upstream of a particular
                turbine within the sector, the turbine will be classified as a freestream turbine. This should be a
                tuple when UQ = True (values are Monte-Carlo sampled within the specified range) or a single value when
                UQ = False. If undefined (None), a value of 90 degrees will be used if UQ = False and values of (50,
                110) will be used if UQ = True. Defaults to None.
            freestream_power_method (str, optional): Method used to determine the representative power
                prouction of the freestream turbines ("mean", "median", "max"). Defaults to "mean".
            freestream_wind_speed_method (str, optional): Method used to determine the representative wind
                speed of the freestream turbines ("mean", "median"). Defaults to "mean".
            correct_for_derating (bool, optional): Indicates whether derated, curtailed, or otherwise
                unavailable turbines should be flagged and excluded from the calculation of ideal freestream wind plant
                power production for a given time stamp. If True, ideal freestream power production will be calculated
                as the sum of the derated turbine powers added to the mean power of the freestream turbines in normal
                operation multiplied by the number of turbines operating normally in the wind plant. Defaults to True.
            derating_filter_wind_speed_start (tuple | float, optional): The wind speed above which
                turbines will be flagged as derated/curtailed/shutdown if power is less than 1% of rated power (m/s).
                Only used when correct_for_derating is True. This should be a tuple when UQ = True (values are
                Monte-Carlo sampled within the specified range) or a single value when UQ = False. If undefined (None),
                a value of 4.5 m/s will be used if UQ = False and values of (4.0, 5.0) will be used if UQ = True.
                Defaults to None.
            max_power_filter (tuple | float, optional): Maximum power threshold, defined as a fraction
                of rated power, to which the power curve bin filter should be applied. Only used when
                correct_for_derating is True. This should be a tuple when UQ = True (values are Monte-Carlo sampled
                within the specified range) or a single value when UQ = False. If undefined (None), a value of 0.95 will
                be used if UQ = False and values of (0.92, 0.98) will be used if UQ = True. Defaults to None.
            wind_bin_mad_thresh (tuple | float, optional): The filter threshold for each power bin used
                to identify derated/curtailed/shutdown turbines, expressed as the number of median absolute deviations
                above the median wind speed. Only used when correct_for_derating is True. This should be a tuple when
                UQ = True (values are Monte-Carlo sampled within the specified range) or a single value when UQ =
                False. If undefined (None), a value of 7.0 will be used if UQ = False and values of (4.0,
                13.0) will be used if UQ = True. Defaults to None.
            wd_bin_width_LT_corr (float, optional): Size of wind direction bins used to calculate long-term
                frequencies from historical reanalysis data and correct wake losses during the period of record
                (degrees). Defaults to 5 degrees.
            ws_bin_width_LT_corr (float, optional): Size of wind speed bins used to calculate long-term
                frequencies from historical reanalysis data and correct wake losses during the period of record (m/s).
                Defaults to 1 m/s.
            num_years_LT (tuple | int, optional): Number of years of historical reanalysis data to use
                for long-term correction. This should be a tuple when UQ = True (values are Monte-Carlo sampled within
                the specified range) or a single value when UQ = False. If undefined (None), a value of 20 will be
                used if UQ = False and values of (10, 20) will be used if UQ = True. Defaults to None.
            assume_no_wakes_high_ws_LT_corr (bool, optional): If True, wind direction and wind speed bins for
                which operational data are missing above a certain wind speed threshold are corrected by assigning the
                wind turbines' rated power to both the actual and potential power production variables during the long
                term-correction process. This assumes there are no wake losses above the wind speed threshold. Defaults
                to True.
            no_wakes_ws_thresh_LT_corr (float, optional): The wind speed threshold (inclusive) above which rated
                power is assigned to both the actual and potential power production variables if operational data are
                missing for any wind direction and wind speed bin during the long term-correction process. This wind
                speed corresponds to the wind speed measured at freestream wind turbines. Only used if
                assume_no_wakes_high_ws_LT_corr is True. Defaults to 13 m/s.
        Returns:
            (None)
        """

        self._num_sim = num_sim
        self._wd_bin_width_LT_corr = wd_bin_width_LT_corr
        self._ws_bin_width_LT_corr = ws_bin_width_LT_corr
        self._assume_no_wakes_high_ws_LT_corr = assume_no_wakes_high_ws_LT_corr
        self._no_wakes_ws_thresh_LT_corr = no_wakes_ws_thresh_LT_corr

        # Assign default parameter values depending on whether UQ is performed
        if freestream_sector_width is not None:
            self._freestream_sector_width = freestream_sector_width
        elif self.UQ:
            self._freestream_sector_width = (50.0, 110.0)
        else:
            self._freestream_sector_width = 90.0

        if derating_filter_wind_speed_start is not None:
            self._derating_filter_wind_speed_start = derating_filter_wind_speed_start
        elif self.UQ:
            self._derating_filter_wind_speed_start = (4.0, 5.0)
        else:
            self._derating_filter_wind_speed_start = 4.5

        if max_power_filter is not None:
            self._max_power_filter = max_power_filter
        elif self.UQ:
            self._max_power_filter = (0.92, 0.98)
        else:
            self._max_power_filter = 0.95

        if wind_bin_mad_thresh is not None:
            self._wind_bin_mad_thresh = wind_bin_mad_thresh
        elif self.UQ:
            self._wind_bin_mad_thresh = (4.0, 13.0)
        else:
            self._wind_bin_mad_thresh = 7.0

        if num_years_LT is not None:
            self._num_years_LT = num_years_LT
        elif self.UQ:
            self._num_years_LT = (10, 20)
        else:
            self._num_years_LT = 20

        # Set up Monte Carlo simulation inputs if UQ = True or single simulation inputs if UQ = False.
        self._setup_monte_carlo_inputs()

        for n in tqdm(range(self._num_sim)):

            self._run = self.inputs.loc[n].copy()

            # Estimate periods when each turbine is unavailable, derated, or curtailed, based on power curve filtering
            for t in self.turbine_ids:
                self.aggregate_df[("derate_flag", t)] = False

            if correct_for_derating:
                self._identify_derating()

            # Randomly resample 10-minute periods for bootstrapping
            if self.UQ:
                self.aggregate_df_sample = self.aggregate_df.sample(frac=1.0, replace=True)
            else:
                self.aggregate_df_sample = self.aggregate_df.copy()

            # For a set of wind direction bins, identify freestream turbines and calculate mean energy production and
            # wind speed
            self.aggregate_df_sample["power_mean_freestream"] = np.nan
            self.aggregate_df_sample["windspeed_mean_freestream"] = np.nan

            wd_bins = np.arange(0.0, 360.0, wd_bin_width)

            # Create columns for turbine power and wind speed during normal operation (NaN otherwise)

            for t in self.turbine_ids:
                valid_inds = ~self.aggregate_df_sample[("derate_flag", t)]
                self.aggregate_df_sample.loc[
                    valid_inds, ("power_normal", t)
                ] = self.aggregate_df_sample.loc[valid_inds, ("power", t)]

            for t in self.turbine_ids:
                valid_inds = ~self.aggregate_df_sample[("derate_flag", t)]
                self.aggregate_df_sample.loc[
                    valid_inds, ("windspeed_normal", t)
                ] = self.aggregate_df_sample.loc[valid_inds, ("windspeed", t)]

            # Find freestream turbines for each wind direction. Update the dictionary only when the set of turbines
            # differs from the previous wind direction bin.
            freestream_turbine_dict = {}

            freestream_turbine_ids_prev = []

            for wd in wd_bins:

                # identify freestream turbines
                freestream_turbine_ids = self.plant.get_freestream_turbines(
                    wd, sector_width=self._run.freestream_sector_width
                )

                if freestream_turbine_ids != freestream_turbine_ids_prev:
                    freestream_turbine_dict[wd] = freestream_turbine_ids
                    freestream_turbine_ids_prev = freestream_turbine_ids

            if freestream_turbine_dict[0.0] == list(freestream_turbine_dict.values())[-1]:
                freestream_turbine_dict.pop(0.0)

            # Find freestream energy production for each wind direction sector containing the same freestream turbines

            freestream_sector_wds = list(freestream_turbine_dict.keys())

            for i_wd, wd in enumerate(freestream_sector_wds):

                freestream_turbine_ids = freestream_turbine_dict[wd]

                # if UQ is enabled, randomly resample set of freestream turbines
                if self.UQ:
                    freestream_turbine_ids = random.choices(
                        freestream_turbine_ids, k=len(freestream_turbine_ids)
                    )

                # Check whether last wind direction in dictionary and handle wind direction wrapping
                # between 0 and 360 degrees
                if wd == 0.0:
                    wd_bin_flag = (
                        self.aggregate_df_sample["wind_direction_ref"]
                        >= (360.0 - 0.5 * wd_bin_width)
                    ) | (
                        self.aggregate_df_sample["wind_direction_ref"]
                        < (freestream_sector_wds[i_wd + 1] - 0.5 * wd_bin_width)
                    )
                elif i_wd < len(freestream_sector_wds) - 1:
                    wd_bin_flag = (
                        self.aggregate_df_sample["wind_direction_ref"] >= (wd - 0.5 * wd_bin_width)
                    ) & (
                        self.aggregate_df_sample["wind_direction_ref"]
                        < (freestream_sector_wds[i_wd + 1] - 0.5 * wd_bin_width)
                    )
                elif (i_wd == len(freestream_sector_wds) - 1) & (freestream_sector_wds[0] == 0.0):
                    wd_bin_flag = (
                        self.aggregate_df_sample["wind_direction_ref"] >= (wd - 0.5 * wd_bin_width)
                    ) & (
                        self.aggregate_df_sample["wind_direction_ref"]
                        < (360.0 - 0.5 * wd_bin_width)
                    )
                else:  # last wind direction in dictionary and first wind direction is not zero:
                    wd_bin_flag = (
                        self.aggregate_df_sample["wind_direction_ref"] >= (wd - 0.5 * wd_bin_width)
                    ) | (
                        self.aggregate_df_sample["wind_direction_ref"]
                        < (freestream_sector_wds[0] - 0.5 * wd_bin_width)
                    )

                # Assign representative energy and wind speed of freestream turbines. If correct_for_derating
                # is True, only freestream turbines operating normally will be considered.

                if freestream_power_method == "mean":
                    self.aggregate_df_sample.loc[wd_bin_flag, "power_mean_freestream"] = (
                        self.aggregate_df_sample.loc[wd_bin_flag, "power_normal"]
                    )[freestream_turbine_ids].mean(axis=1)
                elif freestream_power_method == "median":
                    self.aggregate_df_sample.loc[wd_bin_flag, "power_mean_freestream"] = (
                        self.aggregate_df_sample.loc[wd_bin_flag, "power_normal"]
                    )[freestream_turbine_ids].median(axis=1)
                elif freestream_power_method == "max":
                    self.aggregate_df_sample.loc[wd_bin_flag, "power_mean_freestream"] = (
                        self.aggregate_df_sample.loc[wd_bin_flag, "power_normal"]
                    )[freestream_turbine_ids].max(axis=1)

                if freestream_wind_speed_method == "mean":
                    self.aggregate_df_sample.loc[wd_bin_flag, "windspeed_mean_freestream"] = (
                        self.aggregate_df_sample.loc[wd_bin_flag, "windspeed_normal"]
                    )[freestream_turbine_ids].mean(axis=1)
                elif freestream_wind_speed_method == "median":
                    self.aggregate_df_sample.loc[wd_bin_flag, "windspeed_mean_freestream"] = (
                        self.aggregate_df_sample.loc[wd_bin_flag, "windspeed_normal"]
                    )[freestream_turbine_ids].median(axis=1)

            # Remove rows where no freestream turbines in normal operation were identified
            self.aggregate_df_sample = self.aggregate_df_sample.dropna(
                subset=[("power_mean_freestream", ""), ("windspeed_mean_freestream", "")]
            )

            # calculate total plant-level wake losses during period of record

            # Determine ideal wind plant energy, correcting for derated turbines if correct_for_derating is True. If
            # correct_for_derating is True, ideal energy is calculated as the sum of the power produced by derated
            # turbines and the mean power produced by freestream turbines operating normally multiplied by the total
            # number of turbines operating normally
            total_derated_turbine_power = (
                self.aggregate_df_sample["power"] * self.aggregate_df_sample["derate_flag"]
            ).sum(axis=1)

            total_potential_freestream_power = self.aggregate_df_sample["power_mean_freestream"] * (
                ~self.aggregate_df_sample["derate_flag"]
            ).sum(axis=1)

            # Assign total potential power
            self.aggregate_df_sample["potential_plant_power"] = (
                total_potential_freestream_power + total_derated_turbine_power
            )

            # Assign actual total power produced by wind plant
            self.aggregate_df_sample["actual_plant_power"] = self.aggregate_df_sample["power"].sum(
                axis=1
            )

            wake_losses_por = (
                1
                - self.aggregate_df_sample["actual_plant_power"].sum()
                / self.aggregate_df_sample["potential_plant_power"].sum()
            )

            # bin wake losses by wind direction
            # group wind farm efficiency by wind direction bin
            self.aggregate_df_sample["wind_direction_bin"] = (
                self._wd_bin_width_LT_corr
                * (
                    self.aggregate_df_sample["wind_direction_ref"] / self._wd_bin_width_LT_corr
                ).round()
            )
            self.aggregate_df_sample.loc[
                self.aggregate_df_sample["wind_direction_bin"] == 360.0, "wind_direction_bin"
            ] = 0.0

            # calculate turbine-level wake losses during period of record
            turbine_wake_losses_por = len(self.turbine_ids) * [0.0]
            for i, t in enumerate(self.turbine_ids):
                # determine ideal turbine energy as sum of the power produced by the turbine when it is derated and the
                # mean power produced by all freestream turbines when the turbine is operating normally

                self.aggregate_df_sample.loc[
                    ~self.aggregate_df_sample[("derate_flag", t)], ("potential_turbine_power", t)
                ] = self.aggregate_df_sample.loc[
                    ~self.aggregate_df_sample[("derate_flag", t)], "power_mean_freestream"
                ]

                self.aggregate_df_sample.loc[
                    self.aggregate_df_sample[("derate_flag", t)], ("potential_turbine_power", t)
                ] = self.aggregate_df_sample.loc[
                    self.aggregate_df_sample[("derate_flag", t)], ("power", t)
                ]

                turbine_wake_losses_por[i] = (
                    1
                    - self.aggregate_df_sample[("power", t)].sum()
                    / self.aggregate_df_sample[("potential_turbine_power", t)].sum()
                )

            df_wd_bin = self.aggregate_df_sample.groupby("wind_direction_bin").sum()

            # Save plant and turbine-level wake losses binned by wind direction
            wake_losses_por_wd = (
                df_wd_bin["actual_plant_power"] / df_wd_bin["potential_plant_power"]
            ).values

            turbine_wake_losses_por_wd = np.empty(
                [len(self.turbine_ids), int(360.0 / self._wd_bin_width_LT_corr)]
            )
            for i, t in enumerate(self.turbine_ids):
                turbine_wake_losses_por_wd[i, :] = (
                    df_wd_bin[("power", t)] / df_wd_bin[("potential_turbine_power", t)]
                ).values

            if self.UQ:
                self.wake_losses_por[n] = wake_losses_por
                self.turbine_wake_losses_por[n, :] = turbine_wake_losses_por
                self.wake_losses_por_wd[n, :] = wake_losses_por_wd
                self.turbine_wake_losses_por_wd[n, :, :] = turbine_wake_losses_por_wd
                self.energy_por_wd[n, :] = (
                    df_wd_bin["actual_plant_power"].values / df_wd_bin["actual_plant_power"].sum()
                )

                # apply long-term correction to wake losses
                (
                    wake_losses_lt,
                    turbine_wake_losses_lt,
                    wake_losses_lt_wd,
                    turbine_wake_losses_lt_wd,
                    energy_lt_wd,
                    wake_losses_por_ws,
                    turbine_wake_losses_por_ws,
                    energy_por_ws,
                    wake_losses_lt_ws,
                    turbine_wake_losses_lt_ws,
                    energy_lt_ws,
                ) = self._apply_LT_correction()

                self.wake_losses_lt[n] = wake_losses_lt
                self.turbine_wake_losses_lt[n, :] = turbine_wake_losses_lt
                self.wake_losses_lt_wd[n, :] = wake_losses_lt_wd
                self.turbine_wake_losses_lt_wd[n, :, :] = turbine_wake_losses_lt_wd
                self.energy_lt_wd[n, :] = energy_lt_wd
                self.wake_losses_por_ws[n, :] = wake_losses_por_ws
                self.turbine_wake_losses_por_ws[n, :, :] = turbine_wake_losses_por_ws
                self.energy_por_ws[n, :] = energy_por_ws
                self.wake_losses_lt_ws[n, :] = wake_losses_lt_ws
                self.turbine_wake_losses_lt_ws[n, :, :] = turbine_wake_losses_lt_ws
                self.energy_lt_ws[n, :] = energy_lt_ws

        if not self.UQ:
            # apply long-term correction to wake losses and average results over all reanalysis products
            self.wake_losses_por = wake_losses_por
            self.turbine_wake_losses_por = turbine_wake_losses_por
            self.wake_losses_por_wd = wake_losses_por_wd
            self.turbine_wake_losses_por_wd = turbine_wake_losses_por_wd
            self.energy_por_wd = (
                df_wd_bin["actual_plant_power"].values / df_wd_bin["actual_plant_power"].sum()
            )

            wake_losses_lt_all_products = np.empty([len(self.reanal_products), 1])
            turbine_wake_losses_lt_all_products = np.empty(
                [len(self.reanal_products), len(self.turbine_ids)]
            )

            wake_losses_lt_wd_all_products = np.empty(
                [len(self.reanal_products), int(360.0 / self._wd_bin_width_LT_corr)]
            )
            turbine_wake_losses_lt_wd_all_products = np.empty(
                [
                    len(self.reanal_products),
                    len(self.turbine_ids),
                    int(360.0 / self._wd_bin_width_LT_corr),
                ]
            )
            energy_lt_wd_all_products = np.empty(
                [len(self.reanal_products), int(360.0 / self._wd_bin_width_LT_corr)]
            )

            wake_losses_por_ws_all_products = np.empty(
                [len(self.reanal_products), int(30.0 / self._ws_bin_width_LT_corr) + 1]
            )
            turbine_wake_losses_por_ws_all_products = np.empty(
                [
                    len(self.reanal_products),
                    len(self.turbine_ids),
                    int(30.0 / self._ws_bin_width_LT_corr) + 1,
                ]
            )
            energy_por_ws_all_products = np.empty(
                [len(self.reanal_products), int(30.0 / self._ws_bin_width_LT_corr) + 1]
            )

            wake_losses_lt_ws_all_products = np.empty(
                [len(self.reanal_products), int(30.0 / self._ws_bin_width_LT_corr) + 1]
            )
            turbine_wake_losses_lt_ws_all_products = np.empty(
                [
                    len(self.reanal_products),
                    len(self.turbine_ids),
                    int(30.0 / self._ws_bin_width_LT_corr) + 1,
                ]
            )
            energy_lt_ws_all_products = np.empty(
                [len(self.reanal_products), int(30.0 / self._ws_bin_width_LT_corr) + 1]
            )

            for i_rean, product in enumerate(self.reanal_products):
                self._run.reanalysis_product = product

                (
                    wake_losses_lt,
                    turbine_wake_losses_lt,
                    wake_losses_lt_wd,
                    turbine_wake_losses_lt_wd,
                    energy_lt_wd,
                    wake_losses_por_ws,
                    turbine_wake_losses_por_ws,
                    energy_por_ws,
                    wake_losses_lt_ws,
                    turbine_wake_losses_lt_ws,
                    energy_lt_ws,
                ) = self._apply_LT_correction()

                wake_losses_lt_all_products[i_rean] = wake_losses_lt
                turbine_wake_losses_lt_all_products[i_rean] = turbine_wake_losses_lt

                wake_losses_lt_wd_all_products[i_rean] = wake_losses_lt_wd
                turbine_wake_losses_lt_wd_all_products[i_rean] = turbine_wake_losses_lt_wd

                energy_lt_wd_all_products[i_rean] = energy_lt_wd

                wake_losses_por_ws_all_products[i_rean] = wake_losses_por_ws
                turbine_wake_losses_por_ws_all_products[i_rean] = turbine_wake_losses_por_ws

                energy_por_ws_all_products[i_rean] = energy_por_ws

                wake_losses_lt_ws_all_products[i_rean] = wake_losses_lt_ws
                turbine_wake_losses_lt_ws_all_products[i_rean] = turbine_wake_losses_lt_ws

                energy_lt_ws_all_products[i_rean] = energy_lt_ws

            self.wake_losses_lt = np.mean(wake_losses_lt_all_products)
            self.turbine_wake_losses_lt = np.mean(turbine_wake_losses_lt_all_products, axis=0)

            self.wake_losses_lt_wd = np.mean(wake_losses_lt_wd_all_products, axis=0)
            self.turbine_wake_losses_lt_wd = np.mean(turbine_wake_losses_lt_wd_all_products, axis=0)
            self.energy_lt_wd = np.mean(energy_lt_wd_all_products, axis=0)

            self.wake_losses_por_ws = np.mean(wake_losses_por_ws_all_products, axis=0)
            self.turbine_wake_losses_por_ws = np.mean(
                turbine_wake_losses_por_ws_all_products, axis=0
            )
            self.energy_por_ws = np.mean(energy_por_ws_all_products, axis=0)

            self.wake_losses_lt_ws = np.mean(wake_losses_lt_ws_all_products, axis=0)
            self.turbine_wake_losses_lt_ws = np.mean(turbine_wake_losses_lt_ws_all_products, axis=0)
            self.energy_lt_ws = np.mean(energy_lt_ws_all_products, axis=0)

        else:
            # Calculate mean and standard deviation of wake losses from Monte Carlo simulations
            self.wake_losses_lt_mean = np.mean(self.wake_losses_lt)
            self.turbine_wake_losses_lt_mean = np.mean(self.turbine_wake_losses_lt, axis=0)
            self.wake_losses_por_mean = np.mean(self.wake_losses_por)
            self.turbine_wake_losses_por_mean = np.mean(self.turbine_wake_losses_por, axis=0)

            self.wake_losses_lt_std = np.std(self.wake_losses_lt)
            self.turbine_wake_losses_lt_std = np.std(self.turbine_wake_losses_lt, axis=0)
            self.wake_losses_por_std = np.std(self.wake_losses_por)
            self.turbine_wake_losses_por_std = np.std(self.turbine_wake_losses_por, axis=0)

    @logged_method_call
    def _setup_monte_carlo_inputs(self):
        """
        Create and populate the data frame defining the Monte Carlo simulation parameters. This data frame is stored as
        self.inputs.

        Args:
            (None)

        Returns:
            (None)
        """

        if self.UQ:
            inputs = {
                "reanalysis_product": random.choices(self.reanal_products, k=self._num_sim),
                "freestream_sector_width": np.random.randint(
                    self._freestream_sector_width[0],
                    self._freestream_sector_width[1] + 1,
                    self._num_sim,
                ),
                "wind_bin_mad_thresh": np.random.randint(
                    self._wind_bin_mad_thresh[0], self._wind_bin_mad_thresh[1] + 1, self._num_sim
                ),
                "derating_filter_wind_speed_start": np.random.randint(
                    self._derating_filter_wind_speed_start[0] * 10,
                    self._derating_filter_wind_speed_start[1] * 10 + 1,
                    self._num_sim,
                )
                / 10.0,
                "max_power_filter": np.random.randint(
                    self._max_power_filter[0] * 100,
                    self._max_power_filter[1] * 100 + 1,
                    self._num_sim,
                )
                / 100.0,
                "num_years_LT": np.random.randint(
                    self._num_years_LT[0], self._num_years_LT[1] + 1, self._num_sim
                ),
            }
            self.inputs = pd.DataFrame(inputs)

            self.wake_losses_por = np.empty([self._num_sim, 1])
            self.turbine_wake_losses_por = np.empty([self._num_sim, len(self.turbine_ids)])
            self.wake_losses_lt = np.empty([self._num_sim, 1])
            self.turbine_wake_losses_lt = np.empty([self._num_sim, len(self.turbine_ids)])

            # For saving wake losses and energy production binned by wind direction
            self.wake_losses_por_wd = np.empty(
                [self._num_sim, int(360.0 / self._wd_bin_width_LT_corr)]
            )
            self.turbine_wake_losses_por_wd = np.empty(
                [self._num_sim, len(self.turbine_ids), int(360.0 / self._wd_bin_width_LT_corr)]
            )
            self.wake_losses_lt_wd = np.empty(
                [self._num_sim, int(360.0 / self._wd_bin_width_LT_corr)]
            )
            self.turbine_wake_losses_lt_wd = np.empty(
                [self._num_sim, len(self.turbine_ids), int(360.0 / self._wd_bin_width_LT_corr)]
            )

            self.energy_por_wd = np.empty([self._num_sim, int(360.0 / self._wd_bin_width_LT_corr)])
            self.energy_lt_wd = np.empty([self._num_sim, int(360.0 / self._wd_bin_width_LT_corr)])

            # For saving wake losses and energy production binned by wind speed
            self.wake_losses_por_ws = np.empty(
                [self._num_sim, int(30.0 / self._ws_bin_width_LT_corr) + 1]
            )
            self.turbine_wake_losses_por_ws = np.empty(
                [self._num_sim, len(self.turbine_ids), int(30.0 / self._ws_bin_width_LT_corr) + 1]
            )
            self.wake_losses_lt_ws = np.empty(
                [self._num_sim, int(30.0 / self._ws_bin_width_LT_corr) + 1]
            )
            self.turbine_wake_losses_lt_ws = np.empty(
                [self._num_sim, len(self.turbine_ids), int(30.0 / self._ws_bin_width_LT_corr) + 1]
            )

            self.energy_por_ws = np.empty(
                [self._num_sim, int(30.0 / self._ws_bin_width_LT_corr) + 1]
            )
            self.energy_lt_ws = np.empty(
                [self._num_sim, int(30.0 / self._ws_bin_width_LT_corr) + 1]
            )

        elif not self.UQ:
            inputs = {
                "reanalysis_product": self.reanal_products,
                "freestream_sector_width": len(self.reanal_products)
                * [self._freestream_sector_width],
                "wind_bin_mad_thresh": len(self.reanal_products) * [self._wind_bin_mad_thresh],
                "derating_filter_wind_speed_start": len(self.reanal_products)
                * [self._derating_filter_wind_speed_start],
                "max_power_filter": len(self.reanal_products) * [self._max_power_filter],
                "num_years_LT": len(self.reanal_products) * [self._num_years_LT],
            }
            self.inputs = pd.DataFrame(inputs)

            self._num_sim = 1

    @logged_method_call
    def _calculate_aggregate_dataframe(self):
        """
        Creates a data frame with relevant scada columns, plant-level columns, and reanalysis variables to be used for
        the wake loss analysis. The reference mean wind direction is then added to the data frame.

        Args:
            (None)

        Returns:
            (None)
        """

        # keep relevant SCADA columns, create a unique time index and two-level turbine variable columns
        # (variable name and turbine ID)
        valid_times = (self.plant.scada.index.get_level_values("time") >= self.start_date) & (
            self.plant.scada.index.get_level_values("time") <= self.end_date
        )

        # include scada wind direction column only if using scada to determine mean wind direction for wind plant
        if self.wind_direction_data_type == "scada":
            scada_cols = ["windspeed", self.wind_direction_col, "power"]
        else:
            scada_cols = ["windspeed", "power"]

        self.aggregate_df = self.plant.scada.loc[valid_times, scada_cols].unstack()

        # Calculate reference mean wind direction
        self._calculate_mean_wind_direction()

        # Add reanalysis data to aggregate data frame
        self._include_reanal_data()

        # remove times with any missing data
        # TODO: revisit because this may remove too many samples
        self.aggregate_df = self.aggregate_df.dropna(how="any")

        # Drop turbine-level wind direction column
        if self.wind_direction_data_type == "scada":
            self.aggregate_df = self.aggregate_df.drop(columns=[self.wind_direction_col])

    @logged_method_call
    def _calculate_mean_wind_direction(self):
        """
        Calculates the mean wind direction at each time step using the specified wind direction column for the
        specified subset of turbines or met towers. This reference mean wind direction is added to the plant-level data
        frame.

        Args:
            (None)
        Returns:
            (None)
        """

        def circular_average(x):
            return (
                np.degrees(
                    np.arctan2(
                        np.sin(np.radians(x)).mean(axis=1),
                        np.cos(np.radians(x)).mean(axis=1),
                    )
                )
                % 360.0
            )

        if self.wind_direction_data_type == "scada":
            self.aggregate_df["wind_direction_ref"] = circular_average(
                self.aggregate_df[self.wind_direction_col][self.wind_direction_asset_ids]
            )
        elif self.wind_direction_data_type == "tower":
            df_tower = self.plant.tower[[self.wind_direction_col]].unstack()

            self.aggregate_df["wind_direction_ref"] = circular_average(
                df_tower[self.wind_direction_col][self.wind_direction_asset_ids]
            )

    @logged_method_call
    def _include_reanal_data(self):
        """
        Combines reanalysis data columns with the aggregate data frame for use in long-term correction.

        Args:
            (None)
        Returns:
            (None)
        """

        # combine all wind speed and wind direction reanalysis variables into aggregate data frame

        for product in self.reanal_products:

            df_rean = self.plant.reanalysis[product][["windspeed", "wind_direction"]].copy()

            # Drop minute field
            df_rean.index = df_rean.index.floor("H")

            # Upsample to 10-minute samples to match SCADA data
            df_rean = df_rean.resample("10T").ffill()
            df_rean = df_rean.add_suffix(f"_{product}")
            df_rean = df_rean[df_rean.index.isin(self.aggregate_df.index)]

            self.aggregate_df[[col for col in df_rean.columns]] = df_rean

    @logged_method_call
    def _identify_derating(self):
        """
        Estimates whether each turbine is derated, curtailed, or otherwise not operating for each time stamp based on
        power curve filtering. A derated flag is then added to the aggregate data frame for each turbine.

        Args:
            (None)

        Returns:
            (None)
        """

        for t in self.turbine_ids:
            # Apply window range filter to flag samples for which wind speed is greater than a threshold and power is
            # below 1% of rated power

            turb_capac = self.plant.asset.loc[t, "rated_power"]

            flag_window = filters.window_range_flag(
                window_col=self.aggregate_df[("windspeed", t)],
                window_start=self._run.derating_filter_wind_speed_start,
                window_end=40,
                value_col=self.aggregate_df[("power", t)],
                value_min=0.01 * turb_capac,
                value_max=1.2 * turb_capac,
            )

            # Apply bin-based filter to flag samples for which wind speed is greater than a threshold from the median
            # wind speed in each power bin
            bin_width_frac = 0.04 * (
                self._run.max_power_filter - 0.01
            )  # split into 25 bins TODO: make this an optional argument?
            flag_bin = filters.bin_filter(
                bin_col=self.aggregate_df[("power", t)],
                value_col=self.aggregate_df[("windspeed", t)],
                bin_width=bin_width_frac * turb_capac,
                threshold=self._run.wind_bin_mad_thresh,  # wind bin thresh
                center_type="median",
                bin_min=0.01 * turb_capac,
                bin_max=self._run.max_power_filter * turb_capac,
                threshold_type="mad",
                direction="above",
            )

            self.aggregate_df[("derate_flag", t)] = flag_window | flag_bin

    @logged_method_call
    def _apply_LT_correction(self):
        """
        Estimates long term-corrected wake losses by binning wake losses by wind direction and wind speed and weighting
        by bin frequencies from long-term historical reanalysis data.

        Args:
            (None)

        Returns:
            tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The estimated long term-corrected wake
                losses, an array containing the estimated turbine-level long term-corrected wake losses, and arrays containing the long-term corrected plant and turbine-level wake losses as well as the normalized wind plant energy productionbinned by wind direction
        """

        # TODO: make arguments?
        # The minimum wind speed bin to consider when finding linear regression from SCADA freestream wind speeds to
        # reanalysis wind speeds
        min_ws_bin_lin_reg = 3.0
        # The minimum number of samples required in a wind speed bin to include when finding linear regression from
        # SCADA freestream wind speeds to reanalysis wind speeds
        bin_count_thresh_lin_reg = 50

        # First, create hourly data frame for LT correction to match resolution of reanalysis data
        df_1hr = self.aggregate_df_sample[
            [
                ("wind_direction_ref", ""),
                ("windspeed_mean_freestream", ""),
                ("actual_plant_power", ""),
                ("potential_plant_power", ""),
            ]
            + [("power", t) for t in self.turbine_ids]
            + [("potential_turbine_power", t) for t in self.turbine_ids]
            + [(f"windspeed_{self._run.reanalysis_product}", "")]
        ].copy()

        df_1hr = df_1hr.resample("H").mean().dropna(how="any")

        df_1hr["windspeed_mean_freestream_bin"] = df_1hr["windspeed_mean_freestream"].round()

        # Bin by integer wind speeds
        df_ws_bin = df_1hr.groupby(("windspeed_mean_freestream_bin", "")).mean()
        df_ws_bin_count = df_1hr.groupby(("windspeed_mean_freestream_bin", "")).count()

        valid_ws_bins = (df_ws_bin.index >= min_ws_bin_lin_reg) & (
            df_ws_bin_count["windspeed_mean_freestream"] >= bin_count_thresh_lin_reg
        )

        # Find linear regression mapping from SCADA freestream wind speed to reanalysis wind speeds and use to correct
        # SCADA freestream wind speeds
        reg = LinearRegression().fit(
            df_ws_bin.loc[valid_ws_bins].index.values.reshape(-1, 1),
            df_ws_bin.loc[valid_ws_bins, f"windspeed_{self._run.reanalysis_product}"].values,
        )

        df_1hr[f"windspeed_mean_freestream_corr_{self._run.reanalysis_product}"] = reg.predict(
            df_1hr["windspeed_mean_freestream"].values.reshape(-1, 1)
        )

        # adjust the _no_wakes_ws_thresh_LT_corr parameter to relect the SCADA wind speed correction as well
        no_wakes_ws_corr_thresh_LT_corr = np.round(
            reg.predict(np.array(self._no_wakes_ws_thresh_LT_corr).reshape(1, -1))[0]
        )

        # Create data frame with long-term frequencies of wind direction and wind speed bins from reanalysis data
        df_reanal_freqs = pd.DataFrame()

        # get reanalysis data and limit date range
        df_reanal = self.plant.reanalysis[self._run.reanalysis_product].copy()
        df_reanal = df_reanal.loc[
            (df_reanal.index <= self.end_date_lt)
            & (
                df_reanal.index
                > self.end_date_lt - pd.offsets.DateOffset(years=self._run.num_years_LT)
            )
        ]
        df_reanal["windspeed_bin"] = (
            self._ws_bin_width_LT_corr
            * (df_reanal["windspeed"] / self._ws_bin_width_LT_corr).round()
        )
        df_reanal["wind_direction_bin"] = (
            self._wd_bin_width_LT_corr
            * (df_reanal["wind_direction"] / self._wd_bin_width_LT_corr).round()
        )
        df_reanal.loc[df_reanal["wind_direction_bin"] == 360.0, "wind_direction_bin"] = 0.0

        df_reanal["freq"] = 1.0
        df_reanal = df_reanal.groupby(["wind_direction_bin", "windspeed_bin"]).count()["freq"]

        df_reanal_freqs = pd.DataFrame(df_reanal / df_reanal.sum())

        # Weight wake losses in each wind direction and wind speed bin by long-term frequencies to estimate long-term
        # wake losses
        df_1hr["windspeed_bin"] = (
            self._ws_bin_width_LT_corr
            * (
                df_1hr[f"windspeed_mean_freestream_corr_{self._run.reanalysis_product}"]
                / self._ws_bin_width_LT_corr
            ).round()
        )
        df_1hr["wind_direction_bin"] = (
            self._wd_bin_width_LT_corr
            * (df_1hr["wind_direction_ref"] / self._wd_bin_width_LT_corr).round()
        )
        df_1hr.loc[df_1hr["wind_direction_bin"] == 360.0, "wind_direction_bin"] = 0.0

        # First, compute POR wake losses as a function of wind speed
        df_1hr_ws_por_bin = df_1hr.groupby(("windspeed_bin", "")).sum()

        # reindex to fill in missing wind speed bins
        index = np.arange(0.0, 31.0, self._ws_bin_width_LT_corr).tolist()
        df_1hr_ws_por_bin = df_1hr_ws_por_bin.reindex(index)

        wake_losses_por_ws = (
            df_1hr_ws_por_bin[("actual_plant_power", "")]
            / df_1hr_ws_por_bin[("potential_plant_power", "")]
        ).values

        energy_por_ws = (
            df_1hr_ws_por_bin[("actual_plant_power", "")].values
            / df_1hr_ws_por_bin[("actual_plant_power", "")].sum()
        )

        turbine_wake_losses_por_ws = np.empty(
            [len(self.turbine_ids), int(30.0 / self._ws_bin_width_LT_corr) + 1]
        )
        for i, t in enumerate(self.turbine_ids):
            turbine_wake_losses_por_ws[i, :] = (
                df_1hr_ws_por_bin[("power", t)] / df_1hr_ws_por_bin[("potential_turbine_power", t)]
            ).values

        # Bin variables by wind direction and wind speed
        df_1hr_bin = df_1hr.groupby([("wind_direction_bin", ""), ("windspeed_bin", "")]).mean()

        df_1hr_bin = pd.concat([df_reanal_freqs, df_1hr_bin], axis=1)

        # If specified, assume no wake losses at wind speeds above a given threshold for bins where data are
        # missing by assigning rated power to the actual and potential power production
        if self._assume_no_wakes_high_ws_LT_corr:
            fill_inds = (df_1hr_bin[("actual_plant_power", "")].isna()) & (
                df_1hr_bin.index.get_level_values(1) >= no_wakes_ws_corr_thresh_LT_corr
            )
            df_1hr_bin.loc[
                fill_inds, [("actual_plant_power", ""), ("potential_plant_power", "")]
            ] = (self.plant.metadata.capacity * 1e3)
            df_1hr_bin.loc[
                fill_inds,
                [("power", t) for t in self.turbine_ids]
                + [("potential_turbine_power", t) for t in self.turbine_ids],
            ] = 2 * [self.plant.asset.loc[t, "rated_power"] for t in self.turbine_ids]

        df_1hr_bin["actual_plant_energy"] = (
            df_1hr_bin["freq"] * df_1hr_bin[("actual_plant_power", "")]
        )
        df_1hr_bin["potential_plant_energy"] = (
            df_1hr_bin["freq"] * df_1hr_bin[("potential_plant_power", "")]
        )

        wake_losses_lt = 1 - (
            df_1hr_bin["actual_plant_energy"].sum() / df_1hr_bin["potential_plant_energy"].sum()
        )

        # Calculate long-term corrected turbine-level wake losses
        turbine_wake_losses_lt = len(self.turbine_ids) * [0.0]
        for i, t in enumerate(self.turbine_ids):
            # determine ideal turbine energy as sum of the power produced by the turbine when it is derated and the
            # mean power produced by all freestream turbines when the turbine is operating normally

            df_1hr_bin[("energy_avg", t)] = df_1hr_bin["freq"] * df_1hr_bin[("power", t)]
            df_1hr_bin[("potential_turbine_energy", t)] = (
                df_1hr_bin["freq"] * df_1hr_bin[("potential_turbine_power", t)]
            )

            turbine_wake_losses_lt[i] = 1 - (
                df_1hr_bin[("energy_avg", t)].sum()
                / df_1hr_bin[("potential_turbine_energy", t)].sum()
            )

        # Save long-term corrected plant and turbine-level wake losses binned by wind direction
        df_1hr_wd_bin = df_1hr_bin.groupby(level=[0]).sum()

        wake_losses_lt_wd = (
            df_1hr_wd_bin["actual_plant_energy"] / df_1hr_wd_bin["potential_plant_energy"]
        ).values

        energy_lt_wd = (
            df_1hr_wd_bin["actual_plant_energy"].values / df_1hr_wd_bin["actual_plant_energy"].sum()
        )

        turbine_wake_losses_lt_wd = np.empty(
            [len(self.turbine_ids), int(360.0 / self._wd_bin_width_LT_corr)]
        )
        for i, t in enumerate(self.turbine_ids):
            turbine_wake_losses_lt_wd[i, :] = (
                df_1hr_wd_bin[("energy_avg", t)] / df_1hr_wd_bin[("potential_turbine_energy", t)]
            ).values

        # Save long-term corrected plant and turbine-level wake losses binned by wind speed
        df_1hr_ws_bin = df_1hr_bin.groupby(level=[1]).sum()

        # reindex to fill in missing wind speed bins
        index = np.arange(0.0, 31.0, self._ws_bin_width_LT_corr).tolist()
        df_1hr_ws_bin = df_1hr_ws_bin.reindex(index)

        wake_losses_lt_ws = (
            df_1hr_ws_bin["actual_plant_energy"] / df_1hr_ws_bin["potential_plant_energy"]
        ).values

        energy_lt_ws = (
            df_1hr_ws_bin["actual_plant_energy"].values / df_1hr_ws_bin["actual_plant_energy"].sum()
        )

        turbine_wake_losses_lt_ws = np.empty(
            [len(self.turbine_ids), int(30.0 / self._ws_bin_width_LT_corr) + 1]
        )
        for i, t in enumerate(self.turbine_ids):
            turbine_wake_losses_lt_ws[i, :] = (
                df_1hr_ws_bin[("energy_avg", t)] / df_1hr_ws_bin[("potential_turbine_energy", t)]
            ).values

        return (
            wake_losses_lt,
            turbine_wake_losses_lt,
            wake_losses_lt_wd,
            turbine_wake_losses_lt_wd,
            energy_lt_wd,
            wake_losses_por_ws,
            turbine_wake_losses_por_ws,
            energy_por_ws,
            wake_losses_lt_ws,
            turbine_wake_losses_lt_ws,
            energy_lt_ws,
        )

    def plot_wake_losses_by_wind_direction(
        self, plot_norm_energy: bool = True, turbine_id: str = None
    ):
        """
        Plots wake losses during the period of record in the form of wind farm efficiency as a function of wind
        direction as well as normalized wind plant energy production as a function of wind direction.

        Args:
            plot_norm_energy (bool, optional): If True, include a plot of normalized wind plant energy
                production as a function of wind direction in addition to the wind farm efficiency plot. Defaults to
                True.
            turbine_id (str, optional): Turbine ID to plot wake losses for. If None, wake losses for the
                entire wind plant will be plotted. Defaults to None.
        Returns:
            matplotlib.pyplot.axes: An axes object or array of two axes corresponding to the wake loss plot or
                wake loss and normalized energy plots
        """

        import matplotlib.pyplot as plt

        color_codes = ["#4477AA", "#228833"]

        wd_bins = np.arange(0.0, 360.0, self._wd_bin_width_LT_corr)

        if plot_norm_energy:
            _, axs = plt.subplots(2, 1, figsize=(9, 9.1), sharex=True)
        else:
            _, ax = plt.subplots(figsize=(9, 5))
            axs = [ax]
        axs[0].plot([0, 360.0 - self._wd_bin_width_LT_corr], [1, 1], "k", linewidth=1.5)

        if self.UQ:
            if turbine_id is None:  # plot wind plant wake losses
                axs[0].plot(
                    wd_bins,
                    np.mean(self.wake_losses_por_wd, axis=0),
                    color=color_codes[0],
                    label="Period of Record",
                )
                axs[0].fill_between(
                    wd_bins,
                    np.percentile(self.wake_losses_por_wd, 2.5, axis=0),
                    np.percentile(self.wake_losses_por_wd, 97.5, axis=0),
                    alpha=0.2,
                    color=color_codes[0],
                    label="_nolegend_",
                )

                axs[0].plot(
                    wd_bins,
                    np.mean(self.wake_losses_lt_wd, axis=0),
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )
                axs[0].fill_between(
                    wd_bins,
                    np.percentile(self.wake_losses_lt_wd, 2.5, axis=0),
                    np.percentile(self.wake_losses_lt_wd, 97.5, axis=0),
                    alpha=0.2,
                    color=color_codes[1],
                    label="_nolegend_",
                )
            else:  # plot wake losses for specific turbine
                turbine_index = self.turbine_ids.index(turbine_id)

                axs[0].plot(
                    wd_bins,
                    np.mean(self.turbine_wake_losses_por_wd[:, turbine_index, :], axis=0),
                    color=color_codes[0],
                    label="Period of Record",
                )
                axs[0].fill_between(
                    wd_bins,
                    np.percentile(
                        self.turbine_wake_losses_por_wd[:, turbine_index, :], 2.5, axis=0
                    ),
                    np.percentile(
                        self.turbine_wake_losses_por_wd[:, turbine_index, :], 97.5, axis=0
                    ),
                    alpha=0.2,
                    color=color_codes[0],
                    label="_nolegend_",
                )

                axs[0].plot(
                    wd_bins,
                    np.mean(self.turbine_wake_losses_lt_wd[:, turbine_index, :], axis=0),
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )
                axs[0].fill_between(
                    wd_bins,
                    np.percentile(self.turbine_wake_losses_lt_wd[:, turbine_index, :], 2.5, axis=0),
                    np.percentile(
                        self.turbine_wake_losses_lt_wd[:, turbine_index, :], 97.5, axis=0
                    ),
                    alpha=0.2,
                    color=color_codes[1],
                    label="_nolegend_",
                )

                axs[0].set_title(f"Wind Turbine {turbine_id}")

            if plot_norm_energy:
                axs[1].plot(
                    wd_bins,
                    np.mean(self.energy_por_wd, axis=0),
                    color=color_codes[0],
                    label="Period of Record",
                )
                axs[1].fill_between(
                    wd_bins,
                    np.percentile(self.energy_por_wd, 2.5, axis=0),
                    np.percentile(self.energy_por_wd, 97.5, axis=0),
                    alpha=0.2,
                    color=color_codes[0],
                    label="_nolegend_",
                )

                axs[1].plot(
                    wd_bins,
                    np.mean(self.energy_lt_wd, axis=0),
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )
                axs[1].fill_between(
                    wd_bins,
                    np.percentile(self.energy_lt_wd, 2.5, axis=0),
                    np.percentile(self.energy_lt_wd, 97.5, axis=0),
                    alpha=0.2,
                    color=color_codes[1],
                    label="_nolegend_",
                )

        else:  # without UQ
            if turbine_id is None:  # plot wind plant wake losses
                axs[0].plot(
                    wd_bins, self.wake_losses_por_wd, color=color_codes[0], label="Period of Record"
                )

                axs[0].plot(
                    wd_bins,
                    self.wake_losses_lt_wd,
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )

            else:  # plot wake losses for specific turbine
                turbine_index = self.turbine_ids.index(turbine_id)

                axs[0].plot(
                    wd_bins,
                    self.turbine_wake_losses_por_wd[turbine_index, :],
                    color=color_codes[0],
                    label="Period of Record",
                )

                axs[0].plot(
                    wd_bins,
                    self.turbine_wake_losses_lt_wd[turbine_index, :],
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )

                axs[0].set_title(f"Wind Turbine {turbine_id}")

            if plot_norm_energy:
                axs[1].plot(
                    wd_bins, self.energy_por_wd, color=color_codes[0], label="Period of Record"
                )

                axs[1].plot(
                    wd_bins,
                    self.energy_lt_wd,
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )

        axs[0].set_xlim([0, 360.0 - self._wd_bin_width_LT_corr])
        axs[len(axs) - 1].set_xlabel(r"Wind Direction ($^\circ$)")
        axs[0].legend()
        axs[0].set_ylabel("Wind Plant Efficiency (-)")

        if plot_norm_energy:
            axs[1].legend()
            axs[1].set_ylabel("Normalized Wind Plant\nEnergy Production (-)")

            plt.tight_layout()

            return axs
        else:
            return ax

    def plot_wake_losses_by_wind_speed(self, plot_norm_energy: bool = True, turbine_id: str = None):
        """
        Plots wake losses during the period of record in the form of wind farm efficiency as a function of wind
        speed as well as normalized wind plant energy production as a function of wind speee.

        Args:
            plot_norm_energy (bool, optional): If True, include a plot of normalized wind plant energy
                production as a function of wind speed in addition to the wind farm efficiency plot. Defaults to
                True.
            turbine_id (str, optional): Turbine ID to plot wake losses for. If None, wake losses for the
                entire wind plant will be plotted. Defaults to None.
        Returns:
            matplotlib.pyplot.axes: An axes object or array of two axes corresponding to the wake loss plot or
                wake loss and normalized energy plots
        """

        import matplotlib.pyplot as plt

        color_codes = ["#4477AA", "#228833"]

        # Limit to 4 - 20 m/s. TODO: Customize this later?
        ws_min = 4.0
        ws_max = 20.0
        ws_bins_orig = np.arange(0.0, 31.0, self._ws_bin_width_LT_corr)
        ws_bins = np.arange(ws_min, ws_max + 1, self._ws_bin_width_LT_corr)
        mask = (ws_bins_orig >= ws_min) & (ws_bins_orig <= ws_max)

        if plot_norm_energy:
            _, axs = plt.subplots(2, 1, figsize=(9, 9.1), sharex=True)
        else:
            _, ax = plt.subplots(figsize=(9, 5))
            axs = [ax]
        axs[0].plot([ws_min, ws_max], [1, 1], "k", linewidth=1.5)

        if self.UQ:
            if turbine_id is None:  # plot wind plant wake losses
                axs[0].plot(
                    ws_bins,
                    np.mean(self.wake_losses_por_ws[:, mask], axis=0),
                    color=color_codes[0],
                    label="Period of Record",
                )
                axs[0].fill_between(
                    ws_bins,
                    np.percentile(self.wake_losses_por_ws[:, mask], 2.5, axis=0),
                    np.percentile(self.wake_losses_por_ws[:, mask], 97.5, axis=0),
                    alpha=0.2,
                    color=color_codes[0],
                    label="_nolegend_",
                )

                axs[0].plot(
                    ws_bins,
                    np.mean(self.wake_losses_lt_ws[:, mask], axis=0),
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )
                axs[0].fill_between(
                    ws_bins,
                    np.percentile(self.wake_losses_lt_ws[:, mask], 2.5, axis=0),
                    np.percentile(self.wake_losses_lt_ws[:, mask], 97.5, axis=0),
                    alpha=0.2,
                    color=color_codes[1],
                    label="_nolegend_",
                )

            else:  # plot wake losses for specific turbine
                turbine_index = self.turbine_ids.index(turbine_id)

                axs[0].plot(
                    ws_bins,
                    np.mean(self.turbine_wake_losses_por_ws[:, turbine_index, mask], axis=0),
                    color=color_codes[0],
                    label="Period of Record",
                )
                axs[0].fill_between(
                    ws_bins,
                    np.percentile(
                        self.turbine_wake_losses_por_ws[:, turbine_index, mask], 2.5, axis=0
                    ),
                    np.percentile(
                        self.turbine_wake_losses_por_ws[:, turbine_index, mask], 97.5, axis=0
                    ),
                    alpha=0.2,
                    color=color_codes[0],
                    label="_nolegend_",
                )

                axs[0].plot(
                    ws_bins,
                    np.mean(self.turbine_wake_losses_lt_ws[:, turbine_index, mask], axis=0),
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )
                axs[0].fill_between(
                    ws_bins,
                    np.percentile(
                        self.turbine_wake_losses_lt_ws[:, turbine_index, mask], 2.5, axis=0
                    ),
                    np.percentile(
                        self.turbine_wake_losses_lt_ws[:, turbine_index, mask], 97.5, axis=0
                    ),
                    alpha=0.2,
                    color=color_codes[1],
                    label="_nolegend_",
                )

                axs[0].set_title(f"Wind Turbine {turbine_id}")

            if plot_norm_energy:
                axs[1].plot(
                    ws_bins,
                    np.mean(self.energy_por_ws[:, mask], axis=0),
                    color=color_codes[0],
                    label="Period of Record",
                )
                axs[1].fill_between(
                    ws_bins,
                    np.percentile(self.energy_por_ws[:, mask], 2.5, axis=0),
                    np.percentile(self.energy_por_ws[:, mask], 97.5, axis=0),
                    alpha=0.2,
                    color=color_codes[0],
                    label="_nolegend_",
                )

                axs[1].plot(
                    ws_bins,
                    np.mean(self.energy_lt_ws[:, mask], axis=0),
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )
                axs[1].fill_between(
                    ws_bins,
                    np.percentile(self.energy_lt_ws[:, mask], 2.5, axis=0),
                    np.percentile(self.energy_lt_ws[:, mask], 97.5, axis=0),
                    alpha=0.2,
                    color=color_codes[1],
                    label="_nolegend_",
                )

        else:  # without UQ
            if turbine_id is None:  # plot wind plant wake losses
                axs[0].plot(
                    ws_bins,
                    self.wake_losses_por_ws[mask],
                    color=color_codes[0],
                    label="Period of Record",
                )

                axs[0].plot(
                    ws_bins,
                    self.wake_losses_lt_ws[mask],
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )

            else:  # plot wake losses for specific turbine
                turbine_index = self.turbine_ids.index(turbine_id)

                axs[0].plot(
                    ws_bins,
                    self.turbine_wake_losses_por_ws[turbine_index, mask],
                    color=color_codes[0],
                    label="Period of Record",
                )

                axs[0].plot(
                    ws_bins,
                    self.turbine_wake_losses_lt_ws[turbine_index, mask],
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )

                axs[0].set_title(f"Wind Turbine {turbine_id}")

            if plot_norm_energy:
                axs[1].plot(
                    ws_bins,
                    self.energy_por_ws[mask],
                    color=color_codes[0],
                    label="Period of Record",
                )

                axs[1].plot(
                    ws_bins,
                    self.energy_lt_ws[mask],
                    color=color_codes[1],
                    label="Long-Term Corrected",
                )

        axs[0].set_xlim([ws_min, ws_max])
        axs[len(axs) - 1].set_xlabel("Freestream Wind Speed (m/s)")
        axs[0].legend()
        axs[0].set_ylabel("Wind Plant Efficiency (-)")

        if plot_norm_energy:
            axs[1].legend()
            axs[1].set_ylabel("Normalized Wind Plant\nEnergy Production (-)")

            plt.tight_layout()

            return axs
        else:
            return ax
