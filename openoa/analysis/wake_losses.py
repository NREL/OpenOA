# This class defines key analytical routines for estimating wake losses for an operating
# wind plant using SCADA data. At a high level, for each SCADA time step, freestream wind
# turbines are identified using the turbine coordinates and a reference wind direction
# signal. The mean power production for all turbines in the wind plant is summed over all
# time steps and compared to the mean power of the freestream turbines summed over all time
# steps to estimate wake losses during the period of record. Methods for calclating the
# long-term wake losses using reanalaysis data and quantifying uncertainty are provided as well.

# The general approach for estimating wake losses and quantifying uncertainty using bootstrapping
# is based in part on the following publications:
# 1. Barthelmie, R. J. and Jensen, L. E. Evaluation of wind farm efficiency and wind turbine wakes
#    at the Nysted offshore wind farm, *Wind Energy* 13(6):573–586 (2010).
#    https://doi.org/10.1002/we.408.
# 2. Nygaard, N. G. Systematic quantification of wake model uncertainty. Proc. EWEA Offshore,
#    Copenhagen, Denmark, March 10-12 (2015).
# 3. Walker, K., Adams, N., Gribben, B., Gellatly, B., Nygaard, N. G., Henderson, A., Marchante
#    Jimémez, M., Schmidt, S. R., Rodriguez Ruiz, J., Paredes, D., Harrington, G., Connell, N.,
#    Peronne, O., Cordoba, M., Housley, P., Cussons, R., Håkansson, M., Knauer, A., and Maguire,
#    E.: An evaluation of the predictive accuracy of wake effects models for offshore wind farms.
#    *Wind Energy* 19(5):979–996 (2016). https://doi.org/10.1002/we.1871.


from __future__ import annotations

import random
from copy import deepcopy

import attrs
import numpy as np
import pandas as pd
import numpy.typing as npt
from tqdm import tqdm
from attrs import field, define
from sklearn.linear_model import LinearRegression

from openoa.plant import PlantData, convert_to_list
from openoa.utils import plot, filters
from openoa.utils import met_data_processing as met
from openoa.schema import FromDictMixin, ResetValuesMixin
from openoa.logging import logging, logged_method_call
from openoa.analysis._analysis_validators import (
    validate_UQ_input,
    validate_half_closed_0_1_right,
    validate_reanalysis_selections,
)


logger = logging.getLogger(__name__)
NDArrayFloat = npt.NDArray[np.float64]
plot.set_styling()


@define(auto_attribs=True)
class WakeLosses(FromDictMixin, ResetValuesMixin):
    """
    A serial implementation of a method for estimating wake losses from SCADA data. Wake losses are
    estimated for the entire wind plant as well as for each individual turbine for a) the period of
    record for which data are available, and b) the estimated long-term wind conditions the wind
    plant will experience based on historical reanalysis wind resource data.

    The method is comprised of the following core steps:
        1. Calculate a representative wind plant-level wind direction at each time step using the
           mean wind direction of the specified wind turbines or meteorological (met) towers.
           Note that time steps for which any necessary plant-level or turbine-level data are
           missing are discarded.

           a. If :py:attr:`UQ` is selected, wake losses are calculated multiple times using a Monte
              Carlo approach with randomly chosen analysis parameters and randomly sampled, with
              replacement, time steps for each iteration. The remaining steps described below are
              performed for each Monte Carlo iteration. If UQ is not used, wake losses are calculated
              once using the specified analysis parameters for the full set of available time steps.

        2. Identify the set of derated, curtailed, or unavailable turbines (i.e., turbines whose power
           production is limited not by wake losses but by operating mode) for each time step using a
           power curve outlier detection method.
        3. Calculate the average wind speed and power production for the set of normally operating
           (i.e., not derated) freestream turbines for each time step.

           a. Freestream turbines are those without any upstream turbines located within a
              user-specified sector of wind directions centered on the representative plant-level wind
              direction.

        4. Calculate the POR wake losses for the wind plant by comparing the potential energy production
           (sum of the mean freestream power production at each time step multiplied by the number of
           turbines in the wind power plant) to the actual energy production (sum of the actual wind
           plant power production at each time step). This procedure is then used to estimate the wake
           losses for each individual wind turbine.

           a. If :py:attr:`correct_for_derating` is True, then the potential power production of the
              wind plant is assumed to be the actual power produced by the derated turbines plus the
              mean power production of the freestream turbines for all other turbines in the wind
              plant. Again, a similar procedure is used to estimate individual turbine wake losses.

        5. Finally, estimate the long-term corrected wake losses using the long-term historical
           reanalysis data. Note that the long-term correction is determined for each reanalysis
           product specified by the user. If UQ is used, a random reanalysis product is selected
           each iteration. If UQ is not selected, the long-term corrected wake losses are calculated
           as the average wake losses determined for all reanalysis products.

           a. Calculate the long-term occurence frequencies for a set of wind direction and wind
              speed bins based on the hourly reanalysis data (typically, 10-20 years).
           b. Next, using a linear regression, compare the mean freestream wind speeds calculated
              from the SCADA data to the wind speeds from the reanalysis data and correct to remove
              biases.
           c. Compute the average potential and actual wind power plant production using the
              representative wind plant wind directions from the SCADA or met tower data in
              conjunction with the corrected freestream wind speeds for each wind direction and wind
              speed bin.
           d. Estimate the long-term corrected wake losses by comparing the long-term
              corrected potential and actual energy production. These are computed by weighting
              the average potential and actual power production for each wind condition bin
              with the long-term frequencies.
           e. Repeat to estimate the long-term corrected wake losses for each individual turbine.

    Args:
        plant (:obj:`PlantData`): A :py:attr:`openoa.plant.PlantData` object that has been validated
            with at least :py:attr:`openoa.plant.PlantData.analysis_type` = "WakeLosses".
        wind_direction_col (:obj:`string`, optional): Column name to use for wind direction.
            Defaults to "WMET_HorWdDir"
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
        reanalysis_products (:obj:`list`, optional): List of reanalysis products to use for long-term
            correction. If UQ = True, a single product will be selected form this list each Monte
            Carlo iteration. Defaults to ["merra2", "era5"].
        end_date_lt (:obj:`string` or :obj:`pandas.Timestamp`): The last date to use for the
            long-term correction. If None, the most recent date common to all reanalysis products
            will be used.
        wd_bin_width (float, optional): Wind direction bin size when identifying freestream wind
            turbines (degrees). Defaults to 5 degrees.
        freestream_sector_width (tuple | float, optional): Wind direction sector size to use when
            identifying freestream wind turbines (degrees). If no turbines are located upstream
            of a particular turbine within the sector, the turbine will be classified as a
            freestream turbine. When :py:attr:`UQ` = True, then this should be a tuple of the
            lower and upper bounds for the Monte Carlo sampling, and when :py:attr:`UQ` = False
            this should be a single value. If None, then a default value of 90 degrees will be
            used if :py:attr:`UQ` = False and a default value of (50, 110) will be used if
            :py:attr:`UQ` = True. Defaults to None.
        freestream_power_method (str, optional): Method used to determine the representative power
            prouction of the freestream turbines ("mean", "median", "max"). Defaults to "mean".
        freestream_wind_speed_method (str, optional): Method used to determine the representative
            wind speed of the freestream turbines ("mean", "median"). Defaults to "mean".
        correct_for_derating (bool, optional): Indicates whether derated, curtailed, or otherwise
            unavailable turbines should be flagged and excluded from the calculation of ideal
            freestream wind plant power production for a given time stamp. If True, ideal
            freestream power production will be calculated as the sum of the derated turbine
            powers added to the mean power of the freestream turbines in normal operation
            multiplied by the number of turbines operating normally in the wind plant. Defaults
            to True.
        derating_filter_wind_speed_start (tuple | float, optional): The wind speed above which
            turbines will be flagged as derated/curtailed/shutdown if power is less than 1% of
            rated power (m/s). Only used when :py:attr:`correct_for_derating` is True. This should
            be a tuple when :py:attr:`UQ` = True (values are Monte-Carlo sampled within the
            specified range) or a single value when :py:attr:`UQ` = False. If undefined (None), a
            value of 4.5 m/s will be used if :py:attr:`UQ` = False and values of (4.0, 5.0) will
            be used if :py:attr:`UQ` = True. Defaults to None.
        max_power_filter (tuple | float, optional): Maximum power threshold, defined as a fraction
            of rated power, to which the power curve bin filter should be applied. Only used when
            :py:attr:`correct_for_derating` = True. This should be a tuple when :py:attr:`UQ` =
            True (values are Monte-Carlo sampled within the specified range) or a single value
            when :py:attr:`UQ` = False. If undefined (None), a value of 0.95 will be used if
            :py:attr:`UQ` = False and values of (0.92, 0.98) will be used if :py:attr:`UQ` =
            True. Defaults to None.
        wind_bin_mad_thresh (tuple | float, optional): The filter threshold for each power bin
            used to identify derated/curtailed/shutdown turbines, expressed as the number of
            median absolute deviations above the median wind speed. Only used when
            :py:attr:`correct_for_derating` is True. This should be a tuple when
            :py:attr:`UQ` = True (values are Monte-Carlo sampled within the specified range) or
            a single value when :py:attr:`UQ` = False. If undefined (None), a value of 7.0 will
            be used if :py:attr:`UQ` = False and values of (4.0, 13.0) will be used if
            :py:attr:`UQ` = True. Defaults to None.
        wd_bin_width_LT_corr (float, optional): Size of wind direction bins used to calculate
            long-term frequencies from historical reanalysis data and correct wake losses during
            the period of record (degrees). Defaults to 5 degrees.
        ws_bin_width_LT_corr (float, optional): Size of wind speed bins used to calculate
            long-term frequencies from historical reanalysis data and correct wake losses during
            the period of record (m/s). Defaults to 1 m/s.
        num_years_LT (tuple | int, optional): Number of years of historical reanalysis data to
            use for long-term correction. This should be a tuple when :py:attr:`UQ` = True
            (values are Monte-Carlo sampled within the specified range) or a single value when
            :py:attr:`UQ` = False. If undefined (None), a value of 20 will be used if
            :py:attr:`UQ` = False and values of (10, 20) will be used if :py:attr:`UQ` = True.
            Defaults to None.
        assume_no_wakes_high_ws_LT_corr (bool, optional): If True, wind direction and wind speed
            bins for which operational data are missing above a certain wind speed threshold are
            corrected by assigning the wind turbines' rated power to both the actual and
            potential power production variables during the long term-correction process. This
            assumes there are no wake losses above the wind speed threshold. Defaults to True.
        no_wakes_ws_thresh_LT_corr (float, optional): The wind speed threshold (inclusive) above
            which rated power is assigned to both the actual and potential power production
            variables if operational data are missing for any wind direction and wind speed bin
            during the long term-correction process. This wind speed corresponds to the wind
            speed measured at freestream wind turbines. Only used if
            :py:attr:`assume_no_wakes_high_ws_LT_corr` = True. Defaults to 13 m/s.
        min_ws_bin_lin_reg (float, optional): The minimum wind speed bin to consider when finding
            linear regression from SCADA freestream wind speeds to reanalysis wind speeds. Defaults
            to 3.0
        bin_count_thresh_lin_reg (int, optional): The minimum number of samples required in a wind
            speed bin to include when finding linear regression from SCADA freestream wind speeds to
            reanalysis wind speeds. Defaults to 50.
    """

    plant: PlantData = field(converter=deepcopy, validator=attrs.validators.instance_of(PlantData))
    wind_direction_col: str = field(default="WMET_HorWdDir", converter=str)
    wind_direction_data_type: str = field(
        default="scada", validator=attrs.validators.in_(("scada", "tower"))
    )
    wind_direction_asset_ids: list[str] = field(default=None)
    UQ: bool = field(default=True, converter=bool)
    num_sim: int = field(default=100, converter=int)
    start_date: str | pd.Timestamp = field(default=None)
    end_date: str | pd.Timestamp = field(default=None)
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
    end_date_lt: str | pd.Timestamp = field(default=None)
    wd_bin_width: float = field(default=5.0)
    freestream_sector_width: float | tuple[float, float] = field(
        default=(50.0, 110.0), validator=validate_UQ_input
    )
    freestream_power_method: str = field(default="mean")
    freestream_wind_speed_method: str = field(default="mean")
    correct_for_derating: bool = field(default=True)
    derating_filter_wind_speed_start: float | tuple[float, float] = field(
        default=(4.0, 5.0), validator=validate_UQ_input
    )
    max_power_filter: float | tuple[float, float] = field(
        default=(0.92, 0.98), validator=validate_UQ_input
    )
    wind_bin_mad_thresh: float | tuple[float, float] = field(
        default=(4.0, 13.0), validator=validate_UQ_input
    )
    wd_bin_width_LT_corr: float = field(default=5.0)
    ws_bin_width_LT_corr: float = field(default=1.0)
    num_years_LT: int | tuple[int, int] = field(default=(10, 20), validator=validate_UQ_input)
    assume_no_wakes_high_ws_LT_corr: bool = field(default=True)
    no_wakes_ws_thresh_LT_corr: float = field(default=13.0)
    min_ws_bin_lin_reg: float = field(default=3.0)
    bin_count_thresh_lin_reg: int = field(default=50, validator=attrs.validators.instance_of(int))

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
    _run: pd.DataFrame = field(init=False)
    run_parameters: list[str] = field(
        init=False,
        default=[
            "num_sim",
            "reanalysis_products",
            "wd_bin_width",
            "freestream_sector_width",
            "freestream_power_method",
            "freestream_wind_speed_method",
            "correct_for_derating",
            "derating_filter_wind_speed_start",
            "max_power_filter",
            "wind_bin_mad_thresh",
            "wd_bin_width_LT_corr",
            "ws_bin_width_LT_corr",
            "num_years_LT",
            "assume_no_wakes_high_ws_LT_corr",
            "no_wakes_ws_thresh_LT_corr",
            "min_ws_bin_lin_reg",
            "bin_count_thresh_lin_reg",
        ],
    )

    @reanalysis_products.validator
    def check_reanalysis_products(self, attribute: attrs.Attribute, value: list[str]) -> None:
        """Checks that the provided reanalysis products actually exist in the reanalysis data."""
        if value == [None]:
            return
        valid = [*self.plant.reanalysis]
        invalid = list(set(value).difference(valid))
        if invalid:
            raise ValueError(
                f"The following input to `reanalysis_products`: {invalid} are not contained in `plant.reanalysis`: {valid}"
            )

    @logged_method_call
    def __attrs_post_init__(self):
        """
        Initialize logging and post-initialization setup steps.
        """
        logger.info("Initializing WakeLosses analysis object")

        if self.wind_direction_data_type == "scada":
            if {"WakeLosses-scada", "all"}.intersection(self.plant.analysis_type) == set():
                self.plant.analysis_type.append("WakeLosses-scada")
        if self.wind_direction_data_type == "tower":
            if {"WakeLosses-tower", "all"}.intersection(self.plant.analysis_type) == set():
                self.plant.analysis_type.append("WakeLosses-tower")

        # Ensure the data are up to spec before continuing with initialization
        self.plant.validate()

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
                [self.plant.reanalysis[product].index.max() for product in self.reanalysis_products]
            ).replace(minute=30)

        # Run preprocessing steps
        self._calculate_aggregate_dataframe()

    @logged_method_call
    def run(
        self,
        num_sim: int | None = None,
        reanalysis_products: list[str] | None = None,
        wd_bin_width: float | None = None,
        freestream_sector_width: float | None = None,
        freestream_power_method: str | None = None,
        freestream_wind_speed_method: str | None = None,
        correct_for_derating: bool | None = None,
        derating_filter_wind_speed_start: float | None = None,
        max_power_filter: float | None = None,
        wind_bin_mad_thresh: float | None = None,
        wd_bin_width_LT_corr: float | None = None,
        ws_bin_width_LT_corr: float | None = None,
        num_years_LT: int | None = None,
        assume_no_wakes_high_ws_LT_corr: bool | None = None,
        no_wakes_ws_thresh_LT_corr: float | None = None,
        min_ws_bin_lin_reg: float | None = None,
        bin_count_thresh_lin_reg: int | None = None,
    ):
        """
        Estimates wake losses by comparing wind plant energy production to energy production of the
        turbines identified as operating in freestream conditions. Wake losses are expressed as a
        fractional loss (e.g., 0.05 indicates a wake loss values of 5%).

        .. note:: If None is provided to any of the inputs, then the last used input value will be
            used for the analysis, and if no prior values were set, then this is the model's defaults.

        Args:
            num_sim (int, optional): Number of Monte Carlo iterations to perform. Only used if
                :py:attr:`UQ` = True. Defaults to 100.
            wd_bin_width (float, optional): Wind direction bin size when identifying freestream wind
                turbines (degrees). Defaults to 5 degrees.
            freestream_sector_width (tuple | float, optional): Wind direction sector size to use when
                identifying freestream wind turbines (degrees). If no turbines are located upstream
                of a particular turbine within the sector, the turbine will be classified as a
                freestream turbine. When :py:attr:`UQ` = True, then this should be a tuple of the
                lower and upper bounds for the Monte Carlo sampling, and when :py:attr:`UQ` = False
                this should be a single value. If None, then a default value of 90 degrees will be
                used if :py:attr:`UQ` = False and a default value of (50, 110) will be used if
                :py:attr:`UQ` = True. Defaults to None.
            freestream_power_method (str, optional): Method used to determine the representative power
                prouction of the freestream turbines ("mean", "median", "max"). Defaults to "mean".
            freestream_wind_speed_method (str, optional): Method used to determine the representative
                wind speed of the freestream turbines ("mean", "median"). Defaults to "mean".
            correct_for_derating (bool, optional): Indicates whether derated, curtailed, or otherwise
                unavailable turbines should be flagged and excluded from the calculation of ideal
                freestream wind plant power production for a given time stamp. If True, ideal
                freestream power production will be calculated as the sum of the derated turbine
                powers added to the mean power of the freestream turbines in normal operation
                multiplied by the number of turbines operating normally in the wind plant. Defaults
                to True.
            derating_filter_wind_speed_start (tuple | float, optional): The wind speed above which
                turbines will be flagged as derated/curtailed/shutdown if power is less than 1% of
                rated power (m/s). Only used when :py:attr:`correct_for_derating` is True. This should
                be a tuple when :py:attr:`UQ` = True (values are Monte-Carlo sampled within the
                specified range) or a single value when :py:attr:`UQ` = False. If undefined (None), a
                value of 4.5 m/s will be used if :py:attr:`UQ` = False and values of (4.0, 5.0) will
                be used if :py:attr:`UQ` = True. Defaults to None.
            max_power_filter (tuple | float, optional): Maximum power threshold, defined as a fraction
                of rated power, to which the power curve bin filter should be applied. Only used when
                :py:attr:`correct_for_derating` = True. This should be a tuple when :py:attr:`UQ` =
                True (values are Monte-Carlo sampled within the specified range) or a single value
                when :py:attr:`UQ` = False. If undefined (None), a value of 0.95 will be used if
                :py:attr:`UQ` = False and values of (0.92, 0.98) will be used if :py:attr:`UQ` =
                True. Defaults to None.
            wind_bin_mad_thresh (tuple | float, optional): The filter threshold for each power bin
                used to identify derated/curtailed/shutdown turbines, expressed as the number of
                median absolute deviations above the median wind speed. Only used when
                :py:attr:`correct_for_derating` is True. This should be a tuple when
                :py:attr:`UQ` = True (values are Monte-Carlo sampled within the specified range) or
                a single value when :py:attr:`UQ` = False. If undefined (None), a value of 7.0 will
                be used if :py:attr:`UQ` = False and values of (4.0, 13.0) will be used if
                :py:attr:`UQ` = True. Defaults to None.
            wd_bin_width_LT_corr (float, optional): Size of wind direction bins used to calculate
                long-term frequencies from historical reanalysis data and correct wake losses during
                the period of record (degrees). Defaults to 5 degrees.
            ws_bin_width_LT_corr (float, optional): Size of wind speed bins used to calculate
                long-term frequencies from historical reanalysis data and correct wake losses during
                the period of record (m/s). Defaults to 1 m/s.
            num_years_LT (tuple | int, optional): Number of years of historical reanalysis data to
                use for long-term correction. This should be a tuple when :py:attr:`UQ` = True
                (values are Monte-Carlo sampled within the specified range) or a single value when
                :py:attr:`UQ` = False. If undefined (None), a value of 20 will be used if
                :py:attr:`UQ` = False and values of (10, 20) will be used if :py:attr:`UQ` = True.
                Defaults to None.
            assume_no_wakes_high_ws_LT_corr (bool, optional): If True, wind direction and wind speed
                bins for which operational data are missing above a certain wind speed threshold are
                corrected by assigning the wind turbines' rated power to both the actual and
                potential power production variables during the long term-correction process. This
                assumes there are no wake losses above the wind speed threshold. Defaults to True.
            no_wakes_ws_thresh_LT_corr (float, optional): The wind speed threshold (inclusive) above
                which rated power is assigned to both the actual and potential power production
                variables if operational data are missing for any wind direction and wind speed bin
                during the long term-correction process. This wind speed corresponds to the wind
                speed measured at freestream wind turbines. Only used if
                :py:attr:`assume_no_wakes_high_ws_LT_corr` = True. Defaults to 13 m/s.
            min_ws_bin_lin_reg (float, optional): The minimum wind speed bin to consider when
                finding linear regression from SCADA freestream wind speeds to reanalysis wind
                speeds. Defaults to 3.0
            bin_count_thresh_lin_reg (int, optional): The minimum number of samples required in a
                wind speed bin to include when finding linear regression from SCADA freestream wind
                speeds to reanalysis wind speeds. Defaults to 50.
        """
        initial_parameters = {}
        # Assign default parameter values depending on whether UQ is performed
        if num_sim is not None:
            initial_parameters["num_sim"] = num_sim
            self.num_sim = num_sim
        if reanalysis_products is not None:
            initial_parameters["reanalysis_products"] = reanalysis_products
            self.reanalysis_products = reanalysis_products
            logger.warning(
                f"`reanalysis_products` has been changed, be sure the `end_date_lt`"
                f"({self.end_date_lt}) is contained in the updated reanalyis products subset."
            )
        if wd_bin_width is not None:
            initial_parameters["wd_bin_width"] = self.wd_bin_width
            self.wd_bin_width = wd_bin_width
        if freestream_sector_width is not None:
            initial_parameters["freestream_sector_width"] = self.freestream_sector_width
            self.freestream_sector_width = freestream_sector_width
        if freestream_power_method is not None:
            initial_parameters["freestream_power_method"] = self.freestream_power_method
            self.freestream_power_method = freestream_power_method
        if freestream_wind_speed_method is not None:
            initial_parameters["freestream_wind_speed_method"] = self.freestream_wind_speed_method
            self.freestream_wind_speed_method = freestream_wind_speed_method
        if correct_for_derating is not None:
            initial_parameters["correct_for_derating"] = correct_for_derating
            self.correct_for_derating = correct_for_derating
        if derating_filter_wind_speed_start is not None:
            initial_parameters[
                "derating_filter_wind_speed_start"
            ] = self.derating_filter_wind_speed_start
            self.derating_filter_wind_speed_start = derating_filter_wind_speed_start
        if max_power_filter is not None:
            initial_parameters["max_power_filter"] = self.max_power_filter
            self.max_power_filter = max_power_filter
        if wind_bin_mad_thresh is not None:
            initial_parameters["wind_bin_mad_thresh"] = self.wind_bin_mad_thresh
            self.wind_bin_mad_thresh = wind_bin_mad_thresh
        if wd_bin_width_LT_corr is not None:
            initial_parameters["wd_bin_width_LT_corr"] = self.wd_bin_width_LT_corr
            self.wd_bin_width_LT_corr = wd_bin_width_LT_corr
        if ws_bin_width_LT_corr is not None:
            initial_parameters["ws_bin_width_LT_corr"] = self.ws_bin_width_LT_corr
            self.ws_bin_width_LT_corr = ws_bin_width_LT_corr
        if num_years_LT is not None:
            initial_parameters["num_years_LT"] = self.num_years_LT
            self.num_years_LT = num_years_LT
        if assume_no_wakes_high_ws_LT_corr is not None:
            initial_parameters[
                "assume_no_wakes_high_ws_LT_corr"
            ] = self.assume_no_wakes_high_ws_LT_corr
            self.assume_no_wakes_high_ws_LT_corr = assume_no_wakes_high_ws_LT_corr
        if no_wakes_ws_thresh_LT_corr is not None:
            initial_parameters["no_wakes_ws_thresh_LT_corr"] = self.no_wakes_ws_thresh_LT_corr
            self.no_wakes_ws_thresh_LT_corr = no_wakes_ws_thresh_LT_corr
        if min_ws_bin_lin_reg is not None:
            initial_parameters["min_ws_bin_lin_reg"] = self.min_ws_bin_lin_reg
            self.min_ws_bin_lin_reg = min_ws_bin_lin_reg
        if bin_count_thresh_lin_reg is not None:
            initial_parameters["bin_count_thresh_lin_reg"] = self.bin_count_thresh_lin_reg
            self.bin_count_thresh_lin_reg = bin_count_thresh_lin_reg

        # Set up Monte Carlo simulation inputs if UQ = True or single simulation inputs if UQ = False.
        self._setup_monte_carlo_inputs()

        for n in tqdm(range(self.num_sim)):
            self._run = self.inputs.loc[n].copy()

            # Estimate periods when each turbine is unavailable, derated, or curtailed, based on power curve filtering
            for t in self.turbine_ids:
                self.aggregate_df[("derate_flag", t)] = False

            if self.correct_for_derating:
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

            wd_bins = np.arange(0.0, 360.0, self.wd_bin_width)

            # Create columns for turbine power and wind speed during normal operation (NaN otherwise)
            for t in self.turbine_ids:
                valid_inds = ~self.aggregate_df_sample[("derate_flag", t)]
                self.aggregate_df_sample.loc[
                    valid_inds, ("power_normal", t)
                ] = self.aggregate_df_sample.loc[valid_inds, ("WTUR_W", t)]

            for t in self.turbine_ids:
                valid_inds = ~self.aggregate_df_sample[("derate_flag", t)]
                self.aggregate_df_sample.loc[
                    valid_inds, ("windspeed_normal", t)
                ] = self.aggregate_df_sample.loc[valid_inds, ("WMET_HorWdSpd", t)]

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
                _agg_wd = self.aggregate_df_sample["wind_direction_ref"]
                if wd == 0.0:
                    wd_bin_flag = _agg_wd >= 360.0 - 0.5 * self.wd_bin_width
                    wd_bin_flag |= _agg_wd < (
                        freestream_sector_wds[i_wd + 1] - 0.5 * self.wd_bin_width
                    )
                elif i_wd < len(freestream_sector_wds) - 1:
                    wd_bin_flag = _agg_wd >= (wd - 0.5 * self.wd_bin_width)
                    wd_bin_flag &= _agg_wd < (
                        freestream_sector_wds[i_wd + 1] - 0.5 * self.wd_bin_width
                    )
                elif (i_wd == len(freestream_sector_wds) - 1) & (freestream_sector_wds[0] == 0.0):
                    wd_bin_flag = _agg_wd >= (wd - 0.5 * self.wd_bin_width)
                    wd_bin_flag &= _agg_wd < (360.0 - 0.5 * self.wd_bin_width)
                else:  # last wind direction in dictionary and first wind direction is not zero:
                    wd_bin_flag = _agg_wd >= (wd - 0.5 * self.wd_bin_width)
                    wd_bin_flag |= _agg_wd < (freestream_sector_wds[0] - 0.5 * self.wd_bin_width)

                # Assign representative energy and wind speed of freestream turbines. If correct_for_derating
                # is True, only freestream turbines operating normally will be considered.

                _power = self.aggregate_df_sample.loc[wd_bin_flag, "power_normal"]
                if self.freestream_power_method == "mean":
                    _power = _power[freestream_turbine_ids].mean(axis=1)
                elif self.freestream_power_method == "median":
                    _power = _power[freestream_turbine_ids].median(axis=1)
                elif self.freestream_power_method == "max":
                    _power = _power[freestream_turbine_ids].max(axis=1)
                self.aggregate_df_sample.loc[wd_bin_flag, "power_mean_freestream"] = _power

                _ws = self.aggregate_df_sample.loc[wd_bin_flag, "windspeed_normal"]
                if self.freestream_wind_speed_method == "mean":
                    _ws = _ws[freestream_turbine_ids].mean(axis=1)
                elif self.freestream_wind_speed_method == "median":
                    _ws = _ws[freestream_turbine_ids].median(axis=1)
                self.aggregate_df_sample.loc[wd_bin_flag, "windspeed_mean_freestream"] = _ws

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
                self.aggregate_df_sample["WTUR_W"] * self.aggregate_df_sample["derate_flag"]
            ).sum(axis=1)

            total_potential_freestream_power = self.aggregate_df_sample["power_mean_freestream"] * (
                ~self.aggregate_df_sample["derate_flag"]
            ).sum(axis=1)

            # Assign total potential power
            self.aggregate_df_sample["potential_plant_power"] = (
                total_potential_freestream_power + total_derated_turbine_power
            )

            # Assign actual total power produced by wind plant
            self.aggregate_df_sample["actual_plant_power"] = self.aggregate_df_sample["WTUR_W"].sum(
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
                self.wd_bin_width_LT_corr
                * (
                    self.aggregate_df_sample["wind_direction_ref"] / self.wd_bin_width_LT_corr
                ).round()
            )
            self.aggregate_df_sample.loc[
                self.aggregate_df_sample["wind_direction_bin"] == 360.0, "wind_direction_bin"
            ] = 0.0

            # Calculate turbine-level wake losses during period of record
            turbine_wake_losses_por = len(self.turbine_ids) * [0.0]
            for i, t in enumerate(self.turbine_ids):
                # Determine ideal turbine energy as sum of the power produced by the turbine when it
                # is derated and the mean power produced by all freestream turbines when the turbine
                # is operating normally
                self.aggregate_df_sample.loc[
                    ~self.aggregate_df_sample[("derate_flag", t)], ("potential_turbine_power", t)
                ] = self.aggregate_df_sample.loc[
                    ~self.aggregate_df_sample[("derate_flag", t)], "power_mean_freestream"
                ]

                self.aggregate_df_sample.loc[
                    self.aggregate_df_sample[("derate_flag", t)], ("potential_turbine_power", t)
                ] = self.aggregate_df_sample.loc[
                    self.aggregate_df_sample[("derate_flag", t)], ("WTUR_W", t)
                ]
                turbine_wake_losses_por[i] = (
                    1
                    - self.aggregate_df_sample[("WTUR_W", t)].sum()
                    / self.aggregate_df_sample[("potential_turbine_power", t)].sum()
                )

            df_wd_bin = self.aggregate_df_sample.groupby("wind_direction_bin").sum()

            # Save plant and turbine-level wake losses binned by wind direction
            wake_losses_por_wd = (
                df_wd_bin["actual_plant_power"] / df_wd_bin["potential_plant_power"]
            ).values

            turbine_wake_losses_por_wd = np.empty(
                [len(self.turbine_ids), int(360.0 / self.wd_bin_width_LT_corr)]
            )
            for i, t in enumerate(self.turbine_ids):
                turbine_wake_losses_por_wd[i, :] = (
                    df_wd_bin[("WTUR_W", t)] / df_wd_bin[("potential_turbine_power", t)]
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

            wake_losses_lt_all_products = np.empty([len(self.reanalysis_products), 1])
            turbine_wake_losses_lt_all_products = np.empty(
                [len(self.reanalysis_products), len(self.turbine_ids)]
            )

            wake_losses_lt_wd_all_products = np.empty(
                [len(self.reanalysis_products), int(360.0 / self.wd_bin_width_LT_corr)]
            )
            turbine_wake_losses_lt_wd_all_products = np.empty(
                [
                    len(self.reanalysis_products),
                    len(self.turbine_ids),
                    int(360.0 / self.wd_bin_width_LT_corr),
                ]
            )
            energy_lt_wd_all_products = np.empty(
                [len(self.reanalysis_products), int(360.0 / self.wd_bin_width_LT_corr)]
            )

            wake_losses_por_ws_all_products = np.empty(
                [len(self.reanalysis_products), int(30.0 / self.ws_bin_width_LT_corr) + 1]
            )
            turbine_wake_losses_por_ws_all_products = np.empty(
                [
                    len(self.reanalysis_products),
                    len(self.turbine_ids),
                    int(30.0 / self.ws_bin_width_LT_corr) + 1,
                ]
            )
            energy_por_ws_all_products = np.empty(
                [len(self.reanalysis_products), int(30.0 / self.ws_bin_width_LT_corr) + 1]
            )

            wake_losses_lt_ws_all_products = np.empty(
                [len(self.reanalysis_products), int(30.0 / self.ws_bin_width_LT_corr) + 1]
            )
            turbine_wake_losses_lt_ws_all_products = np.empty(
                [
                    len(self.reanalysis_products),
                    len(self.turbine_ids),
                    int(30.0 / self.ws_bin_width_LT_corr) + 1,
                ]
            )
            energy_lt_ws_all_products = np.empty(
                [len(self.reanalysis_products), int(30.0 / self.ws_bin_width_LT_corr) + 1]
            )

            for i_rean, product in enumerate(self.reanalysis_products):
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

        self.set_values(initial_parameters)

    @logged_method_call
    def _setup_monte_carlo_inputs(self):
        """
        Create and populate the data frame defining the Monte Carlo simulation parameters. This
        data frame is stored as ``self.inputs``.
        """

        if self.UQ:
            inputs = {
                "reanalysis_product": random.choices(self.reanalysis_products, k=self.num_sim),
                "freestream_sector_width": np.random.randint(
                    self.freestream_sector_width[0],
                    self.freestream_sector_width[1] + 1,
                    self.num_sim,
                ),
                "wind_bin_mad_thresh": np.random.randint(
                    self.wind_bin_mad_thresh[0], self.wind_bin_mad_thresh[1] + 1, self.num_sim
                ),
                "derating_filter_wind_speed_start": np.random.randint(
                    self.derating_filter_wind_speed_start[0] * 10,
                    self.derating_filter_wind_speed_start[1] * 10 + 1,
                    self.num_sim,
                )
                / 10.0,
                "max_power_filter": np.random.randint(
                    self.max_power_filter[0] * 100,
                    self.max_power_filter[1] * 100 + 1,
                    self.num_sim,
                )
                / 100.0,
                "num_years_LT": np.random.randint(
                    self.num_years_LT[0], self.num_years_LT[1] + 1, self.num_sim
                ),
            }
            self.inputs = pd.DataFrame(inputs)

            self.wake_losses_por = np.empty([self.num_sim, 1])
            self.turbine_wake_losses_por = np.empty([self.num_sim, len(self.turbine_ids)])
            self.wake_losses_lt = np.empty([self.num_sim, 1])
            self.turbine_wake_losses_lt = np.empty([self.num_sim, len(self.turbine_ids)])

            # For saving wake losses and energy production binned by wind direction
            self.wake_losses_por_wd = np.empty(
                [self.num_sim, int(360.0 / self.wd_bin_width_LT_corr)]
            )
            self.turbine_wake_losses_por_wd = np.empty(
                [self.num_sim, len(self.turbine_ids), int(360.0 / self.wd_bin_width_LT_corr)]
            )
            self.wake_losses_lt_wd = np.empty(
                [self.num_sim, int(360.0 / self.wd_bin_width_LT_corr)]
            )
            self.turbine_wake_losses_lt_wd = np.empty(
                [self.num_sim, len(self.turbine_ids), int(360.0 / self.wd_bin_width_LT_corr)]
            )

            self.energy_por_wd = np.empty([self.num_sim, int(360.0 / self.wd_bin_width_LT_corr)])
            self.energy_lt_wd = np.empty([self.num_sim, int(360.0 / self.wd_bin_width_LT_corr)])

            # For saving wake losses and energy production binned by wind speed
            self.wake_losses_por_ws = np.empty(
                [self.num_sim, int(30.0 / self.ws_bin_width_LT_corr) + 1]
            )
            self.turbine_wake_losses_por_ws = np.empty(
                [self.num_sim, len(self.turbine_ids), int(30.0 / self.ws_bin_width_LT_corr) + 1]
            )
            self.wake_losses_lt_ws = np.empty(
                [self.num_sim, int(30.0 / self.ws_bin_width_LT_corr) + 1]
            )
            self.turbine_wake_losses_lt_ws = np.empty(
                [self.num_sim, len(self.turbine_ids), int(30.0 / self.ws_bin_width_LT_corr) + 1]
            )

            self.energy_por_ws = np.empty([self.num_sim, int(30.0 / self.ws_bin_width_LT_corr) + 1])
            self.energy_lt_ws = np.empty([self.num_sim, int(30.0 / self.ws_bin_width_LT_corr) + 1])

        elif not self.UQ:
            inputs = {
                "reanalysis_product": self.reanalysis_products,
                "freestream_sector_width": len(self.reanalysis_products)
                * [self.freestream_sector_width],
                "wind_bin_mad_thresh": len(self.reanalysis_products) * [self.wind_bin_mad_thresh],
                "derating_filter_wind_speed_start": len(self.reanalysis_products)
                * [self.derating_filter_wind_speed_start],
                "max_power_filter": len(self.reanalysis_products) * [self.max_power_filter],
                "num_years_LT": len(self.reanalysis_products) * [self.num_years_LT],
            }
            self.inputs = pd.DataFrame(inputs)

            self.num_sim = 1

    @logged_method_call
    def _calculate_aggregate_dataframe(self):
        """
        Creates a data frame with relevant scada columns, plant-level columns, and reanalysis
        variables to be used for the wake loss analysis. The reference mean wind direction is then
        added to the data frame.
        """

        # keep relevant SCADA columns, create a unique time index and two-level turbine variable columns
        # (variable name and turbine asset_id)

        # include scada wind direction column only if using scada to determine mean wind direction for wind plant
        scada_cols = ["WMET_HorWdSpd", "WTUR_W"]
        if self.wind_direction_data_type == "scada":
            scada_cols.insert(1, self.wind_direction_col)

        self.aggregate_df = self.plant.scada.loc[
            self.start_date : self.end_date, scada_cols
        ].unstack()

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
        """

        if self.wind_direction_data_type == "scada":
            self.aggregate_df["wind_direction_ref"] = met.circular_mean(
                self.aggregate_df[self.wind_direction_col][self.wind_direction_asset_ids], axis=1
            )
        elif self.wind_direction_data_type == "tower":
            df_tower = self.plant.tower[[self.wind_direction_col]].unstack()

            self.aggregate_df["wind_direction_ref"] = met.circular_mean(
                df_tower[self.wind_direction_col][self.wind_direction_asset_ids], axis=1
            )

    @logged_method_call
    def _include_reanal_data(self):
        """
        Combines reanalysis data columns with the aggregate data frame for use in long-term correction.
        """

        # combine all wind speed and wind direction reanalysis variables into aggregate data frame

        for product in self.reanalysis_products:
            df_rean = self.plant.reanalysis[product][["WMETR_HorWdSpd", "WMETR_HorWdDir"]].copy()

            # Drop minute field
            df_rean.index = df_rean.index.floor("h")

            # Upsample to match SCADA data frequency
            df_rean = df_rean.resample(self.plant.metadata.scada.frequency).ffill()
            df_rean = df_rean.add_suffix(f"_{product}")
            df_rean = df_rean[df_rean.index.isin(self.aggregate_df.index)]

            self.aggregate_df[[col for col in df_rean.columns]] = df_rean

    @logged_method_call
    def _identify_derating(self):
        """
        Estimates whether each turbine is derated, curtailed, or otherwise not operating for each time stamp based on
        power curve filtering. A derated flag is then added to the aggregate data frame for each turbine.
        """

        for t in self.turbine_ids:
            # Apply window range filter to flag samples for which wind speed is greater than a threshold and power is
            # below 1% of rated power

            turb_capac = self.plant.asset.loc[t, "rated_power"]

            flag_window = filters.window_range_flag(
                window_col=self.aggregate_df[("WMET_HorWdSpd", t)],
                window_start=self._run.derating_filter_wind_speed_start,
                window_end=40,
                value_col=self.aggregate_df[("WTUR_W", t)],
                value_min=0.01 * turb_capac,
                value_max=1.2 * turb_capac,
            )

            # Apply bin-based filter to flag samples for which wind speed is greater than a threshold from the median
            # wind speed in each power bin
            bin_width_frac = 0.04 * (
                self._run.max_power_filter - 0.01
            )  # split into 25 bins TODO: make this an optional argument?
            flag_bin = filters.bin_filter(
                bin_col=self.aggregate_df[("WTUR_W", t)],
                value_col=self.aggregate_df[("WMET_HorWdSpd", t)],
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
        Estimates long term-corrected wake losses by binning wake losses by wind direction and wind
        speed and weighting by bin frequencies from long-term historical reanalysis data.

        Returns:
            tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The estimated long
                term-corrected wake losses, an array containing the estimated turbine-level long
                term-corrected wake losses, and arrays containing the long-term corrected plant and
                turbine-level wake losses as well as the normalized wind plant energy production
                binned by wind direction
        """
        # First, create hourly data frame for LT correction to match resolution of reanalysis data
        df_1hr = self.aggregate_df_sample[
            [
                ("wind_direction_ref", ""),
                ("windspeed_mean_freestream", ""),
                ("actual_plant_power", ""),
                ("potential_plant_power", ""),
            ]
            + [("WTUR_W", t) for t in self.turbine_ids]
            + [("potential_turbine_power", t) for t in self.turbine_ids]
            + [(f"WMETR_HorWdSpd_{self._run.reanalysis_product}", "")]
        ].copy()

        df_1hr = df_1hr.resample("h").mean().dropna(how="any")

        df_1hr["windspeed_mean_freestream_bin"] = df_1hr["windspeed_mean_freestream"].round()

        # Bin by integer wind speeds
        df_ws_bin = df_1hr.groupby(("windspeed_mean_freestream_bin", "")).mean()
        df_ws_bin_count = df_1hr.groupby(("windspeed_mean_freestream_bin", "")).count()

        valid_ws_bins = (df_ws_bin.index >= self.min_ws_bin_lin_reg) & (
            df_ws_bin_count["windspeed_mean_freestream"] >= self.bin_count_thresh_lin_reg
        )

        # Find linear regression mapping from SCADA freestream wind speed to reanalysis wind speeds
        # and use to correct SCADA freestream wind speeds
        reg = LinearRegression().fit(
            df_ws_bin.loc[valid_ws_bins].index.values.reshape(-1, 1),
            df_ws_bin.loc[valid_ws_bins, f"WMETR_HorWdSpd_{self._run.reanalysis_product}"].values,
        )

        df_1hr[f"windspeed_mean_freestream_corr_{self._run.reanalysis_product}"] = reg.predict(
            df_1hr["windspeed_mean_freestream"].values.reshape(-1, 1)
        )

        # adjust the no_wakes_ws_thresh_LT_corr parameter to relect the SCADA wind speed correction as well
        no_wakes_ws_corr_thresh_LT_corr = np.round(
            reg.predict(np.array(self.no_wakes_ws_thresh_LT_corr).reshape(1, -1))[0]
        )

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
            self.ws_bin_width_LT_corr
            * (df_reanal["WMETR_HorWdSpd"] / self.ws_bin_width_LT_corr).round()
        )
        df_reanal["wind_direction_bin"] = (
            self.wd_bin_width_LT_corr
            * (df_reanal["WMETR_HorWdDir"] / self.wd_bin_width_LT_corr).round()
        )
        df_reanal.loc[df_reanal["wind_direction_bin"] == 360.0, "wind_direction_bin"] = 0.0

        df_reanal["freq"] = 1.0
        df_reanal = df_reanal.groupby(["wind_direction_bin", "windspeed_bin"]).count()["freq"]

        # Create data frame with long-term frequencies of wind direction and wind speed bins from reanalysis data
        df_reanal_freqs = pd.DataFrame(df_reanal / df_reanal.sum())

        # Weight wake losses in each wind direction and wind speed bin by long-term frequencies to
        # estimate long-term wake losses
        df_1hr["windspeed_bin"] = (
            self.ws_bin_width_LT_corr
            * (
                df_1hr[f"windspeed_mean_freestream_corr_{self._run.reanalysis_product}"]
                / self.ws_bin_width_LT_corr
            ).round()
        )
        df_1hr["wind_direction_bin"] = (
            self.wd_bin_width_LT_corr
            * (df_1hr["wind_direction_ref"] / self.wd_bin_width_LT_corr).round()
        )
        df_1hr.loc[df_1hr["wind_direction_bin"] == 360.0, "wind_direction_bin"] = 0.0

        # First, compute POR wake losses as a function of wind speed
        df_1hr_ws_por_bin = df_1hr.groupby(("windspeed_bin", "")).sum()

        # reindex to fill in missing wind speed bins
        index = np.arange(0.0, 31.0, self.ws_bin_width_LT_corr).tolist()
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
            [len(self.turbine_ids), int(30.0 / self.ws_bin_width_LT_corr) + 1]
        )
        for i, t in enumerate(self.turbine_ids):
            turbine_wake_losses_por_ws[i, :] = (
                df_1hr_ws_por_bin[("WTUR_W", t)] / df_1hr_ws_por_bin[("potential_turbine_power", t)]
            ).values

        # Bin variables by wind direction and wind speed
        df_1hr_bin = df_1hr.groupby([("wind_direction_bin", ""), ("windspeed_bin", "")]).mean()

        df_1hr_bin = pd.concat([df_reanal_freqs, df_1hr_bin], axis=1)

        # If specified, assume no wake losses at wind speeds above a given threshold for bins where
        # data are missing by assigning rated power to the actual and potential power production
        if self.assume_no_wakes_high_ws_LT_corr:
            fill_inds = (df_1hr_bin[("actual_plant_power", "")].isna()) & (
                df_1hr_bin.index.get_level_values(1) >= no_wakes_ws_corr_thresh_LT_corr
            )
            df_1hr_bin.loc[
                fill_inds, [("actual_plant_power", ""), ("potential_plant_power", "")]
            ] = (self.plant.metadata.capacity * 1e3)
            df_1hr_bin.loc[
                fill_inds,
                [("WTUR_W", t) for t in self.turbine_ids]
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

            df_1hr_bin[("energy_avg", t)] = df_1hr_bin["freq"] * df_1hr_bin[("WTUR_W", t)]
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
            [len(self.turbine_ids), int(360.0 / self.wd_bin_width_LT_corr)]
        )
        for i, t in enumerate(self.turbine_ids):
            turbine_wake_losses_lt_wd[i, :] = (
                df_1hr_wd_bin[("energy_avg", t)] / df_1hr_wd_bin[("potential_turbine_energy", t)]
            ).values

        # Save long-term corrected plant and turbine-level wake losses binned by wind speed
        df_1hr_ws_bin = df_1hr_bin.groupby(level=[1]).sum()

        # reindex to fill in missing wind speed bins
        index = np.arange(0.0, 31.0, self.ws_bin_width_LT_corr).tolist()
        df_1hr_ws_bin = df_1hr_ws_bin.reindex(index)

        wake_losses_lt_ws = (
            df_1hr_ws_bin["actual_plant_energy"] / df_1hr_ws_bin["potential_plant_energy"]
        ).values

        energy_lt_ws = (
            df_1hr_ws_bin["actual_plant_energy"].values / df_1hr_ws_bin["actual_plant_energy"].sum()
        )

        turbine_wake_losses_lt_ws = np.empty(
            [len(self.turbine_ids), int(30.0 / self.ws_bin_width_LT_corr) + 1]
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
        self,
        plot_norm_energy: bool = True,
        turbine_id: str = None,
        xlim: tuple[float, float] = (None, None),
        ylim_efficiency: tuple[float, float] = (None, None),
        ylim_energy: tuple[float, float] = (None, None),
        return_fig: bool = False,
        figure_kwargs: dict = None,
        plot_kwargs_line: dict = {},
        plot_kwargs_fill: dict = {},
        legend_kwargs: dict = {},
    ):
        """
        Plots wake losses in the form of wind farm efficiency as well as normalized wind plant energy
        production for both the period of record and with the long-term correction as a function of
        wind direction.

        Args:
            plot_norm_energy (bool, optional): If True, include a plot of normalized wind plant energy
                production as a function of wind direction in addition to the wind farm efficiency plot.
                Defaults to True.
            turbine_id (str, optional): Turbine asset_id to plot wake losses for. If None, wake losses for the
                entire wind plant will be plotted. Defaults to None.
            xlim (:obj:`tuple[float, float]`, optional): A tuple of floats representing the x-axis
                wind direction plotting display limits (degrees). Defaults to (None, None).
            ylim_efficiency (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display
                limits for the wind farm efficiency plot (top plot). Defaults to (None, None).
            ylim_energy (:obj:`tuple[float, float]`, optional): If `plot_norm_energy` is True, a tuple
                of the y-axis plotting display limits for the wind farm energy distribution plot (bottom
                plot). Defaults to (None, None).
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to `plt.figure()`. Defaults to None.
            plot_kwargs_line (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                ``ax.plot()`` for plotting lines for the wind farm efficiency and, if `plot_norm_energy` is True,
                energy distributions subplots. Defaults to {}.
            plot_kwargs_fill (:obj:`dict`, optional): If `UQ` is True, additional plotting keyword arguments
                that are passed to ``ax.fill_between()`` for plotting shading regions for 95% confidence
                intervals for the wind farm efficiency and, if `plot_norm_energy` is True, energy
                distributions subplots. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                ``ax.legend()`` for the wind farm efficiency and, if `plot_norm_energy` is True, energy
                distributions subplots. Defaults to {}.
        Returns:
            None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes] | tuple[matplotlib.pyplot.Figure, tuple [matplotlib.pyplot.Axes, matplotlib.pyplot.Axes]]:
                If :py:attr:`return_fig` is True, then the figure and axes object(s), corresponding to the wake
                loss plot or, if :py:attr:`plot_norm_energy` is True, wake loss and normalized energy plots, are
                returned for further tinkering/saving.
        """

        wd_bins = np.arange(0.0, 360.0, self.wd_bin_width_LT_corr)

        if turbine_id is None:
            efficiency_data_por = self.wake_losses_por_wd
            efficiency_data_lt = self.wake_losses_lt_wd

        else:
            turbine_index = self.turbine_ids.index(turbine_id)

            if self.UQ:
                efficiency_data_por = self.turbine_wake_losses_por_wd[:, turbine_index, :]
                efficiency_data_lt = self.turbine_wake_losses_lt_wd[:, turbine_index, :]
            else:
                efficiency_data_por = self.turbine_wake_losses_por_wd[turbine_index, :]
                efficiency_data_lt = self.turbine_wake_losses_lt_wd[turbine_index, :]

        if plot_norm_energy:
            energy_data_por = self.energy_por_wd
            energy_data_lt = self.energy_lt_wd
        else:
            energy_data_por = None
            energy_data_lt = None

        return plot.plot_wake_losses(
            bins=wd_bins,
            efficiency_data_por=efficiency_data_por,
            efficiency_data_lt=efficiency_data_lt,
            energy_data_por=energy_data_por,
            energy_data_lt=energy_data_lt,
            bin_axis_label=r"Wind Direction ($^\circ$)",
            turbine_id=turbine_id,
            xlim=xlim,
            ylim_efficiency=ylim_efficiency,
            ylim_energy=ylim_energy,
            return_fig=return_fig,
            figure_kwargs=figure_kwargs,
            plot_kwargs_line=plot_kwargs_line,
            plot_kwargs_fill=plot_kwargs_fill,
            legend_kwargs=legend_kwargs,
        )

    def plot_wake_losses_by_wind_speed(
        self,
        plot_norm_energy: bool = True,
        turbine_id: str = None,
        xlim: tuple[float, float] = (None, None),
        ylim_efficiency: tuple[float, float] = (None, None),
        ylim_energy: tuple[float, float] = (None, None),
        return_fig: bool = False,
        figure_kwargs: dict = None,
        plot_kwargs_line: dict = {},
        plot_kwargs_fill: dict = {},
        legend_kwargs: dict = {},
    ):
        """
        Plots wake losses in the form of wind farm efficiency as well as normalized wind plant energy
        production for both the period of record and with the long-term correction as a function of
        wind speed.

        Args:
            plot_norm_energy (bool, optional): If True, include a plot of normalized wind plant energy
                production as a function of wind speed in addition to the wind farm efficiency plot. Defaults to
                True.
            turbine_id (str, optional): Turbine asset_id to plot wake losses for. If None, wake losses for the
                entire wind plant will be plotted. Defaults to None.
            xlim (:obj:`tuple[float, float]`, optional): A tuple of floats representing the x-axis
                wind speed plotting display limits (degrees). Defaults to (None, None).
            ylim_efficiency (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display
                limits for the wind farm efficiency plot (top plot). Defaults to (None, None).
            ylim_energy (:obj:`tuple[float, float]`, optional): If `plot_norm_energy` is True, a tuple
                of the y-axis plotting display limits for the wind farm energy distribution plot (bottom
                plot). Defaults to (None, None).
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to ``plt.figure()``. Defaults to None.
            plot_kwargs_line (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                ``ax.plot()`` for plotting lines for the wind farm efficiency and, if :py:attr:`plot_norm_energy` is True,
                energy distributions subplots. Defaults to {}.
            plot_kwargs_fill (:obj:`dict`, optional): If `UQ` is True, additional plotting keyword arguments
                that are passed to ``ax.fill_between()`` for plotting shading regions for 95% confidence
                intervals for the wind farm efficiency and, if :py:attr:`plot_norm_energy` is True, energy
                distributions subplots. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                ``ax.legend()`` for the wind farm efficiency and, if :py:attr:`plot_norm_energy` is True, energy
                distributions subplots. Defaults to {}.
        Returns:
            None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes] | tuple[matplotlib.pyplot.Figure, tuple [matplotlib.pyplot.Axes, matplotlib.pyplot.Axes]]:
                If :py:attr:`return_fig` is True, then the figure and axes object(s), corresponding to the wake
                loss plot or, if :py:attr:`plot_norm_energy` is True, wake loss and normalized energy plots, are
                returned for further tinkering/saving.
        """

        ws_bins_orig = np.arange(0.0, 31.0, self.ws_bin_width_LT_corr)

        if xlim == (None, None):
            # Default to the range 4 - 20 m/s
            ws_min = 4.0
            ws_max = 20.0
        else:
            ws_min = np.max([0.0, np.floor(xlim[0])])
            ws_max = np.min([ws_bins_orig[-1], np.ceil(xlim[1])])

        ws_bins = np.arange(ws_min, ws_max + 1, self.ws_bin_width_LT_corr)
        mask = (ws_bins_orig >= ws_min) & (ws_bins_orig <= ws_max)

        if turbine_id is None:
            if self.UQ:
                efficiency_data_por = self.wake_losses_por_ws[:, mask]
                efficiency_data_lt = self.wake_losses_lt_ws[:, mask]
            else:
                efficiency_data_por = self.wake_losses_por_ws[mask]
                efficiency_data_lt = self.wake_losses_lt_ws[mask]

        else:
            turbine_index = self.turbine_ids.index(turbine_id)

            if self.UQ:
                efficiency_data_por = self.turbine_wake_losses_por_ws[:, turbine_index, mask]
                efficiency_data_lt = self.turbine_wake_losses_lt_ws[:, turbine_index, mask]
            else:
                efficiency_data_por = self.turbine_wake_losses_por_ws[turbine_index, mask]
                efficiency_data_lt = self.turbine_wake_losses_lt_ws[turbine_index, mask]

        if plot_norm_energy:
            if self.UQ:
                energy_data_por = self.energy_por_ws[:, mask]
                energy_data_lt = self.energy_lt_ws[:, mask]
            else:
                energy_data_por = self.energy_por_ws[mask]
                energy_data_lt = self.energy_lt_ws[mask]
        else:
            energy_data_por = None
            energy_data_lt = None

        return plot.plot_wake_losses(
            bins=ws_bins,
            efficiency_data_por=efficiency_data_por,
            efficiency_data_lt=efficiency_data_lt,
            energy_data_por=energy_data_por,
            energy_data_lt=energy_data_lt,
            bin_axis_label=r"Freestream Wind Speed (m/s)",
            turbine_id=turbine_id,
            xlim=xlim,
            ylim_efficiency=ylim_efficiency,
            ylim_energy=ylim_energy,
            return_fig=return_fig,
            figure_kwargs=figure_kwargs,
            plot_kwargs_line=plot_kwargs_line,
            plot_kwargs_fill=plot_kwargs_fill,
            legend_kwargs=legend_kwargs,
        )


__defaults_wind_direction_col = WakeLosses.__attrs_attrs__.wind_direction_col.default
__defaults_wind_direction_data_type = WakeLosses.__attrs_attrs__.wind_direction_data_type.default
__defaults_wind_direction_asset_ids = WakeLosses.__attrs_attrs__.wind_direction_asset_ids.default
__defaults_UQ = WakeLosses.__attrs_attrs__.UQ.default
__defaults_num_sim = WakeLosses.__attrs_attrs__.num_sim.default
__defaults_start_date = WakeLosses.__attrs_attrs__.start_date.default
__defaults_end_date = WakeLosses.__attrs_attrs__.end_date.default
__defaults_reanalysis_products = WakeLosses.__attrs_attrs__.reanalysis_products.default
__defaults_end_date_lt = WakeLosses.__attrs_attrs__.end_date_lt.default
__defaults_wd_bin_width = WakeLosses.__attrs_attrs__.wd_bin_width.default
__defaults_freestream_sector_width = WakeLosses.__attrs_attrs__.freestream_sector_width.default
__defaults_freestream_power_method = WakeLosses.__attrs_attrs__.freestream_power_method.default
__defaults_freestream_wind_speed_method = (
    WakeLosses.__attrs_attrs__.freestream_wind_speed_method.default
)
__defaults_correct_for_derating = WakeLosses.__attrs_attrs__.correct_for_derating.default
__defaults_derating_filter_wind_speed_start = (
    WakeLosses.__attrs_attrs__.derating_filter_wind_speed_start.default
)
__defaults_max_power_filter = WakeLosses.__attrs_attrs__.max_power_filter.default
__defaults_wind_bin_mad_thresh = WakeLosses.__attrs_attrs__.wind_bin_mad_thresh.default
__defaults_wd_bin_width_LT_corr = WakeLosses.__attrs_attrs__.wd_bin_width_LT_corr.default
__defaults_ws_bin_width_LT_corr = WakeLosses.__attrs_attrs__.ws_bin_width_LT_corr.default
__defaults_num_years_LT = WakeLosses.__attrs_attrs__.num_years_LT.default
__defaults_assume_no_wakes_high_ws_LT_corr = (
    WakeLosses.__attrs_attrs__.assume_no_wakes_high_ws_LT_corr.default
)
__defaults_no_wakes_ws_thresh_LT_corr = (
    WakeLosses.__attrs_attrs__.no_wakes_ws_thresh_LT_corr.default
)


def create_WakeLosses(
    project: PlantData,
    wind_direction_col: str = __defaults_wind_direction_col,
    wind_direction_data_type: str = __defaults_wind_direction_data_type,
    wind_direction_asset_ids: list[str] = __defaults_wind_direction_asset_ids,
    UQ: bool = __defaults_UQ,
    num_sim: int = __defaults_num_sim,
    start_date: str | pd.Timestamp = __defaults_start_date,
    end_date: str | pd.Timestamp = __defaults_end_date,
    reanalysis_products: list[str] = __defaults_reanalysis_products,
    end_date_lt: str | pd.Timestamp = __defaults_end_date_lt,
    wd_bin_width: float = __defaults_wd_bin_width,
    freestream_sector_width: float = __defaults_freestream_sector_width,
    freestream_power_method: str = __defaults_freestream_power_method,
    freestream_wind_speed_method: str = __defaults_freestream_wind_speed_method,
    correct_for_derating: bool = __defaults_correct_for_derating,
    derating_filter_wind_speed_start: float = __defaults_derating_filter_wind_speed_start,
    max_power_filter: float = __defaults_max_power_filter,
    wind_bin_mad_thresh: float = __defaults_wind_bin_mad_thresh,
    wd_bin_width_LT_corr: float = __defaults_wd_bin_width_LT_corr,
    ws_bin_width_LT_corr: float = __defaults_ws_bin_width_LT_corr,
    num_years_LT: int = __defaults_num_years_LT,
    assume_no_wakes_high_ws_LT_corr: bool = __defaults_assume_no_wakes_high_ws_LT_corr,
    no_wakes_ws_thresh_LT_corr: float = __defaults_no_wakes_ws_thresh_LT_corr,
) -> WakeLosses:
    return WakeLosses(
        plant=project,
        wind_direction_col=wind_direction_col,
        wind_direction_data_type=wind_direction_data_type,
        wind_direction_asset_ids=wind_direction_asset_ids,
        UQ=UQ,
        num_sim=num_sim,
        start_date=start_date,
        end_date=end_date,
        reanalysis_products=reanalysis_products,
        end_date_lt=end_date_lt,
        wd_bin_width=wd_bin_width,
        freestream_sector_width=freestream_sector_width,
        freestream_power_method=freestream_power_method,
        freestream_wind_speed_method=freestream_wind_speed_method,
        correct_for_derating=correct_for_derating,
        derating_filter_wind_speed_start=derating_filter_wind_speed_start,
        max_power_filter=max_power_filter,
        wind_bin_mad_thresh=wind_bin_mad_thresh,
        wd_bin_width_LT_corr=wd_bin_width_LT_corr,
        ws_bin_width_LT_corr=ws_bin_width_LT_corr,
        num_years_LT=num_years_LT,
        assume_no_wakes_high_ws_LT_corr=assume_no_wakes_high_ws_LT_corr,
        no_wakes_ws_thresh_LT_corr=no_wakes_ws_thresh_LT_corr,
    )


create_WakeLosses.__doc__ = WakeLosses.__doc__
