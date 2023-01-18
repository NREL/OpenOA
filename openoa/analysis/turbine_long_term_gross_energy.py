"""
This class defines key analytical routines for performing a gap analysis on
EYA-estimated annual energy production (AEP) and that from operational data. Categories
considered are availability, electrical losses, and long-term gross energy. The main
output is a 'waterfall' plot linking the EYA-estimated and operational-estiamted AEP values.
"""

from __future__ import annotations

import random
from typing import Callable

import attrs
import numpy as np
import pandas as pd
import numpy.typing as npt
import matplotlib.pyplot as plt
from tqdm import tqdm
from attrs import field, define
from matplotlib.ticker import StrMethodFormatter

from openoa.plant import PlantData, FromDictMixin
from openoa.utils import plot, filters, imputing
from openoa.utils import timeseries as ts
from openoa.utils import met_data_processing as met
from openoa.logging import logging, logged_method_call
from openoa.utils.power_curve import functions
from openoa.analysis._analysis_validators import validate_UQ_input, validate_open_range_0_1


logger = logging.getLogger(__name__)
plot.set_styling()

NDArrayFloat = npt.NDArray[np.float64]

MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24


@define(auto_attribs=True)
class TurbineLongTermGrossEnergy(FromDictMixin):
    """
    Calculates long-term gross energy for each turbine in a wind farm using methods implemented in
    the utils subpackage for data processing and analysis.

    The method proceeds as follows:

        1. Filter turbine data for normal operation
        2. Calculate daily means of wind speed, wind direction, and air density from reanalysis products
        3. Calculate daily sums of energy from each turbine
        4. Fit daily data (features are atmospheric variables, response is turbine power) using a
           generalized additive model (GAM)
        5. Apply model results to long-term atmospheric varaibles to calculate long term
           gross energy for each turbine

    A Monte Carlo approach is implemented to obtain distribution of results, from which uncertainty
    can be quantified for the long-term gross energy estimate. A pandas DataFrame of long-term gross
    energy values is produced, containing each turbine in the wind farm. Note that this gross energy
    metric does not back out losses associated with waking or turbine performance. Rather, gross
    energy in this context is what turbine would have produced under normal operation (i.e.
    excluding downtime and underperformance).

    Required schema of PlantData:

        - _scada_freq
        - reanalysis products with columns ['time', 'WMETR_HorWdSpdU', 'WMETR_HorWdSpdV', 'WMETR_HorWdSpd', 'WMETR_AirDen']
        - scada with columns: ['time', 'WTUR_TurNam', 'WMET_HorWdSpd', 'WTUR_W', 'WTUR_SupWh']

    Args:
        UQ(:obj:`bool`): Indicator to perform (True) or not (False) uncertainty quantification.
        wind_bin_threshold(:obj:`tuple`): The filter threshold for each vertical bin, expressed as
            number of standard deviations from the median in each bin. When :py:attr:`UQ` is True, then this should be a
            tuple of the lower and upper limits of this threshold, otherwise a single value should be used. Defaults to
            (1.0, 3.0)
        max_power_filter(:obj:`tuple`): Maximum power threshold, in the range (0, 1], to which the bin
            filter should be applied. When :py:attr:`UQ` is True, then this should be a tuple of the lower and upper
            limits of this filter, otherwise a single value should be used. Defaults to (0.8, 0.9).
        correction_threshold(:obj:`tuple`): The threshold, in the range of (0, 1], above which daily scada
            energy data should be corrected. When :py:attr:`UQ` is True, then this should be a tuple of the lower and
            upper limits of this threshold, otherwise a single value should be used. Defaults to (0.85, 0.95)
        uncertainty_scada(:obj:`float`): Uuncertainty imposed to the SCADA data when :py:attr:`UQ` is True only Defaults
            to 0.005.
        num_sim(:obj:`int`): Number of simulations to run when `UQ` is True, otherwise set to 1. Defaults to 20000.
    """

    plant: PlantData = field(validator=attrs.validators.instance_of(PlantData))
    UQ: bool = field(default=True, converter=bool)
    wind_bin_threshold: NDArrayFloat = field(
        default=(1.0, 3.0), validator=validate_UQ_input, on_setattr=None
    )
    max_power_filter: NDArrayFloat = field(
        default=(0.8, 0.9), validator=(validate_UQ_input, validate_open_range_0_1), on_setattr=None
    )
    correction_threshold: NDArrayFloat = field(
        default=(0.85, 0.95),
        validator=(validate_UQ_input, validate_open_range_0_1),
        on_setattr=None,
    )
    uncertainty_scada: float = field(default=0.005, converter=float)
    num_sim: int = field(default=20000, converter=int)

    # Internally created attributes need to be given a type before usage
    por_start: pd.Timestamp = field(init=False)
    por_end: pd.Timestamp = field(init=False)
    turbine_ids: np.ndarray = field(init=False)
    scada: pd.DataFrame = field(init=False)
    scada_dict: dict = field(factory=dict, init=False)
    daily_reanal_dict: dict = field(factory=dict, init=False)
    model_dict: dict = field(factory=dict, init=False)
    model_results: dict = field(factory=dict, init=False)
    scada_daily_valid: pd.DataFrame = field(default=pd.DataFrame(), init=False)
    reanalysis_subset: list[str] = field(init=False)
    reanalysis_memo: dict[str, pd.DataFrame] = field(factory=dict, init=False)
    daily_reanalysis: dict[str, pd.DataFrame] = field(factory=dict, init=False)
    _run: pd.DataFrame = field(init=False)
    _inputs: pd.DataFrame = field(init=False)
    scada_valid: pd.DataFrame = field(init=False)
    turbine_model_dict: dict[str, pd.DataFrame] = field(factory=dict, init=False)
    _model_results: dict[str, Callable] = field(factory=dict, init=False)
    turb_lt_gross: pd.DataFrame = field(default=pd.DataFrame(), init=False)
    summary_results: pd.DataFrame = field(init=False)
    plant_gross: dict[int, pd.DataFrame] = field(factory=dict, init=False)

    @plant.validator
    def validate_plant_ready_for_anylsis(
        self, attribute: attrs.Attribute, value: PlantData
    ) -> None:
        """Validates that the value has been validated for a turbine long term gross energy analysis."""
        if set(("TurbineLongTermGrossEnergy", "all")).intersection(value.analysis_type) == set():
            raise TypeError(
                "The input to 'plant' must be validated for at least the 'TurbineLongTermGrossEnergy'"
            )

    @logged_method_call
    def __attrs_post_init__(self):

        """
        Runs any non-automated setup steps for the analysis class.
        """
        logger.info("Initializing TurbineLongTermGrossEnergy Object")

        # Check that selected UQ is allowed
        if self.UQ:
            logger.info("Note: uncertainty quantification will be performed in the calculation")
        else:
            logger.info("Note: uncertainty quantification will NOT be performed in the calculation")
            self.num_sim = 1
        self.turbine_ids = self.plant.turbine_ids

        # Get start and end of POR days in SCADA
        self.por_start = self.plant.scada.index.get_level_values("time").min()
        self.por_end = self.plant.scada.index.get_level_values("time").max()

        # Initially sort the different turbine data into dictionary entries
        logger.info("Processing SCADA data into dictionaries by turbine (this can take a while)")
        self.sort_scada_by_turbine()

    @logged_method_call
    def run(
        self,
        reanalysis_subset=["erai", "ncep2", "merra2"],
    ) -> None:
        """
        Pre-process the run-specific data settings for each simulation, then fit and apply the
        model for each simualtion.

        Args:
            reanalysis_subset(:obj:`list`): The reanalysis products to use for long-term correction.
        """
        self.reanalysis_subset = reanalysis_subset  # Reanalysis data to consider in fitting

        self.setup_inputs()
        logger.info("Running the long term gross energy analysis")

        # Loop through number of simulations, store TIE results
        for i in tqdm(np.arange(self.num_sim)):

            self._run = self._inputs.loc[i]

            self.filter_turbine_data()  # Filter turbine data
            self.setup_daily_reanalysis_data()  # Setup daily reanalysis products
            self.filter_sum_impute_scada()  # Setup daily scada data
            self.setupturbine_model_dict()  # Setup daily data to be fit using the GAM
            self.fit_model()  # Fit daily turbine energy to atmospheric data
            self.apply_model(i)  # Apply fitting result to long-term reanalysis data

        # Log the completion of the run
        logger.info("Run completed")

    def setup_inputs(self) -> None:
        """
        Create and populate the data frame defining the simulation parameters.
        This data frame is stored as self._inputs
        """
        if self.UQ:
            reanal_list = list(
                np.repeat(self.reanalysis_subset, self.num_sim)
            )  # Create extra long list of renanalysis product names to sample from
            inputs = {
                "reanalysis_product": np.asarray(random.sample(reanal_list, self.num_sim)),
                "scada_data_fraction": np.random.normal(1, self.uncertainty_scada, self.num_sim),
                "wind_bin_thresh": np.random.randint(
                    self.wind_bin_threshold[0] * 100,
                    self.wind_bin_threshold[1] * 100,
                    self.num_sim,
                )
                / 100.0,
                "max_power_filter": np.random.randint(
                    self.max_power_filter[0] * 100,
                    self.max_power_filter[1] * 100,
                    self.num_sim,
                )
                / 100.0,
                "correction_threshold": np.random.randint(
                    self.correction_threshold[0] * 100,
                    self.correction_threshold[1] * 100,
                    self.num_sim,
                )
                / 100.0,
            }
            self.plant_gross = np.empty([self.num_sim, 1])

        if not self.UQ:
            inputs = {
                "reanalysis_product": self.reanalysis_subset,
                "scada_data_fraction": 1,
                "wind_bin_thresh": self.wind_bin_threshold,
                "max_power_filter": self.max_power_filter,
                "correction_threshold": self.correction_threshold,
            }
            self.plant_gross = np.empty([len(self.reanalysis_subset), 1])
            self.num_sim = len(self.reanalysis_subset)

        self._inputs = pd.DataFrame(inputs)

    def prepare_scada(self) -> None:
        """
        Performs the following manipulations:
         1. Creates a copy of the SCADA data
         2. Sorts it by turbine ID, then timestamp (the two index columns)
         3. Drops any rows that don't have any windspeed or energy data
         4. Flags windspeed values outside the range [0, 40]
         5. Flags windspeed values that have stayed the same for at least 3 straight readings
         6. Flags power values outside the range [0, turbine capacity]
         7. Flags windspeed and power values that don't mutually coincide within a reasonable range
         8. Combine the flags using an "or" combination to be a new column in scada: "valid"
        """
        self.scada = (
            self.plant.scada.swaplevel().sort_index().dropna(subset=["WMET_HorWdSpd", "WTUR_SupWh"])
        )
        turbine_capacity = self.scada.groupby(level="WTUR_TurNam").max()["WTUR_W"]
        flag_range = filters.range_flag(self.scada.loc[:, "WMET_HorWdSpd"], below=0, above=40)
        flag_frozen = filters.unresponsive_flag(self.scada.loc[:, "WMET_HorWdSpd"], threshold=3)
        flag_neg = pd.Series(index=self.scada.index, dtype=bool)
        flag_window = pd.Series(index=self.scada.index, dtype=bool)
        for t in self.turbine_ids:
            ix_turb = self.scada.index.get_level_values("WTUR_TurNam") == t
            flag_neg.loc[ix_turb] = filters.range_flag(
                self.scada.loc[ix_turb, "power"], below=0, above=turbine_capacity.loc[t]
            )
            flag_window.loc[ix_turb] = filters.window_range_flag(
                window_col=self.scada.loc[ix_turb, "WMET_HorWdSpd"],
                window_start=5.0,
                window_end=40,
                value_col=self.scada.loc[ix_turb, "WTUR_W"],
                value_min=0.02 * turbine_capacity.loc[t],
                value_max=1.2 * turbine_capacity.loc[t],
            )

        flag_final = ~(flag_range | flag_frozen | flag_neg | flag_window).values
        self.scada.assign(valid=flag_final)
        self.scada.assign(valid_run=flag_final)

    def sort_scada_by_turbine(self) -> None:
        """
        Sorts the SCADA DataFrame by the ID and timestamp index columns, respectively.
        """

        df = self.plant.scada.copy()
        dic = self.scada_dict

        # Loop through turbine IDs
        for t in self.turbine_ids:
            # Store relevant variables in dictionary
            dic[t] = df.loc[df.index.get_level_values("WTUR_TurNam") == t].reindex(
                columns=["WMET_HorWdSpd", "WTUR_W", "WTUR_SupWh"]
            )
            dic[t].sort_index(inplace=True)

    def filter_turbine_data(self) -> None:
        """
        Apply a set of filtering algorithms to the turbine wind speed vs power curve to flag
        data not representative of normal turbine operation
        """

        dic = self.scada_dict

        # Loop through turbines
        for t in self.turbine_ids:
            turb_capac = dic[t]["WTUR_W"].max()

            max_bin = (
                self._run.max_power_filter * turb_capac
            )  # Set maximum range for using bin-filter

            dic[t].dropna(
                subset=["WMET_HorWdSpd", "WTUR_SupWh"], inplace=True
            )  # Drop any data where scada wind speed or energy is NaN

            # Flag turbine energy data less than zero
            dic[t].loc[:, "flag_neg"] = filters.range_flag(
                dic[t].loc[:, "WTUR_W"], lower=0, upper=turb_capac
            )
            # Apply range filter
            dic[t].loc[:, "flag_range"] = filters.range_flag(
                dic[t].loc[:, "WMET_HorWdSpd"], lower=0, upper=40
            )
            # Apply frozen/unresponsive sensor filter
            dic[t].loc[:, "flag_frozen"] = filters.unresponsive_flag(
                dic[t].loc[:, "WMET_HorWdSpd"], threshold=3
            )
            # Apply window range filter
            dic[t].loc[:, "flag_window"] = filters.window_range_flag(
                window_col=dic[t].loc[:, "WMET_HorWdSpd"],
                window_start=5.0,
                window_end=40,
                value_col=dic[t].loc[:, "WTUR_W"],
                value_min=0.02 * turb_capac,
                value_max=1.2 * turb_capac,
            )

            threshold_wind_bin = self._run.wind_bin_thresh
            # Apply bin-based filter
            dic[t].loc[:, "flag_bin"] = filters.bin_filter(
                bin_col=dic[t].loc[:, "WTUR_W"],
                value_col=dic[t].loc[:, "WMET_HorWdSpd"],
                bin_width=0.06 * turb_capac,
                threshold=threshold_wind_bin,  # wind bin thresh
                center_type="median",
                bin_min=np.round(0.01 * turb_capac),
                bin_max=np.round(max_bin),
                threshold_type="std",
                direction="all",
            )

            # Create a 'final' flag which is true if any of the previous flags are true
            dic[t].loc[:, "flag_final"] = (
                (dic[t].loc[:, "flag_range"])
                | (dic[t].loc[:, "flag_window"])
                | (dic[t].loc[:, "flag_bin"])
                | (dic[t].loc[:, "flag_frozen"])
            )

            # Set negative turbine data to zero
            dic[t].loc[dic[t]["flag_neg"], "WTUR_W"] = 0

    def setup_daily_reanalysis_data(self) -> None:
        """
        Process reanalysis data to daily means for later use in the GAM model.
        """
        # Memoize the function so you don't have to recompute the same reanalysis product twice
        if (df_daily := self.reanalysis_memo.get(self._run.reanalysis_product, None)) is not None:
            self.daily_reanalysis = df_daily.copy()
            return

        # Capture the runs reanalysis data set and ensure the U/V components exist
        reanalysis_df = self.plant.reanalysis[self._run.reanalysis_product]
        if len(set(("WMETR_HorWdSpdU", "WMETR_HorWdSpdV")).intersection(reanalysis_df.columns)) < 2:
            (
                reanalysis_df["WMETR_HorWdSpdU"],
                reanalysis_df["WMETR_HorWdSpdV"],
            ) = met.compute_u_v_components("WMETR_HorWdSpd", "WMETR_HorWdDir", reanalysis_df)

        # Resample at a daily resolution and recalculate daily average wind direction
        df_daily = reanalysis_df.groupby([pd.Grouper(freq="D", level="time")])[
            ["WMETR_HorWdSpdU", "WMETR_HorWdSpdV", "WMETR_HorWdSpd", "WMETR_AirDen"]
        ].mean()
        wd = met.compute_wind_direction(u="WMETR_HorWdSpdU", v="WMETR_HorWdSpdV", data=df_daily)
        df_daily = df_daily.assign(WMETR_HorWdDir=wd.values)
        self.daily_reanalysis = df_daily

        # Store the results for re-use
        self.reanalysis_memo[self._run.reanalysis_product] = df_daily

    def filter_sum_impute_scada(self) -> None:
        """
        Filter SCADA data for unflagged data, gather SCADA energy data into daily sums, and correct daily summed
        energy based on amount of missing data and a threshold limit. Finally impute missing data for each turbine
        based on reported energy data from other highly correlated turbines.
        threshold
        """

        scada = self.scada_dict
        expected_count = (
            HOURS_PER_DAY
            * MINUTES_PER_HOUR
            / (ts.offset_to_seconds(self.plant.metadata.scada.frequency) / 60)
        )
        num_thres = self._run.correction_threshold * expected_count  # Allowable reported timesteps

        self.scada_valid = pd.DataFrame()

        # Loop through turbines
        for t in self.turbine_ids:
            scada_filt = scada[t].loc[~scada[t]["flag_final"]]  # Filter for valid data
            # Calculate daily energy sum
            scada_daily = (
                scada_filt.groupby([pd.Grouper(freq="D", level="time")])["WTUR_SupWh"]
                .sum()
                .to_frame()
            )

            # Count number of entries in sum
            scada_daily["data_count"] = (
                scada_filt.groupby([pd.Grouper(freq="D", level="time")])["WTUR_SupWh"]
                .count()
                .to_frame()
            )
            scada_daily["percent_nan"] = (
                scada_filt.groupby([pd.Grouper(freq="D", level="time")])["WTUR_SupWh"]
                .apply(ts.percent_nan)
                .to_frame()
            )

            # Correct energy for missing data
            scada_daily["energy_corrected"] = (
                scada_daily["WTUR_SupWh"] * expected_count / scada_daily["data_count"]
            )

            # Discard daily sums if less than 140 data counts (90% reported data)
            scada_daily = scada_daily.loc[scada_daily["data_count"] >= num_thres]

            # Create temporary data frame that is gap filled and to be used for imputing
            temp_df = pd.DataFrame(
                index=pd.date_range(self.por_start, self.por_end, freq="D", name="time")
            )
            temp_df["energy_corrected"] = scada_daily["energy_corrected"]
            temp_df["percent_nan"] = scada_daily["percent_nan"]
            temp_df["WTUR_TurNam"] = np.repeat(t, temp_df.shape[0])
            temp_df["day"] = temp_df.index

            # Append turbine data into single data frame for imputing
            self.scada_valid = self.scada_valid.append(temp_df)

        # Reset index after all turbines has been combined
        self.scada_valid = self.scada_valid.set_index("WTUR_TurNam", append=True)

        # Impute missing days for each turbine - provides progress bar
        self.scada_valid["energy_imputed"] = imputing.impute_all_assets_by_correlation(
            self.scada_valid,
            impute_col="energy_corrected",
            reference_col="energy_corrected",
        )

        # Drop data that could not be imputed
        self.scada_valid.dropna(subset=["energy_imputed"], inplace=True)

    def setupturbine_model_dict(self) -> None:
        """Setup daily atmospheric variable averages and daily energy sums by turbine."""
        reanalysis = self.daily_reanalysis
        for t in self.turbine_ids:
            self.turbine_model_dict[t] = (
                self.scada_valid.loc[self.scada_valid.index.get_level_values("WTUR_TurNam") == t]
                .set_index("day")
                .join(reanalysis)
                .dropna(subset=["energy_imputed", "WMETR_HorWdSpd"])
            )

    def fit_model(self) -> None:
        """Fit the daily turbine energy sum and atmospheric variable averages using a GAM model
        using wind speed, wind direction, and air density.
        """

        mod_dict = self.turbine_model_dict
        mod_results = self._model_results

        for t in self.turbine_ids:  # Loop throuh turbines
            df = mod_dict[t]

            # Add Monte-Carlo sampled uncertainty to SCADA data
            df["energy_imputed"] = df["energy_imputed"] * self._run.scada_data_fraction

            # Consider wind speed, wind direction, and air density as features
            mod_results[t] = functions.gam_3param(
                windspeed_col="WMETR_HorWdSpd",
                wind_direction_col="WMETR_HorWdDir",
                air_density_col="WMETR_AirDen",
                power_col="energy_imputed",
                data=df,
            )
        self._model_results = mod_results

    def apply_model(self, i: int) -> None:
        """
        Apply the model to the reanalysis data to calculate long-term gross energy for each turbine.

        Args:
            i(:obj:`int`): The Monte Carlo iteration number.
        """
        turb_gross = self.turb_lt_gross
        mod_results = self._model_results

        # Create a data frame to store final results
        self.summary_results = pd.DataFrame(index=self.reanalysis_subset, columns=self.turbine_ids)

        daily_reanalysis = self.daily_reanalysis
        turb_gross = pd.DataFrame(index=daily_reanalysis.index)

        # Loop through the turbines and apply the GAM to the reanalysis data
        for t in self.turbine_ids:  # Loop through turbines
            turb_gross.loc[:, t] = mod_results[t](
                daily_reanalysis["WMETR_HorWdSpd"],
                daily_reanalysis["WMETR_HorWdDir"],
                daily_reanalysis["WMETR_AirDen"],
            )

        turb_gross[turb_gross < 0] = 0

        # Calculate monthly sums of energy from long-term estimate
        turb_mo = turb_gross.resample("MS").sum()

        # Get average sum by calendar month
        turb_mo_avg = turb_mo.groupby(turb_mo.index.month).mean()

        # Store sum of turbine gross energy
        self.plant_gross[i] = turb_mo_avg.sum(axis=1).sum(axis=0)
        self.turb_lt_gross = turb_gross

    def plot_filtered_power_curves(
        self,
        turbines: list[str] | None = None,
        flag_labels: tuple[str, str] = None,
        max_cols: int = 3,
        xlim: tuple[float, float] = (None, None),
        ylim: tuple[float, float] = (None, None),
        legend: bool = False,
        return_fig: bool = False,
        figure_kwargs: dict = {},
        legend_kwargs: dict = {},
        plot_kwargs: dict = {},
    ):
        """Plot the raw and flagged power curve data.

        Args:
            turbines(:obj:`list[str]`, optional): The list of turbines to be plot, if not all of the
                keys in :py:attr:`data`.
            flag_labels (:obj:`tuple[str, str]`, optional): The labels to give to the scatter points,
                where the first entryis the flagged points, and the second entry correpsponds to the
                standard power curve. Defaults to None.
            max_cols(:obj:`int`, optional): The maximum number of columns in the plot. Defaults to 3.
            xlim(:obj:`tuple[float, float]`, optional): A tuple of the x-axis (min, max) values.
                Defaults to (None, None).
            ylim(:obj:`tuple[float, float]`, optional): A tuple of the y-axis (min, max) values.
                Defaults to (None, None).
            legend(:obj:`bool`, optional): Set to True to place a legend in the figure, otherwise set
                to False. Defaults to False.
            return_fig(:obj:`bool`, optional): Set to True to return the figure and axes objects,
                otherwise set to False. Defaults to False.
            figure_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be passed
                to `plt.figure`. Defaults to {}.
            plot_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be passed
                to `ax.scatter`. Defaults to {}.
            legend_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be passed
                to `ax.legend`. Defaults to {}.

        Returns:
            None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]: If `return_fig` is True, then
                the figure and axes objects are returned for further tinkering/saving.
        """
        return plot.plot_power_curves(
            data=self.scada_dict,
            windspeed_col="WMET_HorWdSpd",
            power_col="WTUR_W",
            flag_col="flag_final",
            turbines=turbines,
            flag_labels=flag_labels,
            max_cols=max_cols,
            xlim=xlim,
            ylim=ylim,
            legend=legend,
            return_fig=return_fig,
            figure_kwargs=figure_kwargs,
            legend_kwargs=legend_kwargs,
            plot_kwargs=plot_kwargs,
        )

    def plot_daily_fitting_result(
        self,
        turbines: list[str] | None = None,
        flag_labels: tuple[str, str, str] = ("Modeled", "Imputed", "Input"),
        max_cols: int = 3,
        xlim: tuple[float, float] = (None, None),
        ylim: tuple[float, float] = (None, None),
        legend: bool = False,
        return_fig: bool = False,
        figure_kwargs: dict = {},
        legend_kwargs: dict = {},
        plot_kwargs: dict = {},
    ):
        """Plot the raw, imputed, and modeled power curve data.

        Args:
            turbines(:obj:`list[str]`, optional): The list of turbines to be plot, if not all of the
                keys in :py:attr:`data`.
            labels (:obj:`tuple[str, str]`, optional): The labels to give to the scatter points,
                corresponding to the modeled, imputed, and input data, respectively. Defaults to
                ("Modeled", "Imputed", "Input").
            max_cols(:obj:`int`, optional): The maximum number of columns in the plot. Defaults to 3.
            xlim(:obj:`tuple[float, float]`, optional): A tuple of the x-axis (min, max) values.
                Defaults to (None, None).
            ylim(:obj:`tuple[float, float]`, optional): A tuple of the y-axis (min, max) values.
                Defaults to (None, None).
            legend(:obj:`bool`, optional): Set to True to place a legend in the figure, otherwise set
                to False. Defaults to False.
            return_fig(:obj:`bool`, optional): Set to True to return the figure and axes objects,
                otherwise set to False. Defaults to False.
            figure_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be passed
                to `plt.figure`. Defaults to {}.
            plot_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be passed
                to `ax.scatter`. Defaults to {}.
            legend_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be passed
                to `ax.legend`. Defaults to {}.

        Returns:
            None | tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]: If `return_fig` is True, then
                the figure and axes objects are returned for further tinkering/saving.
        """
        turbines = list(self.turbine_model_dict.keys()) if turbines is None else turbines
        num_cols = len(turbines)
        num_rows = int(np.ceil(num_cols / max_cols))

        if flag_labels is None:
            flag_labels = ("Modeled", "Imputed", "Input")

        figure_kwargs.setdefault("dpi", 200)
        figure_kwargs.setdefault("figsize", (15, num_rows * 5))
        fig, axes_list = plt.subplots(num_rows, max_cols, **figure_kwargs)

        ws_daily = self.daily_reanalysis["WMETR_HorWdSpd"]
        for i, (t, ax) in enumerate(zip(turbines, axes_list.flatten())):
            df = self.turbine_model_dict[t]
            df_imputed = df.loc[df["energy_corrected"] != df["energy_imputed"]]

            ax.scatter(
                ws_daily,
                self.turb_lt_gross[t],
                label=flag_labels[0],
                alpha=0.2,
                color="tab:blue",
                **plot_kwargs,
            )
            ax.scatter(
                df["WMETR_HorWdSpd"],
                df["energy_imputed"],
                label=flag_labels[2],
                alpha=0.6,
                color="tab:green",
                **plot_kwargs,
            )
            ax.scatter(
                df_imputed["WMETR_HorWdSpd"],
                df_imputed["energy_imputed"],
                label=flag_labels[1],
                alpha=0.6,
                color="tab:orange",
                **plot_kwargs,
            )

            ax.set_title(t)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

            if legend:
                ax.legend(**legend_kwargs)

            if i % max_cols == 0:
                ax.set_ylabel("Power (kW)")

            if i in range(max_cols * (num_rows - 1), num_cols):
                ax.set_xlabel("Wind Speed (m/s)")

        num_axes = axes_list.size
        if i < num_axes - 1:
            for j in range(i + 1, num_axes):
                fig.delaxes(axes_list.flatten()[j])

        fig.tight_layout()
        plt.show()
        if return_fig:
            return fig, ax
