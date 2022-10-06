"""
This class defines key analytical routines for performing a gap analysis on
EYA-estimated annual energy production (AEP) and that from operational data. Categories
considered are availability, electrical losses, and long-term gross energy. The main
output is a 'waterfall' plot linking the EYA-estimated and operational-estiamted AEP values.
"""

from __future__ import annotations

import random
from typing import Callable
from lib2to3.pytree import convert

import attrs
import numpy as np
import pandas as pd
import numpy.typing as npt
from tqdm import tqdm
from attrs import field, define

from openoa import logging, logged_method_call
from openoa.plant import PlantData, FromDictMixin
from openoa.utils import filters, imputing
from openoa.utils import timeseries as ts
from openoa.utils import met_data_processing as met
from openoa.utils.power_curve import functions


logger = logging.getLogger(__name__)

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
        - reanalysis products ['merra2', 'erai', 'ncep2'] with columns ['time', 'u_ms', 'v_ms', 'windspeed_ms', 'rho_kgm-3']
        - scada with columns: ['time', 'id', 'wmet_wdspd_avg', 'wtur_W_avg', 'energy_kwh']

    Args:
        uncertainty_scada(:obj:`float`): uncertainty imposed to scada data (used in UQ = True case only)
        max_power_filter(:obj:`tuple`): Maximum power threshold (fraction) to which the bin filter
        should be applied (default is the interval between 0.8 and 0.9). This should be a tuple in
        the UQ = True case (the values are Monte-Carlo sampled), a single value when UQ = False.
        wind_bin_thresh(:obj:`tuple`): The filter threshold for each vertical bin, expressed as
        number of standard deviations from the median in each bin (default is the interval
        between 1 and 3 stdev). This should be a tuple in the UQ = True case (the values are
        Monte-Carlo sampled), a single value when UQ = False.
        correction_threshold(:obj:`tuple`): The threshold (fraction) above which daily scada energy data
        should be corrected (default is the interval between 0.85 and 0.95). This should be a
        tuple in the UQ = True case (the values are Monte-Carlo sampled), a single value when UQ = False.
    """

    plant: PlantData = field(validator=attrs.validators.instance_of(PlantData))
    UQ: bool = field(default=False, converter=bool)
    uncertainty_scada: float = field(default=0.005, converter=float)
    wind_bin_thresh: tuple[float, float] = field(
        default=(1.0, 3.0),
        converter=tuple,
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(tuple),
            member_validator=attrs.validators.instance_of(float),
        ),
    )
    max_power_filter: tuple[float, float] = field(
        default=(0.8, 0.9),
        converter=tuple,
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(tuple),
            member_validator=attrs.validators.instance_of(float),
        ),
    )
    correction_threshold: tuple[float, float] = field(
        default=(0.85, 0.95),
        converter=tuple,
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(tuple),
            member_validator=attrs.validators.instance_of(float),
        ),
    )
    num_sim: int = field(default=20000, converter=int)

    # Internally created attributes need to be given a type before usage
    start_por: pd.Timestamp = field(init=False)
    end_por: pd.Timestamp = field(init=False)

    scada: pd.DataFrame = field(init=False)
    scada_dict: dict = field(factory=dict, init=False)
    daily_reanal_dict: dict = field(factory=dict, init=False)
    model_dict: dict = field(factory=dict, init=False)
    model_results: dict = field(factory=dict, init=False)
    turb_lt_gross: dict = field(factory=dict, init=False)
    scada_daily_valid: pd.DataFrame = field(default=pd.DataFrame(), init=False)
    uncertainty_wind_bin_thresh: NDArrayFloat = field(init=False)
    uncertainty_max_power_filter: NDArrayFloat = field(init=False)
    uncertainty_correction_threshold: NDArrayFloat = field(init=False)
    reanalysis_subset: list[str] = field(init=False)
    reanalysis_memo: dict[str, pd.DataFrame] = field(factory=dict, init=False)
    daily_reanalysis: dict[str, pd.DataFrame] = field(factory=dict, intit=False)
    _run: pd.DataFrame = field(init=False)
    scada_valid: pd.DataFrame = field(init=False)
    turbine_model_dict: dict[str, pd.DataFrame] = field(init=False)
    _model_results: dict[str, Callable] = field(init=False)
    _turb_lt_gross: pd.DataFrame = field(init=False)
    _plant_gross: dict[int, pd.DataFrame] = field(init=False)

    @plant.validator
    def validate_plant_ready_for_anylsis(
        self, attribute: attrs.Attribute, value: PlantData
    ) -> None:
        """Validates that the value has been validated for a turbine long term gross energy analysis."""
        if set(("TurbineLongTermGrossEnergy", "all")).intersection(value.analysis_type) == set():
            raise TypeError(
                "The input to 'plant' must be validated for at least the 'TurbineLongTermGrossEnergy'"
            )

    @wind_bin_thresh.validator
    @max_power_filter.validator
    @correction_threshold.validator
    def validate_tuple(self, attribute: attrs.Attribute, value: tuple[float, float]) -> None:
        """Validates that the value is a length-2 tuple."""
        if not isinstance(value, tuple):
            raise TypeError("'{attribute.name}' must be a tuple.")
        if not len(value) == 2:
            raise ValueError("'{attribute.name}' must be a length-2 tuple.")

    @uncertainty_scada.validator
    @max_power_filter.validator
    @correction_threshold.validator
    def validate_decimal_range(self, attribute: attrs.Attribute, value: float | tuple) -> None:
        """Validates that the value is in the range of (0, 1)."""
        if isinstance(value, float):
            if not 0.0 < value < 1.0:
                raise ValueError(f"'{attribute.name}' must be in the range (0, 1).")
        else:
            if not all(0.0 < x < 1.0 for x in value):
                raise ValueError(f"Each value of '{attribute.name}' must be in the range (0, 1).")

    @logged_method_call
    def __attrs_post_init__(self, plant, UQ=True, num_sim=2000):

        """
        Initialize turbine long-term gross energy analysis with data and parameters.

        Args:
         plant(:obj:`PlantData object`): PlantData object from which TurbineLongTermGrossEnergy should draw data.
         UQ:(:obj:`bool`): choice whether to perform (True) or not (False) uncertainty quantification
         num_sim:(:obj:`int`): number of Monte Carlo simulations. Please note that this script is somewhat computationally heavy so the default num_sim value has been adjusted accordingly.
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
        self.por_start = plant._scada.df.index.min()
        self.por_end = plant._scada.df.index.max()

        # Initially sort the different turbine data into dictionary entries
        logger.info("Processing SCADA data into dictionaries by turbine (this can take a while)")
        self.sort_scada_by_turbine()

    @logged_method_call
    def run(
        self,
        reanalysis_subset=["erai", "ncep2", "merra2"],
        enable_plotting=False,
        plot_dir=None,
    ) -> None:
        """
        Perform pre-processing of data into an internal representation for which the analysis can run more quickly.

        Args:
         reanalysis_subset(:obj:`list`): Which reanalysis products to use for long-term correction
         enable_plotting(:obj:`boolean`): Indicate whether to output plots
         plot_dir(:obj:`string`): Location to save figures
        """

        self._reanal = reanalysis_subset  # Reanalysis data to consider in fitting

        # Check uncertainty types
        vars = [self.wind_bin_thresh, self.max_power_filter, self.correction_threshold]
        expected_type = float if not self.UQ else tuple
        for var in vars:
            assert (
                type(var) == expected_type
            ), f"wind_bin_thresh, max_power_filter, correction_threshold must all be {expected_type} for UQ={self.UQ}"

        # Define relevant uncertainties, to be applied in Monte Carlo sampling
        self.uncertainty_wind_bin_thresh = np.array(self.wind_bin_thresh, dtype=np.float64)
        self.uncertainty_max_power_filter = np.array(self.max_power_filter, dtype=np.float64)
        self.uncertainty_correction_threshold = np.array(
            self.correction_threshold, dtype=np.float64
        )

        self.setup_inputs()

        # Loop through number of simulations, store TIE results
        for i in tqdm(np.arange(self.num_sim)):

            self._run = self._inputs.loc[i]

            # MC-sampled parameter in this function!
            logger.info("Filtering turbine data")
            self.filter_turbine_data()  # Filter turbine data

            if self.enable_plotting:
                logger.info("Plotting filtered power curves")
                self.plot_filtered_power_curves(plot_dir)

            # MC-sampled parameter in this function!
            logger.info("Processing reanalysis data to daily averages")
            self.setup_daily_reanalysis_data()  # Setup daily reanalysis products

            # MC-sampled parameter in this function!
            logger.info("Processing scada data to daily sums")
            self.filter_sum_impute_scada()  # Setup daily scada data

            logger.info("Setting up daily data for model fitting")
            self.setupturbine_model_dict()  # Setup daily data to be fit using the GAM

            # MC-sampled parameter in this function!
            logger.info("Fitting model data")
            self.fit_model()  # Fit daily turbine energy to atmospheric data

            logger.info("Applying fitting results to calculate long-term gross energy")
            self.apply_model_to_lt(i)  # Apply fitting result to long-term reanalysis data

            if enable_plotting:
                logger.info("Plotting daily fitted power curves")
                self.plot_daily_fitting_result(plot_dir)  # Setup daily reanalysis products

        # Log the completion of the run
        logger.info("Run completed")

    def setup_inputs(self) -> None:
        """
        Create and populate the data frame defining the simulation parameters.
        This data frame is stored as self._inputs
        """
        if self.UQ:
            reanal_list = list(
                np.repeat(self._reanal, self.num_sim)
            )  # Create extra long list of renanalysis product names to sample from
            inputs = {
                "reanalysis_product": np.asarray(random.sample(reanal_list, self.num_sim)),
                "scada_data_fraction": np.random.normal(1, self.uncertainty_scada, self.num_sim),
                "wind_bin_thresh": np.random.randint(
                    self.uncertainty_wind_bin_thresh[0] * 100,
                    self.uncertainty_wind_bin_thresh[1] * 100,
                    self.num_sim,
                )
                / 100.0,
                "max_power_filter": np.random.randint(
                    self.uncertainty_max_power_filter[0] * 100,
                    self.uncertainty_max_power_filter[1] * 100,
                    self.num_sim,
                )
                / 100.0,
                "correction_threshold": np.random.randint(
                    self.uncertainty_correction_threshold[0] * 100,
                    self.uncertainty_correction_threshold[1] * 100,
                    self.num_sim,
                )
                / 100.0,
            }
            self._plant_gross = np.empty([self.num_sim, 1])

        if not self.UQ:
            inputs = {
                "reanalysis_product": self._reanal,
                "scada_data_fraction": 1,
                "wind_bin_thresh": self.uncertainty_wind_bin_thresh,
                "max_power_filter": self.uncertainty_max_power_filter,
                "correction_threshold": self.uncertainty_correction_threshold,
            }
            self._plant_gross = np.empty([len(self._reanal), 1])
            self.num_sim = len(self._reanal)

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
            self.plant.scada.swaplevel().sort_index().dropna(subset=["windspeed", "energy"])
        )
        turbine_capacity = self.scada.groupby(level="id").max()["power"]
        flag_range = filters.range_flag(self.scada.loc[:, "windpseed"], below=0, above=40)
        flag_frozen = filters.unresponsive_flag(self.scada.loc[:, "windspeed"], threshold=3)
        flag_neg = pd.Series(index=self.scada.index, dtype=bool)
        flag_window = pd.Series(index=self.scada.index, dtype=bool)
        for t in self.turbine_ids:
            ix_turb = self.scada.index.get_level_values("id") == t
            flag_neg.loc[ix_turb] = filters.range_flag(
                self.scada.loc[ix_turb, "power"], below=0, above=turbine_capacity.loc[t]
            )
            flag_window.loc[ix_turb] = filters.window_range_flag(
                window_col=self.scada.loc[ix_turb, "windspeed"],
                window_start=5.0,
                window_end=40,
                value_col=self.scada.loc[ix_turb, "power"],
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

        df = self.plant._scada.df
        dic = self._scada_dict

        # Loop through turbine IDs
        for t in self._turbs:
            # Store relevant variables in dictionary
            dic[t] = df[df["id"] == t].reindex(
                columns=["wmet_wdspd_avg", "wtur_W_avg", "energy_kwh"]
            )
            dic[t].sort_index(inplace=True)

    def filter_turbine_data(self) -> None:
        """
        Apply a set of filtering algorithms to the turbine wind speed vs power curve to flag
        data not representative of normal turbine operation
        """

        dic = self._scada_dict

        # Loop through turbines
        for t in self._turbs:
            turb_capac = dic[t].wtur_W_avg.max()

            max_bin = (
                self._run.max_power_filter * turb_capac
            )  # Set maximum range for using bin-filter

            dic[t].dropna(
                subset=["windspeed", "energy"], inplace=True
            )  # Drop any data where scada wind speed or energy is NaN

            # Flag turbine energy data less than zero
            dic[t].loc[:, "flag_neg"] = filters.range_flag(
                dic[t].loc[:, "power"], below=0, above=turb_capac
            )
            # Apply range filter
            dic[t].loc[:, "flag_range"] = filters.range_flag(
                dic[t].loc[:, "windspeed"], below=0, above=40
            )
            # Apply frozen/unresponsive sensor filter
            dic[t].loc[:, "flag_frozen"] = filters.unresponsive_flag(
                dic[t].loc[:, "windspeed"], threshold=3
            )
            # Apply window range filter
            dic[t].loc[:, "flag_window"] = filters.window_range_flag(
                window_col=dic[t].loc[:, "windspeed"],
                window_start=5.0,
                window_end=40,
                value_col=dic[t].loc[:, "power"],
                value_min=0.02 * turb_capac,
                value_max=1.2 * turb_capac,
            )

            threshold_wind_bin = self._run.wind_bin_thresh
            # Apply bin-based filter
            dic[t].loc[:, "flag_bin"] = filters.bin_filter(
                bin_col=dic[t].loc[:, "power"],
                value_col=dic[t].loc[:, "windspeed"],
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
            dic[t].loc[dic[t]["flag_neg"], "wtur_W_avg"] = 0

    def setup_daily_reanalysis_data(self) -> None:
        """
        Process reanalysis data to daily means for later use in the GAM model.
        """
        # Memoize the function so you don't have to recompute the same reanalysis product twice
        if not hasattr(self, "reanalysis_memo"):
            self.reanalysis_memo = {}
        if self._run.reanalysis_product in self.reanalysis_memo.keys():
            self.daily_reanalysis = self.reanalysis_memo[self._run.reanalysis_product]
            return

        # Capture the runs reanalysis data set and ensure the U/V components exist
        reanalysis_df = self.plant.reanalysis[self._run.reanalysis_product]
        if len(set(("windspeed_u", "windspeed_v")).intersection(reanalysis_df.columns)) < 2:
            reanalysis_df["windspeed_u"], reanalysis_df["windspeed_v"] = met.compute_u_v_components(
                "windspeed", "wind_direction", reanalysis_df
            )

        # Resample at a daily resolution and recalculate daily average wind direction
        df_daily = reanalysis_df.resample("D")[
            "windspeed_u", "windspeed_v", "windspeed", "density"
        ].mean()  # Get daily means
        df_daily["wind_direction"] = met.compute_wind_direction(
            u=df_daily["windspeed_u"], v=df_daily["windspeed_v"]
        )
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

        scada = self._scada_dict
        expected_count = (
            HOURS_PER_DAY
            * MINUTES_PER_HOUR
            / (ts.offset_to_seconds(self.plant.metadata.scada.frequency) / 60)
        )
        num_thres = self._run.correction_threshold * expected_count  # Allowable reported timesteps

        self.scada_valid = pd.DataFrame()

        # Loop through turbines
        for t in self._turbs:
            scada_filt = scada[t].loc[~scada[t]["flag_final"]]  # Filter for valid data
            scada_daily = (
                scada_filt.resample("D")["energy"].sum().to_frame()
            )  # Calculate daily energy sum

            # Count number of entries in sum
            scada_daily["data_count"] = scada_filt.resample("D")["energy"].count()
            scada_daily["percent_nan"] = scada_filt.resample("D")["energy"].apply(ts.percent_nan)

            # Correct energy for missing data
            scada_daily["energy_corrected"] = (
                scada_daily["energy"] * expected_count / scada_daily["data_count"]
            )

            # Discard daily sums if less than 140 data counts (90% reported data)
            scada_daily = scada_daily.loc[scada_daily["data_count"] >= num_thres]

            # Create temporary data frame that is gap filled and to be used for imputing
            temp_df = pd.DataFrame(index=pd.date_range(self.por_start, self.por_end, freq="D"))
            temp_df["energy_corrected"] = scada_daily["energy_corrected"]
            temp_df["percent_nan"] = scada_daily["percent_nan"]
            temp_df["id"] = np.repeat(t, temp_df.shape[0])
            temp_df["day"] = temp_df.index

            # Append turbine data into single data frame for imputing
            self.scada_valid = self.scada_valid.append(temp_df)

        # Reset index after all turbines has been combined
        self.scada_valid.reset_index(inplace=True)

        # Impute missing days for each turbine - provides progress bar
        self.scada_valid["energy_imputed"] = imputing.impute_all_assets_by_correlation(
            self.scada_valid,
            input_col="energy_corrected",
            ref_col="energy_corrected",
            align_col="day",
            id_col="id",
        )

        # Drop data that could not be imputed
        self.scada_valid.dropna(subset=["energy_imputed"], inplace=True)

    def setupturbine_model_dict(self) -> None:
        """Setup daily atmospheric variable averages and daily energy sums by turbine."""
        reanalysis = self.daily_reanalysis
        for t in self._turbs:
            self.turbine_model_dict[t] = (
                self.scada_valid.loc[self.scada_valid.index.get_level_values("id") == t]
                .set_index("day")
                .join(reanalysis)
                .dropna(subset=["energy_imputed", "windspeed_ms"])
            )

    def fit_model(self) -> None:
        """Fit the daily turbine energy sum and atmospheric variable averages using a GAM model
        using wind speed, wind direction, and air density.
        """

        mod_dict = self.turbine_model_dict
        mod_results = self._model_results

        for t in self._turbs:  # Loop throuh turbines
            df = mod_dict[t]

            # Add Monte-Carlo sampled uncertainty to SCADA data
            df["energy_imputed"] = df["energy_imputed"] * self._run.scada_data_fraction

            # Consider wind speed, wind direction, and air density as features
            mod_results[t] = functions.gam_3param(
                windspeed_col="windspeed_ms",
                wind_direction_col="winddirection_deg",
                air_density_col="rho_kgm-3",
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
        turb_gross = self._turb_lt_gross
        mod_results = self._model_results

        # Create a data frame to store final results
        self._summary_results = pd.DataFrame(index=self._reanal, columns=self._turbs)

        daily_reanalysis = self.daily_reanalysis
        turb_gross = pd.DataFrame(index=daily_reanalysis.index)

        # Loop through the turbines and apply the GAM to the reanalysis data
        for t in self._turbs:  # Loop through turbines
            turb_gross.loc[:, t] = mod_results[t](
                daily_reanalysis["windspeed"],
                daily_reanalysis["wind_direction"],
                daily_reanalysis["density"],
            )

        turb_gross[turb_gross < 0] = 0

        # Calculate monthly sums of energy from long-term estimate
        turb_mo = turb_gross.resample("MS").sum()

        # Get average sum by calendar month
        turb_mo_avg = turb_mo.groupby(turb_mo.index.month).mean()

        # Store sum of turbine gross energy
        self._plant_gross[i] = turb_mo_avg.sum(axis=1).sum(axis=0)
        self._turb_lt_gross = turb_gross

    def plot_filtered_power_curves(self, save_folder, output_to_terminal=False):
        """
        Plot the raw and flagged power curve data and save to file.

        Args:
            save_folder('obj':'str'): The pathname to where figure files should be saved
            output_to_terminal('obj':'boolean'): Indicate whether or not to output figures to terminal

        Returns:
            (None)
        """
        import matplotlib.pyplot as plt

        dic = self._scada_dict

        # Loop through turbines
        for t in self._turbs:
            filt_df = dic[t].loc[dic[t]["flag_final"]]  # Filter only for valid data

            plt.figure(figsize=(6, 5))
            plt.scatter(dic[t].wmet_wdspd_avg, dic[t].wtur_W_avg, s=1, label="Raw")  # Plot all data
            plt.scatter(
                filt_df["wmet_wdspd_avg"], filt_df["wtur_W_avg"], s=1, label="Flagged"
            )  # Plot flagged data
            plt.xlim(0, 30)
            plt.xlabel("Wind speed (m/s)")
            plt.ylabel("Power (W)")
            plt.title("Filtered power curve for Turbine %s" % t)
            plt.legend(loc="lower right")
            plt.savefig(
                "%s/filtered_power_curve_%s.png"
                % (
                    save_folder,
                    t,
                ),
                dpi=200,
            )  # Save file

            # Output figure to terminal if desired
            if output_to_terminal:
                plt.show()

            plt.close()

    def plot_daily_fitting_result(self, save_folder, output_to_terminal=False):
        """
        Plot the raw and flagged power curve data and save to file.

        Args:
            save_folder('obj':'str'): The pathname to where figure files should be saved
            output_to_terminal('obj':'boolean'): Indicate whether or not to output figures to terminal

        Returns:
            (None)
        """
        import matplotlib.pyplot as plt

        mod_input = self.turbine_model_dict

        # Loop through turbines
        for t in self._turbs:
            df = mod_input[(t)]
            daily_reanalysis = self.daily_reanalysis
            ws_daily = daily_reanalysis["windspeed_ms"]

            df_imputed = df.loc[df["energy_corrected"] != df["energy_imputed"]]

            plt.figure(figsize=(6, 5))
            plt.plot(ws_daily, self._turb_lt_gross[t], "r.", alpha=0.1, label="Modeled")
            plt.plot(df["windspeed_ms"], df["energy_imputed"], ".", label="Input")
            plt.plot(df_imputed["windspeed_ms"], df_imputed["energy_imputed"], ".", label="Imputed")
            plt.xlabel("Wind speed (m/s)")
            plt.ylabel("Daily Energy (kWh)")
            plt.title("Daily SCADA Energy Fitting, Turbine %s" % t)
            plt.legend(loc="lower right")
            plt.savefig(
                "%s/daily_power_curve_%s.png"
                % (
                    save_folder,
                    t,
                ),
                dpi=200,
            )  # Save file

            # Output figure to terminal if desired
            if output_to_terminal:
                plt.show()

            plt.close()
