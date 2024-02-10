# This class defines key analytical routines for calculating electrical losses for
# a wind plant using operational data. Electrical loss is calculated per month and on
# an average annual basis by comparing monthly energy production from the turbines
# and the revenue meter

from __future__ import annotations

import datetime
from copy import deepcopy

import attrs
import numpy as np
import pandas as pd
import numpy.typing as npt
import matplotlib.pyplot as plt
from tqdm import tqdm
from attrs import field, define

import openoa.utils.timeseries as ts
from openoa.plant import PlantData
from openoa.schema import FromDictMixin, ResetValuesMixin
from openoa.logging import logging, logged_method_call
from openoa.utils.plot import set_styling
from openoa.analysis._analysis_validators import validate_UQ_input, validate_half_closed_0_1_right


logger = logging.getLogger(__name__)
set_styling()

NDArrayFloat = npt.NDArray[np.float64]

MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24


@define(auto_attribs=True)
class ElectricalLosses(FromDictMixin, ResetValuesMixin):
    """
    A serial implementation of calculating the average monthly and annual electrical losses at a
    wind power plant, and the associated uncertainty. Energy output from the turbine SCADA meter and
    the wind plant revenue meter are used to estimate electrical losses.

    First, the daily sums of turbine and revenue meter energy are calculated over the plant's period
    of record where all turbines and the revenue meter contan every considered timestep. Electrical
    losses are then calculated as the difference between the total turbine energy production and the
    meter production over those concurrent days.

    For uncertainty quantification, a Monte Carlo (MC) approach is used to sample the revenue meter
    data and SCADA data with a default 0.5% imposed uncertainty, alongside a sampled filtering
    parameter. The uncertainty in estimated electrical losses is quantified as the standard
    deviation of the distribution of losses obtained from the MC sampling.

    If the revenue meter data is not provided on a daily or sub-daily basis (e.g. monthly), the the
    sum of daily turbine energy is corrected for any missing reported energy data from the turbines
    based on the ratio of expected number of data points per day to the actual data points
    available. The daily corrected sum of turbine energy is then summed on a monthly basis.
    Electrical loss is then the difference between the total corrected turbine energy production and
    meter production over those concurrent months.

    Args:
        plant(:obj:`PlantData`): A :py:attr:`openoa.plant.PlantData` object that has been validated
            with at least `:py:attr:`openoa.plant.PlantData.analysis_type` = "ElectricalLosses".
        UQ(:obj:`bool`): Indicator to perform (True) or not (False) uncertainty quantification.
        num_sim(:obj:`int`): Number of Monte Carlo simulations to perform.
        uncertainty_meter(:obj:`float`): Uncertainty imposed on the revenue meter data (for
            :py:attr:`UQ` = True case).
        uncertainty_scada(:obj:`float`): Uncertainty imposed on the scada data (for :py:attr:`UQ` =
            True case).
        uncertainty_correction_threshold(:obj:`tuple` | `float`): Data availability thresholds, in
            the range of (0, 1), under which months should be eliminated. If :py:attr:`UQ` = True,
            then a 2-element tuple containing an upper and lower bound for a randomly selected value
            should be given, otherwise, a scalar value should be provided.
    """

    plant: PlantData = field(converter=deepcopy, validator=attrs.validators.instance_of(PlantData))
    UQ: bool = field(default=False, validator=attrs.validators.instance_of(bool))
    num_sim: int = field(default=20000, converter=int)
    uncertainty_meter: float = field(default=0.005, validator=validate_half_closed_0_1_right)
    uncertainty_scada: float = field(default=0.005, validator=validate_half_closed_0_1_right)
    uncertainty_correction_threshold: NDArrayFloat | tuple[float, float] | float = field(
        default=(0.9, 0.995), validator=(validate_UQ_input, validate_half_closed_0_1_right)
    )

    # Internally created attributes need to be given a type before usage
    monthly_meter: bool = field(default=False, init=False)
    inputs: pd.DataFrame = field(init=False)
    electrical_losses: NDArrayFloat = field(init=False)
    scada_sum: pd.DataFrame = field(init=False)
    scada_daily: pd.DataFrame = field(init=False)
    scada_full_count: pd.DataFrame = field(init=False)
    meter_daily: pd.DataFrame = field(init=False)
    combined_energy: pd.DataFrame = field(init=False)
    total_turbine_energy: pd.DataFrame = field(init=False)
    total_meter_energy: pd.DataFrame = field(init=False)
    run_parameters: list[str] = field(
        init=False,
        default=[
            "UQ",
            "num_sim",
            "uncertainty_meter",
            "uncertainty_scada",
            "uncertainty_correction_threshold",
        ],
    )

    @logged_method_call
    def __attrs_post_init__(self):
        """
        Initialize logging and post-initialization setup steps.
        """
        if {"ElectricalLosses", "all"}.intersection(self.plant.analysis_type) == set():
            self.plant.analysis_type.append("ElectricalLosses")

        # Ensure the data are up to spec before continuing with initialization
        self.plant.validate()

        logger.info("Initializing Electrical Losses Object")

        # Check that selected UQ is allowed and reset num_sim if no UQ
        if self.UQ:
            logger.info("Note: uncertainty quantification will be performed in the calculation")
        else:
            logger.info("Note: uncertainty quantification will NOT be performed in the calculation")
            self.num_sim = 1  # override in case of bad user input

        # Process the SCADA and meter data appropriately
        self.process_scada()
        if self.plant.metadata.meter.frequency not in ("MS", "ME", "1MS"):
            self.process_meter()
            self.monthly_meter = False

    @logged_method_call
    def run(
        self,
        num_sim: int | None = None,
        uncertainty_meter: NDArrayFloat | float = None,
        uncertainty_scada: NDArrayFloat | float = None,
        uncertainty_correction_threshold: NDArrayFloat | tuple[float, float] | float = None,
    ):
        """
        Run the electrical losses calculation.

        .. note:: If None is provided to any of the inputs, then the last used input value will be
            used for the analysis, and if no prior values were set, then this is the model's defaults.

        Args:
            num_sim(:obj:`int`): Number of Monte Carlo simulations to perform.
            uncertainty_meter(:obj:`float`): Uncertainty imposed on the revenue meter data (for
                :py:attr:`UQ` = True case).
            uncertainty_scada(:obj:`float`): Uncertainty imposed on the scada data (for :py:attr:`UQ` =
                True case).
            uncertainty_correction_threshold(:obj:`tuple` | `float`): Data availability thresholds, in
                the range of (0, 1], under which months should be eliminated. If :py:attr:`UQ` = True,
                then a 2-element tuple containing an upper and lower bound for a randomly selected value
                should be given, otherwise, a scalar value should be provided.
        """
        initial_parameters = {}
        if num_sim is not None:
            if self.UQ:
                initial_parameters["num_sim"] = self.num_sim
                self.num_sim = num_sim
            elif num_sim > 1:
                logger.info(
                    "`num_sim` can NOT be greater than 1 when `UQ=False`, value has not been set."
                )
        if uncertainty_meter is not None:
            initial_parameters["uncertainty_meter"] = self.uncertainty_meter
            self.uncertainty_meter = uncertainty_meter
        if uncertainty_scada is not None:
            initial_parameters["uncertainty_scada"] = self.uncertainty_scada
            self.uncertainty_scada = uncertainty_scada
        if uncertainty_correction_threshold is not None:
            initial_parameters[
                "uncertainty_correction_threshold"
            ] = self.uncertainty_correction_threshold
            self.uncertainty_correction_threshold = uncertainty_correction_threshold

        # Setup Monte Carlo approach, and calculate the electrical losses
        self.setup_inputs()
        self.calculate_electrical_losses()

        # Reset the class arguments back to the initialized values
        self.set_values(initial_parameters)

    @logged_method_call
    def setup_inputs(self):
        """
        Create and populate the data frame defining the simulation parameters.
        This data frame is stored as self.inputs.
        """
        if self.UQ:
            n_decimal = max(
                len(str(el).split(".")[1]) for el in self.uncertainty_correction_threshold
            )
            integer_multiplier = 10**n_decimal
            inputs = {
                "meter_data_fraction": np.random.normal(1, self.uncertainty_meter, self.num_sim),
                "scada_data_fraction": np.random.normal(1, self.uncertainty_scada, self.num_sim),
                "correction_threshold": np.random.randint(
                    self.uncertainty_correction_threshold[0] * integer_multiplier,
                    self.uncertainty_correction_threshold[1] * integer_multiplier,
                    self.num_sim,
                )
                / integer_multiplier,
            }
            self.inputs = pd.DataFrame(inputs)
        else:
            inputs = {
                "meter_data_fraction": 1,
                "scada_data_fraction": 1,
                "correction_threshold": self.uncertainty_correction_threshold,
            }
            self.inputs = pd.DataFrame(inputs, index=[0])

        self.electrical_losses = np.empty([self.num_sim, 1])

    @logged_method_call
    def process_scada(self):
        """
        Calculate daily sum of turbine energy only for days when all turbines are reporting
        at all time steps.
        """
        logger.info("Processing SCADA data")

        scada_df = self.plant.scada.copy()

        # Sum up SCADA data power and energy and count number of entries
        ix_time = self.plant.scada.index.get_level_values("time")
        self.scada_sum = scada_df.groupby(ix_time)[["WTUR_SupWh"]].sum()
        self.scada_sum["count"] = scada_df.groupby(ix_time)[["WTUR_SupWh"]].count()

        # Calculate daily sum of all turbine energy production and count number of entries
        self.scada_daily = self.scada_sum.resample("D")["WTUR_SupWh"].sum().to_frame()
        self.scada_daily["count"] = self.scada_sum.resample("D")["count"].sum()

        # Specify expected count provided all turbines reporting
        expected_count = (
            HOURS_PER_DAY
            * MINUTES_PER_HOUR
            / (ts.offset_to_seconds(self.plant.metadata.scada.frequency) / 60)
            * self.plant.n_turbines
        )

        # Correct sum of turbine energy for cases with missing reported data
        self.scada_daily["corrected_energy"] = (
            self.scada_daily["WTUR_SupWh"] * expected_count / self.scada_daily["count"]
        )
        self.scada_daily["percent"] = self.scada_daily["count"] / expected_count

        # Store daily SCADA data where all turbines reporting for every time step during the day
        self.scada_full_count = self.scada_daily.loc[self.scada_daily["count"] == expected_count]

    @logged_method_call
    def process_meter(self):
        """
        Calculate daily sum of meter energy only for days when meter data is reporting at all time steps.
        """
        logger.info("Processing meter data")

        meter_df = self.plant.meter.copy()

        # Sum up meter data to daily
        self.meter_daily = meter_df.resample("D").sum()
        self.meter_daily["count"] = meter_df.resample("D")["MMTR_SupWh"].count()

        # Specify expected count provided all timestamps reporting
        expected_count = (
            HOURS_PER_DAY
            * MINUTES_PER_HOUR
            / (ts.offset_to_seconds(self.plant.metadata.scada.frequency) / 60)
        )

        # Keep only data with all turbines reporting for every time step during the day
        self.meter_daily = self.meter_daily[self.meter_daily["count"] == expected_count]

    @logged_method_call
    def calculate_electrical_losses(self):
        """
        Apply Monte Carlo approach to calculate electrical losses and their uncertainty based on the
        difference in the sum of turbine and metered energy over the compiled days.
        """
        logger.info("Calculating electrical losses")

        # Loop through number of simulations, calculate losses each time, store results
        for n in tqdm(np.arange(self.num_sim)):
            _run = self.inputs.loc[n]
            meter_df = self.plant.meter.copy()

            # If monthly meter data, sum the corrected daily turbine energy to monthly and merge
            if self.monthly_meter:
                scada_monthly = self.scada_daily.resample("MS")["corrected_energy"].sum().to_frame()
                scada_monthly.columns = ["WTUR_SupWh"]

                # Determine availability for each month represented
                scada_monthly["count"] = self.scada_sum.resample("MS")["count"].sum()
                scada_monthly["expected_count_monthly"] = (
                    scada_monthly.index.daysinmonth
                    * HOURS_PER_DAY
                    * MINUTES_PER_HOUR
                    / (pd.to_timedelta(self.plant.scada.frequency).total_seconds() / 60)
                    * self.plant.n_turbines
                )
                scada_monthly["percent"] = (
                    scada_monthly["count"] / scada_monthly["expected_count_monthly"]
                )

                # Filter out months in which there was less than x% of total running (all turbines at all timesteps)
                scada_monthly = scada_monthly.loc[
                    scada_monthly["percent"] >= _run.correction_threshold, :
                ]
                self.combined_energy = meter_df.join(
                    scada_monthly, lsuffix="_meter", rsuffix="_scada"
                )

            # If sub-monthly meter data, merge the daily data for which all turbines are reporting at all timestamps
            else:
                # Note 'self.scada_full_count' only contains full reported data
                self.combined_energy = self.meter_daily.join(
                    self.scada_full_count, lsuffix="_meter", rsuffix="_scada"
                )

            # Drop non-concurrent timestamps and get total sums over concurrent period of record
            self.combined_energy.dropna(inplace=True)
            merge_sum = self.combined_energy.sum(axis=0)

            # Calculate electrical loss from difference of sum of turbine and meter energy
            self.total_turbine_energy = merge_sum["WTUR_SupWh"] * _run.scada_data_fraction
            self.total_meter_energy = merge_sum["MMTR_SupWh"] * _run.meter_data_fraction

            self.electrical_losses[n] = 1 - self.total_meter_energy / self.total_turbine_energy

    def plot_monthly_losses(
        self,
        xlim: tuple[datetime.datetime | None, datetime.datetime | None] = (None, None),
        ylim: tuple[float | None, float | None] = (None, None),
        return_fig: bool = False,
        figure_kwargs: dict = {},
        legend_kwargs: dict = {},
        plot_kwargs: dict = {},
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots the monthly timeseries of electrical losses as a percent.

        Args:
            xlim(:obj: `tuple[float, float]`, optional): A tuple of the x-axis (min, max) values.
                Defaults to (None, None).
            ylim(:obj: `tuple[float, float]`, optional): A tuple of the y-axis (min, max) values.
                Defaults to (None, None).
            return_fig(:obj:`bool`, optional): Set to True to return the figure and axes objects,
                otherwise set to False. Defaults to False.
            figure_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be
                passed to ``plt.figure()``. Defaults to {}.
            scatter_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be
                passed to ``ax.plot()``. Defaults to {}.
            legend_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be
                passed to ``ax.legend()``. Defaults to {}.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: If :py:attr:`return_fig`, then return the figure
                and axes objects in addition to showing the plot.
        """
        figure_kwargs.setdefault("dpi", 200)
        fig = plt.figure(**figure_kwargs)
        ax = fig.add_subplot(111)

        monthly_energy = self.combined_energy.resample("MS").sum()
        losses = (
            monthly_energy["corrected_energy"] - monthly_energy["MMTR_SupWh"]
        ) / monthly_energy["corrected_energy"]

        mean = losses.mean()
        std = losses.std()
        ax.plot(
            losses * 100,
            label=f"Electrical Losses\n$\\mu$={mean:.2%}, $\\sigma$={std:.2%}",  # noqa: W605
            **plot_kwargs,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.legend(**legend_kwargs)
        ax.set_xlabel("Period of Record")
        ax.set_ylabel("Electrical Losses (%)")

        fig.tight_layout()
        plt.show()
        if return_fig:
            return fig, ax


__defaults_UQ = ElectricalLosses.__attrs_attrs__.UQ.default
__defaults_num_sim = ElectricalLosses.__attrs_attrs__.num_sim.default
__defaults_uncertainty_correction_threshold = (
    ElectricalLosses.__attrs_attrs__.uncertainty_correction_threshold.default
)
__defaults_uncertainty_meter = ElectricalLosses.__attrs_attrs__.uncertainty_meter.default
__defaults_uncertainty_scada = ElectricalLosses.__attrs_attrs__.uncertainty_scada.default


def create_ElectricalLosses(
    project: PlantData,
    UQ: bool = __defaults_UQ,
    num_sim: int = __defaults_num_sim,
    uncertainty_correction_threshold: NDArrayFloat
    | tuple[float, float]
    | float = __defaults_uncertainty_correction_threshold,
    uncertainty_meter: NDArrayFloat | tuple[float, float] | float = __defaults_uncertainty_meter,
    uncertainty_scada: NDArrayFloat | tuple[float, float] | float = __defaults_uncertainty_scada,
) -> ElectricalLosses:
    return ElectricalLosses(
        plant=project,
        UQ=UQ,
        num_sim=num_sim,
        uncertainty_meter=uncertainty_meter,
        uncertainty_scada=uncertainty_scada,
        uncertainty_correction_threshold=uncertainty_correction_threshold,
    )


create_ElectricalLosses.__doc__ = ElectricalLosses.__doc__
