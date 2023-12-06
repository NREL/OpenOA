"""
This class defines key analytical routines for performing a 'gap-analysis' on EYA-estimated annual
energy production (AEP) and that from operational data. Categories considered are availability,
electrical losses, and long-term gross energy. The main output is a 'waterfall' plot linking the EYA-
estimated and operational-estimated AEP values.
"""

from __future__ import annotations

import attrs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from attrs import field, define

from openoa.plant import PlantData
from openoa.utils import plot
from openoa.schema import FromDictMixin
from openoa.logging import logging, logged_method_call
from openoa.analysis._analysis_validators import validate_half_closed_0_1_left


logger = logging.getLogger(__name__)
plot.set_styling()


@define(auto_attribs=True)
class EYAEstimate(FromDictMixin):
    """Dataclass for catalogging and validating the consultant-produced Energy Yield Assessment
    (EYA) data.

    Args:
        aep(:obj:`float`): The EYA predicted Annual Energy Production (AEP), in GWh/yr.
        gross_energy(:obj:`float`): The EYA predicted gross energy, in GWh/yr.
        availability_losses(:obj:`float`): The EYA predicted availability losses, in the range of
            [0, 1).
        electrical_losses(:obj:`float`): The EYA predicted electrical losses, in the range of [0, 1).
        turbine_losses(:obj:`float`): The EYA predicted turbine losses, in the range of [0, 1).
        blade_degradation_losses(:obj:`float`): The EYA predicted blade degradation losses, in the
            range of [0, 1).
        wake_losses(:obj:`float`): The EYA predicted wake losses, in the range of [0, 1).

    """

    aep: float = field(converter=float)
    gross_energy: float = field(converter=float)
    availability_losses: float = field(converter=float, validator=validate_half_closed_0_1_left)
    electrical_losses: float = field(converter=float, validator=validate_half_closed_0_1_left)
    turbine_losses: float = field(converter=float, validator=validate_half_closed_0_1_left)
    blade_degradation_losses: float = field(
        converter=float, validator=validate_half_closed_0_1_left
    )
    wake_losses: float = field(converter=float, validator=validate_half_closed_0_1_left)


@define(auto_attribs=True)
class OAResults(FromDictMixin):
    """Dataclass for catalogging and validating the analysis-produced operation analysis (OA) data.

    Args:
        aep(:obj:`float`): The OA results for Annual Energy Production (AEP), in GWh/yr.
        availability_losses(:obj:`float`): The OA results for availability losses, in the range of
            [0, 1).
        electrical_losses(:obj:`float`): The OA results for electrical losses, in the range of
            [0, 1).
        turbine_ideal_energy(:obj:`float`): The OA results for turbine ideal energy, in GWh/yr.

    """

    aep: float = field(converter=float)
    availability_losses: float = field(converter=float, validator=validate_half_closed_0_1_left)
    electrical_losses: float = field(converter=float, validator=validate_half_closed_0_1_left)
    turbine_ideal_energy: float = field(converter=float)

    @availability_losses.validator
    @electrical_losses.validator
    def validate_0_1(self, attribute: attrs.Attribute, value: float) -> None:
        """Validates that the provided value is in the range of [0, 1)."""
        if not 0.0 <= value < 1.0:
            raise ValueError(f"The input to '{attribute.name}' must be in the range (0, 1).")


@define(auto_attribs=True)
class EYAGapAnalysis(FromDictMixin):
    """
    Performs a gap analysis between the estimated annual energy production (AEP) from an energy
    yield estimate (EYA) and the actual AEP as measured from an operational assessment (OA).

    The gap analysis is based on comparing the following three key metrics:

        1. Availability loss
        2. Electrical loss
        3. Sum of turbine ideal energy

    Here turbine ideal energy is defined as the energy produced during 'normal' or 'ideal' turbine
    operation, i.e., no downtime or considerable underperformance events. This value encompasses
    several different aspects of an EYA (wind resource estimate, wake losses,turbine performance,
    and blade degradation) and in most cases should have the largest impact in a gap analysis
    relative to the first two metrics.

    This gap analysis method is fairly straighforward. Relevant EYA and OA metrics are passed in
    when defining the class, differences in EYA estimates and OA results are calculated, and then a
    'waterfall' plot is created showing the differences between the EYA and OA-estimated AEP values
    and how they are linked from differences in the three key metrics.

    Args:
        plant(:obj:`PlantData object`): PlantData object from which EYAGapAnalysis should draw data.
        eya_estimates(:obj:`EYAEstimate`): Numpy array with EYA estimates listed in required order
        oa_results(:obj:`OAResults`): Numpy array with OA results listed in required order.
    """

    plant: PlantData = field(validator=attrs.validators.instance_of((PlantData, type(None))))
    eya_estimates: EYAEstimate = field(converter=EYAEstimate.from_dict)
    oa_results: OAResults = field(converter=OAResults.from_dict)

    # Internally produced attributes
    data: list = field(factory=list)
    compiled_data: list = field(factory=list)

    @logged_method_call
    def __attrs_post_init__(self):
        """Initialize EYA gap analysis class with data and parameters."""
        if not (isinstance(self.plant, PlantData) or self.plant is None):
            raise TypeError(
                f"The passed `plant` object must be of type `PlantData` or `None`, not: {type(self.plant)}"
            )

        logger.info("Initialized EYA Gap Analysis Object")

    @logged_method_call
    def run(self):
        """
        Run the EYA Gap analysis functions in order by calling this function.

        Args:
            (None)

        Returns:
            (None)
        """

        self.compiled_data = self.compile_data()  # Compile EYA and OA data
        logger.info("Gap analysis complete")

    @logged_method_call
    def compile_data(self):
        """
        Compiles the EYA and OA metrics, and computes the differences.

        Returns:
            :obj:`list[float]`: The list of EYA AEP, and differences in turbine gross energy,
                availability losses, electrical losses, and unaccounted losses.
        """
        logger.info("Compiling EYA and OA data")

        # Calculate EYA ideal turbine energy
        eya_turbine_ideal_energy = (
            self.eya_estimates.gross_energy
            * (1 - self.eya_estimates.turbine_losses)
            * (1 - self.eya_estimates.wake_losses)
            * (1 - self.eya_estimates.blade_degradation_losses)
        )

        # Calculate EYA-OA differences, determine the residual or unaccounted value
        turb_gross_diff = self.oa_results.turbine_ideal_energy - eya_turbine_ideal_energy
        avail_diff = (
            self.eya_estimates.availability_losses - self.oa_results.availability_losses
        ) * eya_turbine_ideal_energy
        elec_diff = (
            self.eya_estimates.electrical_losses - self.oa_results.electrical_losses
        ) * eya_turbine_ideal_energy
        unaccounted = (
            -(self.eya_estimates.aep + turb_gross_diff + avail_diff + elec_diff)
            + self.oa_results.aep
        )

        # Combine calculations into array and return
        return [self.eya_estimates.aep, turb_gross_diff, avail_diff, elec_diff, unaccounted]

    def plot_waterfall(
        self,
        index: list[str] = [
            "EYA AEP",
            "TIE",
            "Availability\nLosses",
            "Electrical\nLosses",
            "Unexplained",
            "OA AEP",
        ],
        ylabel: str = "Energy (GWh/yr)",
        ylim: tuple[float, float] = (None, None),
        return_fig: bool = False,
        plot_kwargs: dict = {},
        figure_kwargs: dict = {},
    ) -> None | tuple:
        """
        Produce a waterfall plot showing the progression from the EYA estimates to the calculated OA
        estimates of AEP.

        Args:
            index(:obj:`list`): List of string values to be used for x-axis labels, which should
                have one more value than the number of points in :py:attr:`data` to account for
                the resulting OA total. Defaults to ["EYA AEP", "TIE",  "Availability Losses",
                "Electrical Losses", "Unexplained", "OA AEP"].
            ylabel(:obj:`str`): The y-axis label. Defaults to "Energy (GWh/yr)".
            ylim(:obj:`tuple[float | None, float | None]`): The y-axis minimum and maximum display
                range. Defaults to (None, None).
            return_fig(:obj:`bool`, optional): Set to True to return the figure and axes objects,
                otherwise set to False. Defaults to False.
            figure_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be
                passed to ``plt.figure()``. Defaults to {}.
            plot_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be
                passed to ``ax.plot()``. Defaults to {}.
            legend_kwargs(:obj:`dict`, optional): Additional keyword arguments that should be
                passed to ``ax.legend()``. Defaults to {}.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: If :py:attr:`return_fig`, then return the figure
                and axes objects in addition to showing the plot.
        """
        return plot.plot_waterfall(
            self.compiled_data,
            index=index,
            ylim=ylim,
            ylabel=ylabel,
            return_fig=return_fig,
            plot_kwargs=plot_kwargs,
            figure_kwargs=figure_kwargs,
        )


def create_EYAGapAnalysis(
    project: PlantData, eya_estimates: dict | EYAEstimate, oa_results: dict | OAResults
) -> EYAGapAnalysis:
    return EYAGapAnalysis(project, eya_estimates, oa_results)


create_EYAGapAnalysis.__doc__ = EYAGapAnalysis.__doc__
