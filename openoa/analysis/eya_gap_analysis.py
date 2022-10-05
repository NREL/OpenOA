# This class defines key analytical routines for performing a 'gap-analysis'
# on EYA-estimated annual energy production (AEP) and that from operational data.
# Categories considered are availability, electrical losses, and long-term
# gross energy. The main output is a 'waterfall' plot linking the EYA-
# estimated and operational-estimated AEP values.

import attrs
import numpy as np
import pandas as pd
from attrs import field, define

from openoa import logging, logged_method_call
from openoa.plant import PlantData, FromDictMixin
from openoa.utils.pandas_plotting import set_styling


logger = logging.getLogger(__name__)
set_styling()


def validate_range_0_1(instance, attribute: attrs.Attribute, value: float):
    """Validates that the provided value is in the range of [0, 1)."""
    if not 0.0 <= value < 1.0:
        raise ValueError(f"The input to '{attribute.name}' must be in the range (0, 1).")


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
    availability_losses: float = field(converter=float, validator=validate_range_0_1)
    electrical_losses: float = field(converter=float, validator=validate_range_0_1)
    turbine_losses: float = field(converter=float, validator=validate_range_0_1)
    blade_degradation_losses: float = field(converter=float, validator=validate_range_0_1)
    wake_losses: float = field(converter=float, validator=validate_range_0_1)

    @availability_losses.validator
    @electrical_losses.validator
    @turbine_losses.validator
    @blade_degradation_losses.validator
    @wake_losses.validator
    def validate_0_1(self, attribute: attrs.Attribute, value: float) -> None:
        """Validates that the provided value is in the range of [0, 1)."""
        if not 0.0 <= value < 1.0:
            raise ValueError(f"The input to '{attribute.name}' must be in the range (0, 1).")


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
    availability_losses: float = field(converter=float, validator=validate_range_0_1)
    electrical_losses: float = field(converter=float, validator=validate_range_0_1)
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
    A serial (Pandas-driven) implementation of performing a gap analysis between the estimated
    annual energy production (AEP) from an energy yield estimate (EYA) and the actual AEP as
    measured from an operational assessment (OA)

    The gap analysis is based on comparing the following three key metrics

        1. Availability loss
        2. Electrical loss
        3. Sum of turbine ideal energy

    Here turbine ideal energy is defined as the energy produced during 'normal' or 'ideal' turbine operation,
    i.e., no downtime or considerable underperformance events. This value encompasses several different aspects
    of an EYA (wind resource estimate, wake losses,turbine performance, and blade degradation) and in most cases
    should have the largest impact in a gap analysis relative to the first two metrics.

    This gap analysis method is fairly straighforward. Relevant EYA and OA metrics are passed in when defining
    the class, differences in EYA estimates and OA results are calculated, and then a 'waterfall' plot is created
    showing the differences between the EYA and OA-estimated AEP values and how they are linked from differences in
    the three key metrics.

    Waterfall plot code was taken and modified from the following post: https://pbpython.com/waterfall-chart.html

    """

    eya_estimates: EYAEstimate = field(converter=EYAEstimate.from_dict)
    oa_results: OAResults = field(converter=OAResults.from_dict)
    plant: PlantData = field(
        default=None, validator=attrs.validators.instance_of((PlantData, type(None)))
    )

    # Internally produced attributes
    data: list = field(factory=list)
    compiled_data: list = field(factory=list)
    plot_index: list = field(
        default=["eya_aep", "ideal_energy", "avail_loss", "elec_loss", "unexplained/uncertain"]
    )

    @logged_method_call
    def __attrs_post_init__(self):
        """
        Initialize EYA gap analysis class with data and parameters.

        Args:
         plant(:obj:`PlantData object`): PlantData object from which EYAGapAnalysis should draw data.
         eya_estimates(:obj:`numpy array`): Numpy array with EYA estimates listed in required order
         oa_results(:obj:`numpy array`): Numpy array with OA results listed in required order.
         make_fig(:obj:`boolean`): Indicate whether to produce the waterfall plot
         save_fig_path(:obj:`boolean` or `string'): Provide path to save waterfall plot, or set to
                                                    False to not save plot

        """
        logger.info("Initializing EYA Gap Analysis Object")

        # # Store EYA inputs into dictionary
        # self._eya_estimates = {
        #     "aep": eya_estimates[0],  # GWh/yr
        #     "gross_energy": eya_estimates[1],  # GWh/yr
        #     "availability_losses": eya_estimates[2],  # Fraction
        #     "electrical_losses": eya_estimates[3],  # Fraction
        #     "turbine_losses": eya_estimates[4],  # Fraction
        #     "blade_degradation_losses": eya_estimates[5],  # Fraction
        #     "wake_losses": eya_estimates[6],
        # }  # Fraction

        # # Store OA results into dictionary
        # self._oa_results = {
        #     "aep": oa_results[0],  # GWh/yr
        #     "availability_losses": oa_results[1],  # Fraction
        #     "electrical_losses": oa_results[2],  # Fraction
        #     "turbine_ideal_energy": oa_results[3],
        # }  # Fraction

        # # Axis labels for waterfall plot
        # self._plot_index = [
        #     "eya_aep",
        #     "ideal_energy",
        #     "avail_loss",
        #     "elec_loss",
        #     "unexplained/uncertain",
        # ]
        # self._makefig = make_fig
        # self._savefigpath = save_fig_path

    @logged_method_call
    def run(self):
        """
        Run the EYA Gap analysis functions in order by calling this function.

        Args:
            (None)

        Returns:
            (None)
        """

        self._ompiled_data = self.compile_data()  # Compile EYA and OA data

        # if self._makefig:
        #     self.waterfall_plot(
        #         self._compiled_data, self._plot_index, self._savefigpath
        #     )  # Produce waterfall plot

        logger.info("Gap analysis complete")

    def compile_data(self):
        """
        Compile EYA and OA metrics, compute differences, and return data needed for waterfall plot.

        Args:
            (None)

        Returns:
            (None)
        """

        # Calculate EYA ideal turbine energy
        eya_turbine_ideal_energy = (
            self.eya_estimates.gross_energy
            * (1 - self.eya_estimates.turbine_losses)
            * (1 - self.eya_estimates.wake_losses)
            * (1 - self.eya_estimates.blade_degradation_losses)
        )

        # Get required gap analysis values from EYA
        eya_aep = self.eya_estimates.aep
        eya_avail = self.eya_estimates.availability_losses
        eya_elec = self.eya_estimates.electrical_losses

        # Get required gap analysis values from OA
        oa_turb_ideal = self.oa_results.turbine_ideal_energy
        oa_aep = self.oa_results.aep
        oa_avail = self.oa_results.availability_losses
        oa_elec = self.oa_results.electrical_losses

        # Calculate EYA-OA differences, determine the residual or unaccounted value
        turb_gross_diff = oa_turb_ideal - eya_turbine_ideal_energy
        avail_diff = (eya_avail - oa_avail) * eya_turbine_ideal_energy
        elec_diff = (eya_elec - oa_elec) * eya_turbine_ideal_energy
        unaccounted = -(eya_aep + turb_gross_diff + avail_diff + elec_diff) + oa_aep

        # Combine calculations into array and return
        data = [eya_aep, turb_gross_diff, avail_diff, elec_diff, unaccounted]
        self._data = data
        return data

    def waterfall_plot(self, data, index, save_fig_path):
        """
        Produce a waterfall plot showing the progression from the EYA to OA estimates of AEP.

        Args:
            data(:obj:`numpy array`): data to be used to create waterfall plot
            index(:obj:`list`): List of string values to be used for x-axis labels
            path(:obj:`string`): Location to save waterfall plot

        Returns:
            (None)
        """

        # Store data and create a blank series to use for the waterfall
        trans = pd.DataFrame(data={"amount": data}, index=index)  # Assign gaps to data frame
        blank = trans.amount.cumsum().shift(1).fillna(0)  # Perform cumulative sum on gap values

        # Get the net total number for the final element in the waterfall
        total = trans.sum().amount
        trans.loc["oa_aep"] = total  # Add new field to gaps data frame
        blank.loc["oa_aep"] = total  # Add new field to cumulative sum data frame

        # The steps graphically show the levels as well as used for label placement
        step = blank.reset_index(drop=True).repeat(3).shift(-1)
        step[1::3] = np.nan

        # When plotting the last element, we want to show the full bar,
        # Set the blank to 0
        blank.loc["oa_aep"] = 0

        # Plot and label
        my_plot = trans.plot(kind="bar", stacked=True, bottom=blank, legend=None, figsize=(12, 6))
        my_plot.plot(step.index, step.values, "k")
        my_plot.set_ylabel("Energy (GWh/yr)")
        my_plot.set_title(self.plant)

        # Get the y-axis position for the labels
        y_height = trans.amount.cumsum().shift(1).fillna(0)

        # Get an offset so labels don't sit right on top of the bar
        mx = trans.max()  # Max value in gap analysis values
        neg_offset = mx / 25
        pos_offset = mx / 50

        # Add labels to each bar
        loop = 0
        for index, row in trans.iterrows():
            # For the last item in the list, we don't want to double count
            if row["amount"] == total:
                y = y_height[loop]
            else:
                y = y_height[loop] + row["amount"]

            # Determine if we want a neg or pos offset
            if row["amount"] > 0:
                y += pos_offset
            else:
                y -= neg_offset
            my_plot.annotate("{:,.0f}".format(row["amount"]), (loop, y), ha="center")
            loop += 1

        # Adjust y-axis to focus on region of interest
        plt_min = blank[1:-1].min()  # Min value in cumulative sum values
        plt_max = blank[1:].max()  # Min value in cumulative sum values
        my_plot.set_ylim(0.9 * plt_min, 1.1 * plt_max)  # blank.max()+int(plot_offset))

        # Rotate the labels
        my_plot.set_xticklabels(trans.index, rotation=0)

        # Save figure
        if save_fig_path:
            my_plot.get_figure().savefig(
                save_fig_path + "/waterfall.png", dpi=200, bbox_inches="tight"
            )

        return my_plot
