"""
This module provides basic methods for unit conversion and calculation of basic wind plant variables
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openoa.utils._converters import series_method


@series_method(data_cols=["power_col"])
def convert_power_to_energy(
    power_col: str | pd.Series, sample_rate_min="10min", data: pd.DataFrame = None
) -> pd.Series:
    """
    Compute energy [kWh] from power [kw] and return the data column

    Args:
        power_col(:obj:`str` | :obj:`pandas.Series`): The power data, in kW, or the name of the column
            in :py:attr:`data`.
        sample_rate_min(:obj:`float`): Sampling rate as a pandas offset alias, in minutes, to use
            for conversion. Defaults to "10min.
        data(:obj:`pandas.DataFrame`): The pandas DataFrame containing the col :py:attr:`power_col`.

    Returns:
        :obj:`pandas.Series`: Energy in kWh that matches the length of the input data frame :py:attr:'df'

    """
    # Get the number of minutes in the sample_rate_min
    _dt_range = pd.date_range(start="09/30/2022", periods=2, freq=sample_rate_min)
    _diff = _dt_range[1] - _dt_range[0]
    hours = _diff.days * 24 + _diff.seconds / 60 / 60

    # Convert the power, in kW, to energy, in kWh
    return power_col * hours


@series_method(data_cols=["net_energy", "availability", "curtailment"])
def compute_gross_energy(
    net_energy: str | pd.Series,
    availability: str | pd.Series,
    curtailment: str | pd.Series,
    availability_type: str = "frac",
    curtailment_type: str = "frac",
    data: str | pd.DataFrame = None,
):
    """
    Computes gross energy for a wind plant or turbine by adding reported :py:attr:`availability` and
    :py:attr:`curtailment` losses to reported net energy.

    Args:
        net_energy(:obj:`str` | `pandas.Series`): A pandas Series, the name of the columnn in
            :py:attr:`data` corresponding to the reported net energy for wind plant or turbine.
        availability(:obj:`str` | `pandas.Series`): A pandas Series, the name of the columnn in
            :py:attr:`data` corresponding to the reported availability losses for wind plant or turbine
        curtailment(:obj:`str` | `pandas.Series`): A pandas Series, the name of the columnn in
            :py:attr:`data` corresponding to the reported curtailment losses for wind plant or turbine
        availability_type(:obj:`str`): Either one of "frac" or "energy" corresponding to if the data
            provided in :py:attr:`availability` is in the range of [0, 1], or representing the energy
            lost.
        curtailment_type(:obj:`str`): Either one of "frac" or "energy" corresponding to if the data
            provided in :py:attr:`curtailment` is in the range of [0, 1], or representing the energy
            lost.
        data(:obj:`pd.DataFrame`, optional): The pandas DataFrame containing the net_energy,
            availability, and curtailment columns.

    Returns:
        gross(:obj:`pandas.Series`): Calculated gross energy for wind plant or turbine
    """
    if np.any(availability < 0) | np.any(curtailment < 0):
        raise ValueError(
            "Cannot have negative availability or curtailment input values. Check your data"
        )

    if (availability_type == "frac") & (curtailment_type == "frac"):
        gross = net_energy / (1 - availability - curtailment)
    elif (availability_type == "frac") & (curtailment_type == "energy"):
        gross = net_energy / (1 - availability) + curtailment
    elif (availability_type == "energy") & (curtailment_type == "frac"):
        gross = net_energy / (1 - curtailment) + availability
    elif (availability_type == "energy") & (curtailment_type == "energy"):
        gross = net_energy + curtailment + availability

    if np.any(gross < net_energy):
        raise ValueError("Gross energy cannot be less than net energy. Check your input values")

    return gross


@series_method(data_cols=["variable"])
def convert_feet_to_meter(variable: str | pd.Series, data: pd.DataFrame = None):
    """
    Compute variable in [meter] from [feet] and return the data column

    Args:
        variable(:obj:`str` | `pandas.Series`): A pandas Series, the name of the columnn in
            :py:attr:`data` corresponding to the data needing to be converted to meters.
        data(:obj:`pandas.DataFrame`): The pandas DataFrame containing the column :py:attr:`variable`.
        variable(:obj:`string`): variable in feet

    Returns:
        :obj:`pandas.Series`: :py:attr:`variable` in meters
    """
    return variable * 0.3048
