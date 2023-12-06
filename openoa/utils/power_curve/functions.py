"""
This module holds ready-to-use power curve functions. They take windspeed and power columns as arguments and return a
python function which can be used to evaluate the power curve at arbitrary locations.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from pygam import LinearGAM
from scipy.optimize import differential_evolution

from openoa.utils._converters import series_method, dataframe_method
from openoa.utils.power_curve.parametric_forms import logistic5param
from openoa.utils.power_curve.parametric_optimize import least_squares, fit_parametric_power_curve


@series_method(data_cols=["windspeed_col", "power_col"])
def IEC(
    windspeed_col: str | pd.Series,
    power_col: str | pd.Series,
    bin_width: float = 0.5,
    windspeed_start: float = 0,
    windspeed_end: float = 30.0,
    data: pd.DataFrame = None,
) -> Callable:
    """
    Use IEC 61400-12-1-2 method for creating a binned wind-speed power curve. Power is set to zero
    for values outside the cutoff range: [:py:attr:`windspeed_start`, :py:attr:`windspeed_end`].

    Args:
        windspeed_col(:obj:`str` | `pandas.Series`): Windspeed data, or the name of the column in
            :py:attr:`data`.
        power_col(:obj:`str` | `pandas.Series`): Power data, or the name of the column in
            :py:attr:`data`.
        bin_width(:obj:`float`): Width of windspeed bin. Defaults to 0.5 m/s, per the standard.
        windspeed_start(:obj:`float`): Left edge of first windspeed bin. Defaults to 0.0.
        windspeed_end(:obj:`float`): Right edge of last windspeed bin. Defaults to 30.0
        data(:obj:`pandas.DataFrame`, optional): a pandas DataFrame containing
            :py:attr:`windspeed_col` and :py:attr:`power_col`. Defaults to None.

    Returns:
        :obj:`Callable`: Python function of type (Array[float] -> Array[float]) implementing the power curve.

    """

    # Set up evenly spaced bins of fixed width, with any value over the maximum getting np.inf
    n_bins = int(np.ceil((windspeed_end - windspeed_start) / bin_width)) + 1
    bins = np.append(np.linspace(windspeed_start, windspeed_end, n_bins), [np.inf])

    # Initialize an array which will hold the mean values of each bin
    P_bin = np.ones(len(bins) - 1) * np.nan

    # Compute the mean of each bin and set corresponding P_bin
    for ibin in range(0, len(bins) - 1):
        indices = (windspeed_col >= bins[ibin]) & (windspeed_col < bins[ibin + 1])
        P_bin[ibin] = power_col.loc[indices].mean()

    # Linearly interpolate any missing bins
    P_bin = pd.Series(data=P_bin).interpolate(method="linear").bfill().values

    # Create a closure over the computed bins which computes the power curve value for arbitrary array-like input
    def pc_iec(x):
        P = np.zeros(np.shape(x))
        for i in range(0, len(bins) - 1):
            idx = np.where((x >= bins[i]) & (x < bins[i + 1]))
            P[idx] = P_bin[i]
        cutoff_idx = (x < windspeed_start) | (x > windspeed_end)
        P[cutoff_idx] = 0.0
        return P

    return pc_iec


@series_method(data_cols=["windspeed_col", "power_col"])
def logistic_5_parametric(
    windspeed_col: str | pd.Series, power_col: str | pd.Series, data: pd.DataFrame = None
) -> Callable:
    """In this case, the function fits the 5 parameter logistics function to observed data via a
    least-squares optimization (i.e. minimizing the sum of the squares of the residual between the
    points as evaluated by the parameterized function and the points of observed data).

    Extra:
    The present implementation follows the filtering method reported in:

        M. Yesilbudaku Partitional clustering-based outlier detection
        for power curve optimization of wind turbines 2016 IEEE International
        Conference on Renewable Energy Research and
        Applications (ICRERA), Birmingham, 2016, pp. 1080-1084.

    and the power curve method developed and reviewed in:

        M Lydia, AI Selvakumar, SS Kumar, GEP. Kumar
        Advanced algorithms for wind turbine power curve modeling
        IEEE Trans Sustainable Energy, 4 (2013), pp. 827-835

        M. Lydia, S.S. Kumar, I. Selvakumar, G.E. Prem Kumar
        A comprehensive review on wind turbine power curve modeling techniques
        Renew. Sust. Energy Rev., 30 (2014), pp. 452-460



    Args:
        windspeed_col(:obj:`str` | `pandas.Series`): Windspeed data, or the name of the column in
            :py:attr:`data`.
        power_col(:obj:`str` | `pandas.Series`): Power data, or the name of the column in
            :py:attr:`data`.
        data(:obj:`pandas.DataFrame`, optional): a pandas DataFrame containing
            :py:attr:`windspeed_col` and :py:attr:`power_col`. Defaults to None.

    Returns:
        :obj:`function`: Python function of type (Array[float] -> Array[float]) implementing the power curve.

    """
    return fit_parametric_power_curve(
        windspeed_col,
        power_col,
        curve=logistic5param,
        optimization_algorithm=differential_evolution,
        cost_function=least_squares,
        bounds=((1200, 1800), (-10, -1e-3), (1e-3, 30), (1e-3, 1), (1e-3, 10)),
    )


@series_method(data_cols=["windspeed_col", "power_col"])
def gam(
    windspeed_col: str | pd.Series,
    power_col: str | pd.Series,
    n_splines: int = 20,
    data: pd.DataFrame = None,
) -> Callable:
    """
    Use the generalized additive model, :py:class:`pygam.LinearGAM` to fit power to wind speed.

    Args:
        windspeed_col(:obj:`str` | `pandas.Series`): Windspeed data, or the name of the column in
            :py:attr:`data`.
        power_col(:obj:`str` | `pandas.Series`): Power data, or the name of the column in
            :py:attr:`data`.
        n_splines (:obj:`int`): Number of splines to use in the fit. Defaults to 20.
        data(:obj:`pandas.DataFrame`, optional): a pandas DataFrame containing
            :py:attr:`windspeed_col` and :py:attr:`power_col`. Defaults to None.

    Returns:
        :obj:`Callable`: Python function of type (Array[float] -> Array[float]) implementing the power curve.

    """
    # Fit the model
    return LinearGAM(n_splines=n_splines).fit(windspeed_col.values, power_col.values).predict


@dataframe_method(data_cols=["windspeed_col", "wind_direction_col", "air_density_col", "power_col"])
def gam_3param(
    windspeed_col: str | pd.Series,
    wind_direction_col: str | pd.Series,
    air_density_col: str | pd.Series,
    power_col: str | pd.Series,
    n_splines: int = 20,
    data: pd.DataFrame = None,
) -> Callable:
    """
    Use a generalized additive model to fit power to wind speed, wind direction and air density.

    Args:
        windspeed_col(:obj:`str` | `pandas.Series`): Windspeed data, or the name of the column in
            :py:attr:`data`.
        wind_direction_col(:obj:`str` | `pandas.Series`): Wind direction data, or the name of the
            column in :py:attr:`data`.
        air_density_col(:obj:`str` | `pandas.Series`): Air density data, or the name of the column
            in :py:attr:`data`.
        power_col(:obj:`str` | `pandas.Series`): Power data, or the name of the column in
            :py:attr:`data`.
        n_splines (:obj:`int`): Number of splines to use in the fit. Defaults to 20.
        data(:obj:`pandas.DataFrame`, optional): a pandas DataFrame containing
            :py:attr:`windspeed_col`, :py:attr:`wind_direction_col`, :py:attr:`air_density_col`,
            and :py:attr:`power_col`. Defaults to None.

    Returns:
        :obj:`Callable`: Python function of type (Array[float] -> Array[float]) implementing the power curve.
    """
    # create dataframe input to LinearGAM and predicted response variable
    X = data[[windspeed_col, wind_direction_col, air_density_col]]
    y = data[power_col]

    # Fit the model
    model = LinearGAM(n_splines=n_splines).fit(X, y)

    # Wrap the prediction function in a closure to pack input variables
    @dataframe_method(data_cols=["windspeed_col", "wind_direction_col", "air_density_col"])
    def predict(
        windspeed_col: str | pd.Series,
        wind_direction_col: str | pd.Series,
        air_density_col: str | pd.Series,
        data: pd.DataFrame = None,
    ):
        X = data[[windspeed_col, wind_direction_col, air_density_col]]
        return model.predict(X)

    return predict
