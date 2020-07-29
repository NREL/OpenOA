from .parametric_forms import logistic5param
from .parametric_optimize import fit_parametric_power_curve, least_squares
from scipy.optimize import differential_evolution
from pygam import LinearGAM
import pandas as pd
import numpy as np

"""
This module holds ready-to-use power curve functions. They take windspeed and power columns as arguments and return a
python function which can be used to evaluate the power curve at arbitrary locations.
"""

def IEC(windspeed_column, power_column, bin_width=0.5, windspeed_start=0, windspeed_end=30.0):
    """
    Use IEC 61400-12-1-2 method for creating wind-speed binned power curve.

    Args:
        windspeed_column (:obj:`pandas.Series`): feature column
        power_column (:obj:`pandas.Series`): response column
        bin_width(:obj:`float`): width of windspeed bin, default is 0.5 m/s according to standard
        windspeed_start(:obj:`float`): left edge of first windspeed bin
        windspeed_end(:obj:`float`): right edge of last windspeed bin

    Returns:
        :obj:`function`: Python function of type (Array[float] -> Array[float]) implementing the power curve.

    """

    # Set up evenly spaced bins of fixed width, with any value over the maximum getting np.inf
    bins = np.append(np.arange(windspeed_start, windspeed_end, bin_width), [np.inf])

    # Initialize an array which will hold the mean values of each bin
    P_bin = np.ones(len(bins) - 1) * np.nan

    # Compute the mean of each bin and set corresponding P_bin
    for ibin in range(0, len(bins) - 1):
        indices = ((windspeed_column >= bins[ibin]) & (windspeed_column < bins[ibin + 1]))
        P_bin[ibin] = power_column.loc[indices].mean()

    # Linearly interpolate any missing bins
    P_bin = pd.Series(data=P_bin).interpolate(method='linear').bfill().values

    # Create a closure over the computed bins which computes the power curve value for arbitrary array-like input
    def pc_iec(x):
        P = np.zeros(np.shape(x))
        for i in range(0, len(bins) - 1):
            idx = np.where((x >= bins[i]) & (x < bins[i + 1]))
            P[idx] = P_bin[i]
        return P

    return pc_iec


def logistic_5_parametric(windspeed_column, power_column):
    """
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

    In this case, the function fits the 5 parameter logistics function to
    observed data via a least-squares optimization (i.e. minimizing the sum of
    the squares of the residual between the points as evaluated by the
    parameterized function and the points of observed data).


    Args:
        windspeed_column (:obj:`pandas.Series`): feature column
        power_column (:obj:`pandas.Series`): response column
        bin_width(:obj:`float`): width of windspeed bin, default is 0.5 m/s according to standard
        windspeed_start(:obj:`float`): left edge of first windspeed bin
        windspeed_end(:obj:`float`): right edge of last windspeed bin

    Returns:
        :obj:`function`: Python function of type (Array[float] -> Array[float]) implementing the power curve.

    """
    return fit_parametric_power_curve(windspeed_column, power_column,
                                      curve=logistic5param,
                                      optimization_algorithm=differential_evolution,
                                      cost_function=least_squares,
                                      bounds=((1200, 1800), (-10, -1e-3), (1e-3, 30), (1e-3, 1), (1e-3, 10)))


def gam(windspeed_column, power_column, n_splines=20):
    """
    Use a generalized additive model to fit power to wind speed.

    Args:
        windspeed_column (:obj:`pandas.Series`): Wind speed feature column
        power_column (:obj:`pandas.Series`): Power response column
        n_splines (:obj:`int`): number of splines to use in the fit

    Returns:
        :obj:`function`: Python function of type (Array[float] -> Array[float]) implementing the power curve.

    """
    # Fit the model
    return LinearGAM(n_splines=n_splines).\
        fit(windspeed_column.values, power_column.values).\
        predict



def gam_3param(windspeed_column, winddir_column, airdens_column, power_column, n_splines=20):
    """
    Use a generalized additive model to fit power to wind speed, wind direction and air density.

    Args:
        windspeed_column (:obj:`pandas.Series`): Wind speed feature column
        power_column (:obj:`pandas.Series`): Power response column
        winddir_column (:obj:`pandas.Series`): Optional. Wind direction feature column
        airdens_column (:obj:`pandas.Series`): Optional. Air density feature column
        n_splines (:obj:`int`): number of splines to use in the fit

    Returns:
        :obj:`function`: Python function of type (Array[float] -> Array[float]) implementing the power curve.

    """
    # create dataframe input to LinearGAM
    X = pd.DataFrame({'ws': windspeed_column,
                      'wd': winddir_column,
                      'dens': airdens_column})

    # Set response
    y = power_column.values

    # Fit the model
    s = LinearGAM(n_splines=n_splines).fit(X, y)

    # Wrap the prediction function in a closure to pack input variables
    def predict(windspeed_column, winddir_column, airdens_column):
        X = pd.DataFrame({'ws': windspeed_column,
                          'wd': winddir_column,
                          'dens': airdens_column})
        return s.predict(X)
    return predict
