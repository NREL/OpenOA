"""
Curve fitting routines

curve + bounds
optimization algorithm
cost function
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def fit_parametric_power_curve(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    curve: Callable,
    optimization_algorithm: Callable,
    cost_function: Callable,
    bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    return_params: bool = False,
):
    """
    Fit curve to filtered power-windspeed data.

    Args:
        x(:obj:`numpy.ndarray` | `pandas.Series`): independent variable
        y(:obj:`numpy.ndarray` | `pandas.Series`): dependent variable
        curve(:obj:`Callable`): function/lambda name for power curve desired default is curves.logistic5param
            optimization_algorithm(Function): scipy.optimize style optimization algorithm
        cost_function(:obj:`Callable`): Python function that takes two np.array 1D of real numbers and returns a real numeric
            cost.
        bounds(:obj:`tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]`):
            bounds on parameters for power curve, default is for logistic5param, with power in kw and windspeed in m/s
        return_params(:obj:`bool`): If True, return a tuple of (Callable, scipy.optimize.fit), and if
            False return only the Callable.

    Returns:
        Callable(np.array -> np.array): function handle to optimized power curve
    """

    # Build opt function as a closure on "x" and "y"
    def f(opt_params):
        return cost_function(curve(x, *opt_params), y)

    # Run the optimization algorithm
    fit = optimization_algorithm(f, bounds)

    # Create closure of curve function with fit params
    def fit_curve(x_2):
        return curve(x_2, *fit.x)

    # Return values based on flag
    if return_params:
        return lambda x_2: fit_curve, fit
    else:
        return fit_curve


"""
Cost Functions
"""


def least_squares(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series):
    """Least Squares loss function

    Args:
        x(:obj:`np.ndarray` | `pandas.Series`): 1-D array of numbers representing x
        y(:obj:`np.ndarray` | `pandas.Series`): 1-D array of numbers representing y

    Returns:
        The least square of x and y.
    """
    return np.sum((x - y) ** 2)
