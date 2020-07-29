import numpy as np

"""
Curve fitting routines

curve + bounds
optimization algorithm
cost function
"""


def fit_parametric_power_curve(x, y, curve, optimization_algorithm, cost_function, bounds, return_params=False):
    """
    Fit curve to filtered power-windspeed data.

    Args:
        x(list[float]): independent variable
        y(list[float]): dependent variable
        curve(Function): function/lambda name for power curve desired default is curves.logistic5param
            optimization_algorithm(Function): scipy.optimize style optimization algorithm
        cost_function(Function): Python function that takes two np.array 1D of real numbers and returns a real numeric
            cost.
        bounds: bounds on parameters for power curve, default is for logistic5param, with power in kw and windspeed
            in m/s
        return_params(Boolean): True = return a tuple of (function, scipy.optimize.fit), False = just return function.

    Returns:
        Function(np.array -> np.array): function handle to optimized power curve
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


def least_squares(x, y):
    """ Least Squares loss function

    Args:
        x(np.array): 1-D array of numbers representing x
        y(np.array): 1-D array of numbers representing y

    Returns:
        real number
    """
    return np.sum((x - y) ** 2)
