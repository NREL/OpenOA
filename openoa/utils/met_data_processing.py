"""
This module provides methods for processing meteorological data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.constants as const

from openoa.utils._converters import df_to_series, series_method


# Define constants used in some of the methods
R = 287.058  # Gas constant for dry air, units of J/kg/K
Rw = 461.5  # Gas constant of water vapour, unit J/kg/K


def wrap_180(x: float | np.ndarray | pd.Series | pd.DataFrame):
    """
    Converts an angle, an array of angles, or a pandas Series or DataFrame of angles in degrees to
    the range -180 to +180 degrees.

    Args:
        x (float | np.ndarray | pd.Series | pd.DataFrame): Input angle(s) (degrees)
    Returns:
        float | np.ndarray: The input angle(s) converted to the range -180 to +180 degrees, returned
            as a float or numpy array (degrees)
    """
    input_type = type(x)
    if (input_type == pd.core.series.Series) | (input_type == pd.core.frame.DataFrame):
        x = x.values

    input_size = np.size(x)

    x = x % 360.0
    x = np.where(x > 180.0, x - 360.0, x)

    return x if input_size > 1 else float(x)


def circular_mean(x: pd.DataFrame | pd.Series | np.ndarray, axis: int = 0):
    """
    Compute circular mean of wind direction data for a pandas Series or 1-dimensional numpy array,
    or along any dimension of a multi-dimensional pandas DataFrame or numpy array

    Args:
        x(pd.DataFrame | pd.Series | np.ndarray): A pandas DataFrame or Series, or a numpy array
            containing wind direction data in degrees.
        axis(int): The axis to which the circular mean will be applied. This value must be less than
            the number of dimensions in x. Defaults to 0.

    Returns:
        pd.Series | float | np.ndarray: The circular mean of the wind directions along the specified
            axis between 0 and 360 degrees (degrees).
    """
    if axis >= x.ndim:
        raise ValueError("The axis argument cannot be greater than the dimension of the data (x).")

    return (
        np.degrees(
            np.arctan2(
                np.sin(np.radians(x)).mean(axis=axis),
                np.cos(np.radians(x)).mean(axis=axis),
            )
        )
        % 360.0
    )


@series_method(data_cols=["u", "v"])
def compute_wind_direction(
    u: pd.Series | str, v: pd.Series | str, data: pd.DataFrame = None
) -> pd.Series:
    """Compute wind direction given u and v wind vector components

    Args:
        u(:obj:`pandas.Series` | `str`): A pandas `Series` of the zonal component of the wind,
            in m/s, or the name of the column in `data`.
        v(:obj:`pandas.Series` | `str`): A pandas `Series` of the meridional component of the wind,
            in m/s, or the name of the column in `data`.
        data(:obj:`pandas.DataFrame`): The pandas `DataFrame` containg the columns `u` and `v`.

    Returns:
        :obj:`pandas.Series`: wind direction; units of degrees
    """
    wd = 180 + np.arctan2(u, v) * 180 / np.pi  # Calculate wind direction in degrees
    return pd.Series(np.where(wd != 360, wd, 0))


@series_method(data_cols=["wind_speed", "wind_dir"])
def compute_u_v_components(
    wind_speed: pd.Series | str, wind_dir: pd.Series | str, data: pd.DataFrame = None
) -> pd.Series:
    """Compute vector components of the horizontal wind given wind speed and direction

    Args:
        wind_speed(:obj:`pandas.Series` | `str`): A pandas `Series` of the horizontal wind speed, in
            m/s, or the name of the column in `data`.
        wind_dir(:obj:`pandas.Series` | `str`): A pandas `Series` of the wind direction, in degrees,
            or the name of the column in `data`.
        data(:obj:`pandas.DataFrame`): The pandas `DataFrame` containg the columns `wind_speed` and
            `wind_direction`.

    Raises:
        ValueError: Raised if any of the `wind_speed` or `wind_dir` values are negative.

    Returns:
        (tuple):
            u(pandas.Series): the zonal component of the wind; units of m/s.
            v(pandas.Series): the meridional component of the wind; units of m/s
    """
    if np.any(wind_speed < 0):
        raise ValueError("Negative values exist in the `wind_speed` data.")
    if np.any(wind_dir < 0):
        raise ValueError("Negative values exist in the `wind_dir` data.")

    u = np.round(-wind_speed * np.sin(wind_dir * np.pi / 180), 10)
    v = np.round(-wind_speed * np.cos(wind_dir * np.pi / 180), 10)

    return u, v


@series_method(data_cols=["temp_col", "pres_col", "humi_col"])
def compute_air_density(
    temp_col: pd.Series | str,
    pres_col: pd.Series | str,
    humi_col: pd.Series | str = None,
    data: pd.DataFrame = None,
):
    """
    Calculate air density from the ideal gas law based on the definition provided by IEC 61400-12
    given pressure, temperature and relative humidity.

    This function assumes temperature and pressure are reported in standard units of measurement
    (i.e. Kelvin for temperature, Pascal for pressure, humidity has no dimension).

    Humidity values are optional. According to the IEC a humiditiy of 50% (0.5) is set as default value.

    Args:
        temp_col(:obj:`pandas.Series` | `str`): A pandas `Series` of the temperature values, in
            Kelvin, or the name of the column in `data`.
        pres_col(:obj:`pandas.Series` | `str`): A pandas `Series` of the pressure values, in Pascals,
            or the name of the column in `data`.
        humi_col(:obj:`pandas.Series` | `str`): An optional pandas `Series` of the relative humidity
            values, as a decimal in the range (0, 1), or the name of the column in `data`, by default
            None.
        data(:obj:`pandas.DataFrame`): The pandas `DataFrame` containg the columns `temp_col` and
            `pres_col`, and optionally `humi_col`.

    Raises:
        ValueError: Raised if any of the `temp_col` or `pres_col`, or `humi_col` values are negative.

    Returns:
        :obj:`pandas.Series`: Rho, calcualted air density; units of kg/m3
    """
    if data is not None:
        temp_col, pres_col, humi_col = df_to_series(data, temp_col, pres_col, humi_col)
    # Check if humidity column is provided and create default humidity array with values of 0.5 if necessary
    rel_humidity = humi_col if humi_col is not None else np.full(temp_col.shape[0], 0.5)

    if np.any(temp_col < 0):
        raise ValueError("Negative values exist in the temperature data.")
    if np.any(pres_col < 0):
        raise ValueError("Negative values exist in the pressure data.")
    if np.any(rel_humidity < 0):
        raise ValueError("Negative values exist in the humidity data.")

    rho = (1 / temp_col) * (
        pres_col / R - rel_humidity * (0.0000205 * np.exp(0.0631846 * temp_col)) * (1 / R - 1 / Rw)
    )

    return rho


@series_method(data_cols=["p0", "temp_avg", "z0", "z1"])
def pressure_vertical_extrapolation(
    p0: pd.Series | str,
    temp_avg: pd.Series | str,
    z0: pd.Series | str,
    z1: pd.Series | str,
    data: pd.DataFrame = None,
) -> pd.Series:
    """
    Extrapolate pressure from height z0 to height z1 given the average temperature in the layer.
    The hydostatic equation is used to peform the extrapolation.

    Args:
        p0(:obj:`pandas.Series`): A pandas `Series` of the pressure at height z0, in Pascals, or the
            name of the column in `data`.
        temp_avg(:obj:`pandas.Series`): A pandas `Series` of the mean temperature between z0 and z1,
            in Kelvin, or the name of the column in `data`.
        z0(:obj:`pandas.Series`): A pandas `Series` of the height above surface, in meters, or the
            name of the column in `data`.
        z1(:obj:`pandas.Series`): A pandas `Series` of the extrapolation height, in meters, or the
            name of the column in `data`.
        data(:obj:`pandas.DataFrame`): The pandas `DataFrame` containg the columns `p0`, `temp_avg`,
            `z0`, and `z1`.

    Raises:
        ValueError: Raised if any of the `p0` or `temp_avg` values are negative.

    Returns:
        :obj:`pandas.Series`: p1, extrapolated pressure at z1, in Pascals
    """
    if np.any(p0 < 0):
        raise ValueError("Negative values exist in the `p0` data.")
    if np.any(temp_avg < 0):
        raise ValueError("Negative values exist in the `temp_avg` data.")

    return p0 * np.exp(-const.g * (z1 - z0) / R / temp_avg)  # Pressure at z1


@series_method(data_cols=["wind_col", "density_col"])
def air_density_adjusted_wind_speed(
    wind_col: pd.Series | str, density_col: pd.Series | str, data: pd.DataFrame = None
) -> pd.Series:
    """
    Apply air density correction to wind speed measurements following IEC-61400-12-1 standard

    Args:
        wind_col(:obj:`pandas.Series` | `str`): A pandas `Series` containing the wind speed data,
            in m/s, or the name of the column in `data`
        density_col(:obj:`pandas.Series` | `str`): A pandas `Series` containing the air density data,
            in kg/m3, or the name of the column in `data`
        data(:obj:`pandas.DataFrame`): The pandas `DataFrame` containg the columns `wind_col` and
            `density_col`.

    Returns:
        :obj:`pandas.Series`: density-adjusted wind speeds, in m/s
    """
    return wind_col * np.power(density_col / density_col.mean(), 1.0 / 3)


@series_method(data_cols=["mean_col", "std_col"])
def compute_turbulence_intensity(
    mean_col: pd.Series | str, std_col: pd.Series | str, data: pd.DataFrame = None
) -> pd.Series:
    """
    Compute turbulence intensity

    Args:
        mean_col(:obj:`pandas.Series` | `str`): A pandas `Series` containing the wind speed mean
            data, in m/s, or the name of the column in `data`.
        std_col(:obj:`pandas.Series` | `str`): A pandas `Series` containing the wind speed standard
            deviation data, in m/s, or the name of the column in `data`.
        data(:obj:`pandas.DataFrame`): The pandas `DataFrame` containg the columns `mean_col` and `std_col`.

    Returns:
        :obj:`pd.Series`: turbulence intensity, (unitless ratio)
    """
    if data is not None:
        mean_col, std_col = df_to_series(data, mean_col, std_col)
    return std_col / mean_col


def compute_shear(
    data: pd.DataFrame, ws_heights: dict[str, float], return_reference_values: bool = False
) -> pd.Series | tuple[pd.Series, float, pd.Series]:
    """
    Computes shear coefficient between wind speed measurements using the power law.
    The shear coefficient is obtained by evaluating the expression for an OLS regression coefficient.

    Args:
        data(:obj:`pandas.DataFrame`): A pandas `DataFrame` with wind speed columns that correspond
            to the keys of `ws_heights`.
        ws_heights(:obj:`dict[str, float]`): A dictionary with wind speed column names of `data` as
            keys and their respective sensor heights (m) as values.
        return_reference_values(:obj: `bool`): If True, this function returns a three element tuple
            where the first element is the array of shear exponents, the second element is the
            reference height (float), and the third element is the array of reference wind speeds.
            These reference values can be used for extrapolating wind speed. Defaults to False.

    Returns:
        :obj:`pandas.Series` | :obj:`tuple[pandas.Series, float, pandas.Series]`: If
            :py:attr:`return_reference_values` is False, return just the shear coefficient
            (unitless), else return the shear coefficent (unitless), reference height (m), and
            reference wind speed (m/s).
    """

    # Extract the wind speed columns from `data` and create "u" 2-D array; where element
    # [i,j] is the wind speed measurement at the ith timestep and jth sensor height
    u: np.ndarray = np.column_stack(df_to_series(data, *ws_heights))

    # create "z" 2_D array; columns are filled with the sensor height
    z: np.ndarray = np.repeat([[*ws_heights.values()]], len(data), axis=0)

    # take log of z & u
    with warnings.catch_warnings():  # suppress log division by zero warning.
        warnings.filterwarnings("ignore", r"divide by zero encountered in log")
        u = np.log(u)
        z = np.log(z)

    # correct -inf or NaN if any.
    nan_or_ninf = np.logical_or(np.isneginf(u), np.isnan(u))
    if np.any(nan_or_ninf):
        # replace -inf or NaN with zero or NaN in u and corresponding location in z such that these
        # elements are excluded from the regression.
        u[nan_or_ninf] = 0
        z[nan_or_ninf] = np.nan

    # shift rows of z by the mean of z to simplify shear calculation
    z = z - (np.nanmean(z, axis=1))[:, None]

    # finally, replace NaN's in z by zero so those points are effectively excluded from the regression
    z[np.isnan(z)] = 0

    # compute shear based on simple linear regression
    alpha = (z * u).sum(axis=1) / (z * z).sum(axis=1)

    if not return_reference_values:
        return alpha

    else:
        # compute reference height
        z_ref: float = np.exp(np.mean(np.log(np.array(list(ws_heights.values())))))

        # replace zeros in u (if any) with NaN
        u[u == 0] = np.nan

        # compute reference wind speed
        u_ref = np.exp(np.nanmean(u, axis=1))

        return alpha, z_ref, u_ref


@series_method(data_cols=["v1", "shear"])
def extrapolate_windspeed(
    v1: pd.Series | str, z1: float, z2: float, shear: pd.Series | str, data: pd.DataFrame = None
):
    """
    Extrapolates wind speed vertically using the Power Law.

    Args:
        v1(:obj: `pandas.Series` | `float` | `str`): A pandas `Series` of the wind
            speed measurements at the reference height, or the name of the column in `data`.
        z1(:obj:`float`): Height of reference wind speed measurements; units in meters
        z2(:obj:`float`): Target extrapolation height; units in meters
        shear(:obj: `pandas.Series` | `float` | `str`): A pandas `Series` of the shear
            values, or the name of the column in `data`.
        data(:obj:`pandas.DataFrame`): The pandas `DataFrame` containg the columns `v1` and `shear`.

    Returns:
        :obj: (`pandas.Series` | `numpy.array` | `float`): Wind speed extrapolated to target height.
    """
    return v1 * (z2 / z1) ** shear


@series_method(data_cols=["wind_a", "wind_b"])
def compute_veer(
    wind_a: pd.Series | str,
    height_a: float,
    wind_b: pd.Series | str,
    height_b: float,
    data: pd.DataFrame = None,
):
    """
    Compute veer between wind direction measurements

    Args:
        wind_a(:obj:`pandas.Series` | `str`): A pandas `Series` containing the wind direction mean
            data, in degrees, or the name of the column in `data`.
        height_a(:obj:`float`): sensor height for `wind_a`
        wind_b(:obj:`pandas.Series` | `str`): A pandas `Series` containing the wind direction mean
            data, in degrees, or the name of the column in `data`.
        height_b(:obj:`float`): sensor height for `wind_b`
        data(:obj:`pandas.DataFrame`): The pandas `DataFrame` containg the columns `wind_a`, and `wind_b`.

    Returns:
        veer(:obj:`array`): veer (deg/m)
    """
    # Calculate wind direction change
    delta_dir = wind_b - wind_a

    # Convert absolute values greater than outside 180 to a normal range
    delta_dir = delta_dir.where(delta_dir <= 180, delta_dir - 360.0).where(
        delta_dir > -180, delta_dir + 360.0
    )

    return delta_dir / (height_b - height_a)
