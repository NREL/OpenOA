"""
This module provides methods for processing meteorological data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.constants as const

from openoa.utils._converters import df_to_series


# Define constants used in some of the methods
R = 287.058  # Gas constant for dry air, units of J/kg/K
Rw = 461.5  # Gas constant of water vapour, unit J/kg/K


def compute_wind_direction(
    u: pd.Series | str, v: pd.Series | str, data: pd.DataFrame = None
) -> pd.Series:
    """Compute wind direction given u and v wind vector components

    Args:
        u(:obj:`pandas.Series` | `str`): A pandas `Series` of the zonal component of the wind,
            in m/s, or the name of the column in `data`.
        v(:obj:`pandas.Series` | `str`): A pandas `Series` of the meridional component of the wind,
            in m/s, or the name of the column in `data`.
        data(:obj:`pandas.DataFrame`): The pandas DataFrame containg the columns `u` and `v`.

    Returns:
        :obj:`pandas.Series`: wind direction; units of degrees
    """
    if data is not None:
        u, v = df_to_series(data, u, v)

    wd = 180 + np.arctan2(u, v) * 180 / np.pi  # Calculate wind direction in degrees
    return pd.Series(np.where(wd != 360, wd, 0))


def compute_u_v_components(
    wind_speed: pd.Series | str, wind_dir: pd.Series | str, data: pd.DataFrame = None
) -> pd.Series:
    """Compute vector components of the horizontal wind given wind speed and direction

    Args:
        wind_speed(pandas.Series): A pandas `Series` of the horizontal wind speed, in m/s, or the
            name of the column in `data`.
        wind_dir(pandas.Series): A pandas `Series` of the wind direction, in degrees, or the name of
            the column in `data`.

    Returns:
        (tuple):
            u(pandas.Series): the zonal component of the wind; units of m/s.
            v(pandas.Series): the meridional component of the wind; units of m/s
    """
    if data is not None:
        wind_speed, wind_dir = df_to_series(data, wind_speed, wind_dir)

    # Send exception if any negative data found
    if (wind_speed[wind_speed < 0].size > 0) | (wind_dir[wind_dir < 0].size > 0):
        raise Exception("Some of your wind speed or direction data is negative. Check your data")

    u = np.round(-wind_speed * np.sin(wind_dir * np.pi / 180), 10)
    v = np.round(-wind_speed * np.cos(wind_dir * np.pi / 180), 10)

    return u, v


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
        temp_col(:obj:`array-like`): array with temperature values; units of Kelvin
        pres_col(:obj:`array-like`): array with pressure values; units of Pascals
        humi_col(:obj:`array-like`): optional array with relative humidity values; dimensionless (range 0 to 1)

    Returns:
        :obj:`pandas.Series`: Rho, calcualted air density; units of kg/m3
    """
    if data is not None:
        temp_col, pres_col, humi_col = df_to_series(data, temp_col, pres_col, humi_col)
    # Check if humidity column is provided and create default humidity array with values of 0.5 if necessary
    if humi_col is not None:
        rel_humidity = humi_col
    else:
        rel_humidity = np.full(temp_col.shape[0], 0.5)

    # Send exception if any negative data found
    if np.any(temp_col < 0) | np.any(pres_col < 0) | np.any(rel_humidity < 0):
        raise Exception(
            "Some of your temperature, pressure or humidity data is negative. Check your data."
        )

    rho = (1 / temp_col) * (
        pres_col / R - rel_humidity * (0.0000205 * np.exp(0.0631846 * temp_col)) * (1 / R - 1 / Rw)
    )

    return rho


def pressure_vertical_extrapolation(p0, temp_avg, z0, z1):
    """
    Extrapolate pressure from height z0 to height z1 given the average temperature in the layer.
    The hydostatic equation is used to peform the extrapolation.

    Args:
        p0(:obj:`pandas.Series`): pressure at height z0; units of Pascals
        temp_avg(:obj:`pandas.Series`): mean temperature between z0 and z1; units of Kelvin
        z0(:obj:`pandas.Series`): height above surface; units of meters
        z1(:obj:`pandas.Series`): extrapolation height; units of meters

    Returns:
        :obj:`pandas.Series`: p1, extrapolated pressure at z1; units of Pascals
    """
    # Send exception if any negative data found
    if (p0[p0 < 0].size > 0) | (temp_avg[temp_avg < 0].size > 0):
        raise Exception("Some of your temperature of pressure data is negative. Check your data")

    p1 = p0 * np.exp(-const.g * (z1 - z0) / R / temp_avg)  # Pressure at z1

    return p1


def air_density_adjusted_wind_speed(wind_col, density_col):
    """
    Apply air density correction to wind speed measurements following IEC-61400-12-1 standard

    Args:
        wind_col(:obj:`str`): array containing the wind speed data; units of m/s
        density_col(:obj:`str`): array containing the air density data; units of kg/m3

    Returns:
        :obj:`pandas.Series`: density-adjusted wind speeds; units of m/s
    """
    rho_mean = density_col.mean()  # Mean air density across sample
    dens_adjusted_ws = wind_col * np.power(
        density_col / rho_mean, 1.0 / 3
    )  # Density adjusted wind speeds

    return dens_adjusted_ws


def compute_turbulence_intensity(mean_col, std_col):
    """
    Compute turbulence intensity

    Args:
        mean_col(:obj:`array`): array containing the wind speed mean  data; units of m/s
        std_col(:obj:`array`): array containing the wind speed standard deviation data; units of m/s

    Returns:
        :obj:`array`: turbulence intensity, (unitless ratio)
    """
    return std_col / mean_col


def compute_shear(
    data: pd.DataFrame, ws_heights: dict, return_reference_values: bool = False
) -> np.array:
    """
    Computes shear coefficient between wind speed measurements using the power law.
    The shear coefficient is obtained by evaluating the expression for an OLS regression coefficient.

    Updated version targeting OpenOA V3 due to the following api breaking change:
        - Removal of ref_col, instead, returning the reference column used

    Args:
        data(:obj:`pandas.DataFrame`): dataframe with wind speed columns
        ws_heights(:obj:`dict`): keys are strings of columns in <data> containing wind speed data, values are
                                 associated sensor heights (m)
        return_reference_values(:obj: `bool`): If True, this function returns a three element tuple where the
                                               first element is the array of shear exponents, the second element
                                               is the reference height (float), and the third element is the
                                               reference wind speed (array). These reference values can be used
                                               for extrapolating wind speed. Defaults to False.

    Returns:
        If return_reference_values is False (default):
        :obj:`numpy.array`: shear coefficient (unitless)

        If return_reference_values is True:
        :obj:`tuple`: (shear coefficient (unitless), reference height (m), reference wind speed)

    """

    # create "u" 2-D array; where element [i,j] is the wind speed measurement at the ith timestep and jth sensor height
    u: np.ndarray = data[ws_heights.keys()].to_numpy()

    # create "z" 2_D array; columns are filled with the sensor height
    n: int = len(data)
    heights: list = [np.full(n, height) for height in ws_heights.values()]
    z: np.ndarray = np.column_stack(tuple(heights))

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


def extrapolate_windspeed(v1, z1: float, z2: float, shear):
    """
    Extrapolates wind speed vertically using the Power Law.

    Args:
        v1(:obj: `pandas.Series` | `numpy.array` | `float`): Wind speed measurements at the reference height.
        z1(:obj:`float`): Height of reference wind speed measurements; units in meters
        z2(:obj:`float`): Target extrapolation height; units in meters
        shear(:obj: `pandas.Series` | `numpy.array` | `float`): Shear value(s)

    Returns:
        :obj: (`pandas.Series` | `numpy.array` | `float`): Wind speed extrapolated to target height.
    """

    target_ws = v1 * (z2 / z1) ** shear

    return target_ws


def compute_veer(wind_a, height_a, wind_b, height_b):
    """
    Compute veer between wind direction measurements

    Args:
        wind_a, wind_b(:obj:`array`): arrays containing the wind direction mean data; units of deg
        height_a, height_b(:obj:`array`): sensor heights (m)

    Returns:
        veer(:obj:`array`): veer (deg/m)
    """

    # Calculate wind direction change
    delta_dir = wind_b - wind_a

    # Convert absolute values greater than 180 to normal range
    delta_dir[delta_dir > 180] = delta_dir[delta_dir > 180] - 360.0
    delta_dir[delta_dir <= (-180)] = delta_dir[delta_dir <= (-180)] + 360.0

    return delta_dir / (height_b - height_a)
