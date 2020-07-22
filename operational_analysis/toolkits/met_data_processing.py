"""
This module provides methods for processing meteorological data.
"""

import numpy as np
import pandas as pd
import scipy.constants as const


def compute_wind_direction(u, v):
    """Compute wind direction given u and v wind vector components

    Args:
        u(:obj:`pandas.Series`): the zonal component of the wind; units of m/s
        v(:obj:`pandas.Series`): the meridional component of the wind; units of m/s

    Returns:
        :obj:`pandas.Series`: wind direction; units of degrees
    """
    wd = 180 + np.arctan2(u, v) * 180 / np.pi  # Calculate wind direction in degrees
    wd[wd == 360] = 0  # Make calculations of 360 equal to 0

    return wd


def compute_u_v_components(wind_speed, wind_dir):
    """Compute vector components of the horizontal wind given wind speed and direction

    Args:
        wind_speed(pandas.Series): horizontal wind speed; units of m/s
        wind_dir(pandas.Series): wind direction; units of degrees

    Returns:
        (tuple):
            u(pandas.Series): the zonal component of the wind; units of m/s.
            v(pandas.Series): the meridional component of the wind; units of m/s
    """
    # Send exception if any negative data found
    if (wind_speed[wind_speed < 0].size > 0) | (wind_dir[wind_dir < 0].size > 0):
        raise Exception('Some of your wind speed or direction data is negative. Check your data')

    u = np.round(-wind_speed * np.sin(wind_dir * np.pi / 180), 10)  # round to 10 digits
    v = np.round(-wind_speed * np.cos(wind_dir * np.pi / 180), 10)

    return u, v


def compute_air_density(temp_col, pres_col, humi_col = None):
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
    # Check if humidity column is provided and create default humidity array with values of 0.5 if necessary
    if humi_col != None:
        rel_humidity = humi_col
    else:
        rel_humidity = np.repeat(.5, temp_col.shape[0])
    # Send exception if any negative data found
    if np.any(temp_col < 0) | np.any(pres_col < 0) | np.any(rel_humidity < 0):
        raise Exception('Some of your temperature, pressure or humidity data is negative. Check your data.')

    #protect against python 2 integer division rules
    temp_col = temp_col.astype(float)
    pres_col = pres_col.astype(float)

    R_const = 287.05  # Gas constant for dry air, units of J/kg/K
    Rw_const = 461.5   # Gas constant of water vapour, unit J/kg/K
    rho = ((1/temp_col)*(pres_col/R_const-rel_humidity*(0.0000205*np.exp(0.0631846*temp_col))*
            (1/R_const-1/Rw_const)))

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
        raise Exception('Some of your temperature of pressure data is negative. Check your data')

    R_const = 287.058  # Gas constant for dry air, units of J/kg/K
    p1 = p0 * np.exp(-const.g * (z1 - z0) / R_const / temp_avg)  # Pressure at z1

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
    dens_adjusted_ws = wind_col * np.power(density_col / rho_mean, 1. / 3)  # Density adjusted wind speeds

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


def compute_shear(df, windspeed_heights, ref_col='empty'):
    """
    Compute shear coefficient between wind speed measurements

    Args:
        df(:obj:`pandas.DataFrame`): dataframe with wind speed columns
        windspeed_heights(:obj:`dict`): keys are strings of columns in <df> containing wind speed data, values are
            associated sensor heights (m)
        ref_col(:obj:`str`): data column name for the data to use as the normalization value; only pertinent if
            optimizing over multiple measurements

    Returns:
        :obj:`pandas.Series`: shear coefficient (unitless)
   """

    # Convert wind speed heights to float
    windspeed_heights = \
        dict(list(zip(list(windspeed_heights.keys()), [float(value) for value in list(windspeed_heights.values())])))

    keys = list(windspeed_heights.keys())
    if len(keys) <= 1:
        raise Exception('More than one wind speed measurement required to compute shear.')
    elif len(keys) == 2:
        # If there are only two measurements, no optimization possible
        wind_a = keys[0]
        wind_b = keys[1]
        height_a = windspeed_heights[wind_a]
        height_b = windspeed_heights[wind_b]
        return (np.log(df[wind_b]) - np.log(df[wind_a])) / (np.log(height_b) - np.log(height_a))
    else:
        from scipy.optimize import curve_fit

        def power_law(x, alpha):
            return (x) ** alpha

        # Normalize wind speeds and heights to reference values
        df_norm = df[keys].div(df[ref_col], axis=0)
        h0 = windspeed_heights[ref_col]
        windspeed_heights = {k: v / h0 for k, v in windspeed_heights.items()}

        # Rename columns to be windspeed measurement heights
        df_norm = df_norm.rename(columns=windspeed_heights)

        alpha = pd.DataFrame(np.ones((len(df_norm), 1)) * np.nan, index=df_norm.index, columns=['alpha'])

        # For each row
        for time in df_norm.index:

            t = (df_norm.loc[time]  # Take the row as a series, the index will be the column names,
                 .reset_index()  # Resetting the index yields the heights as a column
                 .to_numpy())  # Numpy array: each row a sensor, column 0 the heights, column 1 the measurment
            t = t[~np.isnan(t).any(axis=1)]  # Drop rows (sensors) for which the measurement was nan
            h = t[:, 0]  # The measurement heights
            u = t[:, 1]  # The measurements
            if np.shape(u)[0] <= 1:  # If less than two measurements were available, leave value as nan
                continue
            else:
                alpha.loc[time, 'alpha'] = curve_fit(power_law, h, u)[0][0]  # perform least square optimization

        return alpha['alpha']


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
    delta_dir[delta_dir > 180] = delta_dir[delta_dir > 180] - 360.
    delta_dir[delta_dir <= (-180)] = delta_dir[delta_dir <= (-180)] + 360.

    return delta_dir / (height_b - height_a)
