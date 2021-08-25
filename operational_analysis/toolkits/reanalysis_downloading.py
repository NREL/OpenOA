"""
This module provides functions for downloading long-term historical atmospheric data from the
MERRA-2 and ERA5 reanalysis products and returning the data as pandas DataFrames or saving the
data in csv files. Currently, the module supports downloading hourly reanalysis data for a time
period of interest using the PlanetOS API. PlanetOS (https://planetos.com) is a service by
Intertrust Technologies that provides access to a variety of weather, climate, and environmental
datasets, including MERRA-2 and ERA5.

To use this module to download data through the PlanetOS API, users must first create a PlanetOS
account (https://data.planetos.com/plans). Once an account is created, an API key will be provided,
which can be found in Account Settings. They API key should be saved in a text file, the location
of which is passed as an argument to the downloading functions in this module. By default, it is
assumed that the API key is saved in a file called "APIKEY" located in the OpenOA toolkits
directory.

More information about PlanetOS data products can be found here:
http://docs.planetos.com/#planet-os-product-guide

Alternatively, users can download MERRA-2 data directly from the NASA Goddard Earth Sciences Data
and Information Services Center (GES DISC) and ERA5 data directly from the Copernicus Climate Data
Store (CDS) service. However, this module does not currently contain functions for automating the
downloading process.

-Hourly MERRA-2 data can be downloaded directly from NASA GES DISC by selecting the
"Subset / Get Data" link on the following webpage:
https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary. Specific dates, variables, and
coordinates can be selected using the OPeNDAP or GES DISC Subsetter download methods.

-Hourly ERA5 data can be downloaded using either the CDS web interface or the CDS API, as explained
here: https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5. Data for specific dates,
variables, and coordinates can be downloaded using the CDS web interface via the "Download data"
tab here: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview.
Instructions for using the CDS Toolbox API to download ERA5 data programatically can be found here:
https://cds.climate.copernicus.eu/toolbox/doc/how-to/1_how_to_retrieve_data/1_how_to_retrieve_data.html
(note that the "reanalysis-era5-single-levels" dataset should be used).

For reanalysis data downloaded directly from NASA or Copernicus, to import the data into an OpenOA
PlantData object, used by the OpenOA analysis methods, the data should be re-saved as a single csv
file with a separate row for each timestamp and columns corresponding to the variables of interest
(including a timestamp column).

Although any combination of available reanalysis variables can be downloaded, several of the OpenOA
analysis methods require the eastward and northward wind components, temperature, and surface
pressure (temperature and surface pressure can be used to calculate air density) over a 20-year period.
"""

import os
import datetime

import numpy as np
import pandas as pd
import requests


# Maps OpenOA-standard reanalysis product names to PlanetOS datasets
dataset_names = {"merra2": "nasa_merra2_global", "era5": "ecmwf_era5_v2"}


def get_default_var_dicts_planetos(dataset):
    """
    Returns dictionary mapping PlanetOS variable names to OpenOA variable names for a particular dataset

    Args:
        dataset (:obj:`string`): Dataset name ("merra2" or "era5")

    Returns:
        :obj:`dict`: Dictionary mapping reanalysis variable names from PlanetOS to standard OpenOA variable names
    """

    if dataset == "merra2":
        var_dict = {"U50M": "u_50", "V50M": "v_50", "T2M": "t_2m", "PS": "surf_pres"}
    elif dataset == "era5":
        var_dict = {
            "eastward_wind_at_100_metres": "u_100",
            "northward_wind_at_100_metres": "v_100",
            "air_temperature_at_2_metres": "t_2m",
            "surface_air_pressure": "surf_pres",
        }
    else:
        var_dict = None

    return var_dict


def get_dataset_st_end_dates_planetos(dataset, apikey):
    """
    Returns the first and last datetimes for a PlanetOS dataset

    Args:
        dataset (:obj:`string`): Dataset name ("merra2" or "era5")
        apikey (:obj:`string`): PlanetOS API key

    Returns:
        (:obj:`pandas.Timestamp`, :obj:`pandas.Timestamp`): Timestamps of first and last datetimes in PlanetOS dataset
    """

    url = "http://api.planetos.com/v1/datasets/" + dataset_names[dataset] + "/subdatasets?"

    r = requests.get(url, params={"apikey": apikey})

    st_date = pd.to_datetime(r.json()["subdatasets"][0]["temporalCoverage"]["start"] * 1e6)
    en_date = pd.to_datetime(r.json()["subdatasets"][0]["temporalCoverage"]["end"] * 1e6)

    return st_date, en_date


def get_st_end_dates_planetos(st_date, en_date, num_years, st_date_ds, en_date_ds):
    """
    This function determines the start and end datetimes to use for retrieving data from a dataset based on the
    combination of inputs provided by the user. The rules are as follows:
        1. If st_date and en_date are both defined, they will be used as the final start and end dates but will be
           adjusted if they are outside of the start and end dates of the dataset (st_date_ds and en_date_ds)
        2. If st_date and en_date are both undefined, the final end date will be the end of the last full month in the
           dataset and the start date will be num_years before the end date
        3. If only st_date is defined, the final end date will be the lesser of num_years after the start date or the
           last datetime of the dataset
        4. If only en_date is defined, the final start date will be the greater of num_years before the end date or the
           first datetime of the dataset

    Args:
        st_date (:obj:`pandas.Timestamp` or :obj:`string`): Desired start datetime of reanalysis data time series
        en_date (:obj:`pandas.Timestamp` or :obj:`string`): Desired end datetime of reanalysis data time series
        num_years (:obj:`int`): Desired number of years of reanalysis data. This is only used if either st_date or
            end_date are undefined.
        st_date_ds (:obj:`pandas.Timestamp`): The first datetime in the dataset
        en_date_ds (:obj:`pandas.Timestamp`): The last datetime in the dataset

    Returns:
        (:obj:`pandas.Timestamp`, :obj:`pandas.Timestamp`): The final start and end datetimes
    """

    # Convert start and end dates to Timestamps if in another format
    if st_date is not None:
        st_date_new = pd.to_datetime(st_date)

    # Add one hour since the last time step will not be retrieved
    if en_date is not None:
        en_date_new = pd.to_datetime(en_date) + datetime.timedelta(hours=1)

    # Determine start and end dates if both are not defined
    if (st_date is not None) & (en_date is None):
        # Set end date to num_years years after start date

        # Check to see if start date is out of bounds for the dataset
        if st_date_new < st_date_ds:
            print("Start date is out of range. Changing to " + str(st_date_ds))
            st_date_new = st_date_ds

        # Handle rare leap year case where start date is on the last day of February
        try:
            en_date_new = st_date_new.replace(year=st_date_new.year + num_years)
        except ValueError:
            en_date_new = st_date_new.replace(month=2, day=28, year=st_date_new.years + num_years)

    elif (st_date is None) & (en_date is not None):
        # Set start date to num_years years before end date

        # Check to see if end date is out of bounds for the dataset
        if (en_date_new - datetime.timedelta(hours=1)) > en_date_ds:
            print("End date is out of range. Changing to " + str(en_date_ds))
            en_date_new = en_date_ds + datetime.timedelta(hours=1)

        # Handle rare leap year case where start date is on the last day of February
        try:
            st_date_new = en_date_new.replace(year=en_date_new.year - num_years)
        except ValueError:
            st_date_new = en_date_new.replace(month=2, day=28, year=en_date_new.years - num_years)

    elif (st_date is None) & (en_date is None):
        # Find end of last complete month and set start date to num_years before
        # Check if last date is the end of a month

        if en_date_ds.month == (en_date_ds + datetime.timedelta(hours=1)).month:
            en_date_new = en_date_ds.replace(day=1, hour=0, minute=0)
        else:
            en_date_new = (en_date_ds + datetime.timedelta(hours=1)).replace(minute=0)

        st_date_new = en_date_new.replace(year=en_date_new.year - num_years)
    # else: start and end dates are defined already, so don't need to do anything

    # Check once more to see if start and end dates are out of bounds for the dataset
    if st_date_new < st_date_ds:
        print("Start date is out of range. Changing to " + str(st_date_ds))
        st_date_new = st_date_ds

        # If start date minute changes, update end date too
        en_date_new = en_date_new.replace(minute=st_date_new.minute)

    if (en_date_new - datetime.timedelta(hours=1)) > en_date_ds:
        print("End date is out of range. Changing to " + str(en_date_ds))
        en_date_new = en_date_ds + datetime.timedelta(hours=1)

    return st_date_new, en_date_new


def convert_resp_to_df(r, var_dict):
    """
    This function converts an API request Response object to a pandas dataframe containing the desired variables
    renamed to standard OpenOA variable names.

    Args:
        r (:obj:`requests.Response`): An API request response object
        var_dict (:obj:`dict`): Dictionary mapping reanalysis variable names from PlanetOS to standard OpenOA variable
            names

    Returns:
        :obj:`pandas.DataFrame`: A dataframe containing the variables specified in var_dict
    """

    df = pd.json_normalize(r.json()["entries"])
    df["datetime"] = pd.to_datetime(df["axes.time"])

    # Convert to standard variable names
    df.rename(columns={"data." + str(key): val for key, val in var_dict.items()}, inplace=True)

    # Limit to relevant columns
    df = df[["datetime"] + list(var_dict.values())]

    return df


def download_reanalysis_data_planetos(
    dataset,
    lat,
    lon,
    st_date=None,
    en_date=None,
    num_years=20,
    var_dict=None,
    save_pathname=None,
    save_filename=None,
    apikey_file=None,
):
    """
    This function downloads ERA5 or MERRA2 reanalysis data for a specific lat/lon using the PlanetOS API, returning a
    dataframe and, optionally, saving the data as a csv file. Only the indicated variables are downloaded and the start
    and end datetimes of the data are determined based on the provided combination of st_date, en_date, and num_years
    arguments as follows:
        1. If st_date and en_date are both defined, they will be used as the final start and end dates but will be
           adjusted if they are outside of the start and end dates of the PlanetOS dataset
        2. If st_date and en_date are both undefined, the final end date will be the end of the last full month in the
           PlanetOS dataset and the start date will be num_years before the end date
        3. If only st_date is defined, the final end date will be the lesser of num_years after the start date or the
           last datetime of the PlanetOS dataset
        4. If only en_date is defined, the final start date will be the greater of num_years before the end date or the
           first dateitme of the PlanetOS dataset

    Args:
        dataset (:obj:`string`): Dataset name ("merra2" or "era5")
        lat (:obj:`float`): Latitude (degrees)
        lon (:obj:`float`): Longitude (degrees)
        st_date (:obj:`pandas.Timestamp` or :obj:`string`, optional): Desired start datetime of reanalysis data time
            series. Defaults to None.
        en_date (:obj:`pandas.Timestamp` or :obj:`string`, optional): Desired end datetime of reanalysis data time
            series. Defaults to None.
        num_years (int, optional): [description]. Desired number of years of reanalysis data. Only used if either
            st_date or en_date are undefined. Defaults to 20.
        var_dict (:obj:`dict`, optional): Dictionary mapping the desired reanalysis variable names from PlanetOS to
            standard OpenOA variable names. If undefined, default variables will be downloaded (U and V wind speeds,
            temperature, and surface air pressure). Defaults to None.
        save_pathname (:obj:`string`, optional): The path where the downloaded reanalysis data will be saved (if
            defined). Defaults to None.
        save_filename (:obj:`string`, optional): The file name used to save the downloaded reeanalysis data (if
            defined). Defaults to None.
        apikey_file (:obj:`string`, optional): The combined path and file name where the PlanetOS API key is saved. If
            undefined, will assume there is a file called "APIKEY" saved in the operational_analysis.toolkits
            directory. Defaults to None.

    Returns:
        :obj:`pandas.DataFrame`: A dataframe containing time series of the requested reanalysis variables
    """

    # Import PlanetOS apikey
    if apikey_file is None:
        apikey_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "APIKEY")
    apikey = open(apikey_file).readlines()[0].strip()

    # Get start and end dates of dataset
    st_date_ds, en_date_ds = get_dataset_st_end_dates_planetos(dataset, apikey)

    # Depending on the start date, end date, and number of years specified, calculate the appropriate start and end
    # dates
    st_date, en_date = get_st_end_dates_planetos(
        st_date, en_date, num_years, st_date_ds, en_date_ds
    )

    # Determine number of data points
    date_diff = en_date - st_date
    count = int(24 * date_diff.days + date_diff.seconds / 3600)

    # Get default variable dictionaries if not defined
    if var_dict is None:
        var_dict = get_default_var_dicts_planetos(dataset)

    # Download data from PlanetOS
    url = "http://api.planetos.com/v1/datasets/" + dataset_names[dataset] + "/point?"

    kwgs = {
        "apikey": apikey,
        "count": count,
        "lon": lon,
        "lat": lat,
        "vars": ", ".join(list(var_dict.keys())),
        "time_start": st_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "time_end": en_date.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    r = requests.get(url, params=kwgs)

    # convert to standard dataframe
    df = convert_resp_to_df(r, var_dict)

    if save_filename is not None:
        df.to_csv(os.path.join(save_pathname, save_filename + ".csv"), index=False)

    return df
