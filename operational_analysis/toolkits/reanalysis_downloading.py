"""
This module provides functions for downloading long-term historical atmospheric data from the
MERRA-2 and ERA5 reanalysis products and returning the data as pandas DataFrames or saving the
data in csv files. Currently, the module supports downloading hourly reanalysis data for a time
period of interest using the PlanetOS API. PlanetOS (https://planetos.com) is a service by
Intertrust Technologies that provides access to a variety of weather, climate, and environmental
datasets, including MERRA-2 and ERA5. The authors acknowledge Intertrust Technologies Corporation
for providing valuable input on portions of this code.

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

* Hourly MERRA-2 data can be downloaded directly from NASA GES DISC by selecting the
  "Subset / Get Data" link on the following webpage:
  https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary. Specific dates, variables, and
  coordinates can be selected using the OPeNDAP or GES DISC Subsetter download methods.

* Hourly ERA5 data can be downloaded using either the CDS web interface or the CDS API, as explained
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

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import operational_analysis.toolkits.met_data_processing as met


def _get_dataset_names(dataset):
    """
    Maps OpenOA-standard reanalysis product names to PlanetOS datasets

    Args:
        dataset (:obj:`string`): Dataset name ("merra2" or "era5")

    Returns:
        :obj:`string`: Corresponding PlanetOS dataset name
    """

    dataset = dataset.lower().strip()

    dataset_names = {"merra2": "nasa_merra2_global", "era5": "ecmwf_era5_v2"}

    try:
        return dataset_names[dataset]
    except KeyError as e:
        raise KeyError('Invalid dataset name. Currently, "merra2" and "era5" are supported.') from e


def _get_default_var_dicts_planetos(dataset):
    """
    Returns dictionary mapping PlanetOS variable names to OpenOA variable names for a particular dataset

    Args:
        dataset (:obj:`string`): Dataset name ("merra2" or "era5")

    Returns:
        :obj:`dict`: Dictionary mapping reanalysis variable names from PlanetOS to standard OpenOA variable names
    """

    dataset = dataset.lower().strip()

    if dataset == "merra2":
        var_dict = {"U50M": "u_ms", "V50M": "v_ms", "T2M": "temperature_K", "PS": "surf_pres_Pa"}
    elif dataset == "era5":
        var_dict = {
            "eastward_wind_at_100_metres": "u_ms",
            "northward_wind_at_100_metres": "v_ms",
            "air_temperature_at_2_metres": "temperature_K",
            "surface_air_pressure": "surf_pres_Pa",
        }
    else:
        raise ValueError('Invalid dataset name. Currently, "merra2" and "era5" are supported.')

    return var_dict


def _get_start_end_dates_planetos(start_date, end_date, num_years, start_date_ds, end_date_ds):
    """
    This function determines the start and end datetimes to use for retrieving data from a dataset based on the
    combination of inputs provided by the user. The rules are as follows:
        1. If start_date and end_date are both defined, they will be used as the final start and end dates but will be
           adjusted if they are outside of the start and end dates of the dataset (start_date_ds and end_date_ds)
        2. If start_date and end_date are both undefined, the final end date will be the end of the last full month in
           the dataset and the start date will be num_years before the end date
        3. If only start_date is defined, the final end date will be the lesser of num_years after the start date or the
           last datetime of the dataset
        4. If only end_date is defined, the final start date will be the greater of num_years before the end date or the
           first datetime of the dataset

    Args:
        start_date (:obj:`pandas.Timestamp` or :obj:`string`): Desired start datetime of reanalysis data time series
        end_date (:obj:`pandas.Timestamp` or :obj:`string`): Desired end datetime of reanalysis data time series
        num_years (:obj:`int`): Desired number of years of reanalysis data. This is only used if either start_date or
            end_date are undefined.
        start_date_ds (:obj:`pandas.Timestamp`): The first datetime in the dataset
        end_date_ds (:obj:`pandas.Timestamp`): The last datetime in the dataset

    Returns:
        (:obj:`pandas.Timestamp`, :obj:`pandas.Timestamp`): The final start and end datetimes
    """

    # Convert start and end dates to Timestamps if in another format
    if start_date is not None:
        start_date_new = pd.to_datetime(start_date)

    # Add one hour since the last time step will not be retrieved
    if end_date is not None:
        end_date_new = pd.to_datetime(end_date) + datetime.timedelta(hours=1)

    # Determine start and end dates if both are not defined
    if (start_date is not None) & (end_date is None):
        # Set end date to num_years years after start date

        # Check to see if start date is out of bounds for the dataset
        if start_date_new < start_date_ds:
            print("Start date is out of range. Changing to " + str(start_date_ds))
            start_date_new = start_date_ds

        # Handle rare leap year case where start date is on the last day of February
        try:
            end_date_new = start_date_new.replace(year=start_date_new.year + num_years)
        except ValueError:
            end_date_new = start_date_new.replace(
                month=2, day=28, year=start_date_new.year + num_years
            )

    elif (start_date is None) & (end_date is not None):
        # Set start date to num_years years before end date

        # Check to see if end date is out of bounds for the dataset
        if (end_date_new - datetime.timedelta(hours=1)) > end_date_ds:
            print("End date is out of range. Changing to " + str(end_date_ds))
            end_date_new = end_date_ds + datetime.timedelta(hours=1)

        # Handle rare leap year case where end date is on the last day of February
        try:
            start_date_new = end_date_new.replace(year=end_date_new.year - num_years)
        except ValueError:
            start_date_new = end_date_new.replace(
                month=2, day=28, year=end_date_new.year - num_years
            )

    elif (start_date is None) & (end_date is None):
        # Find end of last complete month and set start date to num_years before
        # Check if last date is the end of a month

        if end_date_ds.month == (end_date_ds + datetime.timedelta(hours=1)).month:
            end_date_new = end_date_ds.replace(day=1, hour=0, minute=0)
        else:
            end_date_new = (end_date_ds + datetime.timedelta(hours=1)).replace(minute=0)

        start_date_new = end_date_new.replace(year=end_date_new.year - num_years)
    # else: start and end dates are defined already, so don't need to do anything

    # Check once more to see if start and end dates are out of bounds for the dataset
    if start_date_new < start_date_ds:
        print("Start date is out of range. Changing to " + str(start_date_ds))
        start_date_new = start_date_ds

        # If start date minute changes, update end date too
        end_date_new = end_date_new.replace(minute=start_date_new.minute)

    if (end_date_new - datetime.timedelta(hours=1)) > end_date_ds:
        print("End date is out of range. Changing to " + str(end_date_ds))
        end_date_new = end_date_ds + datetime.timedelta(hours=1)

    # Now check to see if both the start and end dates happen to be out of bounds
    if end_date_new < start_date_ds:
        print(
            "End date is earlier than the start date of the data set. Setting end date equal to start date"
        )
        end_date_new = start_date_new
    elif start_date_new > end_date_ds:
        print(
            "Start date is later than the end date of the data set. Setting start date equal to end date"
        )
        start_date_new = end_date_new

    return start_date_new, end_date_new


def _convert_resp_to_df(r, var_names, var_dict=None):
    """
    This function converts an API request Response object to a pandas dataframe containing the desired variables,
    optionally renamed to custom variable names.

    Args:
        r (:obj:`requests.Response`): An API request response object
        var_names (:obj:`list`): List of reanalysis variable names downloaded from PlanetOS data set
        var_dict (:obj:`dict`, optional): Optional dictionary mapping any number of reanalysis variable names from
        PlanetOS to custom variable names

    Returns:
        :obj:`pandas.DataFrame`: A dataframe containing the variables specified in var_names or var_dict
    """

    try:
        df = pd.json_normalize(r.json()["entries"])
        df["datetime"] = pd.to_datetime(df["axes.time"])
    except KeyError as e:
        raise KeyError(
            "A valid API Response could not be obtained from PlanetOS. Please check the request parameters (e.g., dataset name, date range, variable names)."
        ) from e

    # remove prefix from data column names
    df.columns = df.columns.str.replace("data.", "", regex=False)

    try:
        # Limit to relevant columns
        df = df[["datetime"] + var_names]

        # If var_dict is provided, convert to custom variable names
        if var_dict is not None:
            df.rename(columns=var_dict, inplace=True)

    except KeyError as e:
        raise KeyError(
            "One or more of the desired variable names could not be obtained from PlanetOS. Please check the requested variable names."
        ) from e

    return df


def get_dataset_start_end_dates_planetos(dataset, apikey):
    """
    Returns the first and last datetimes for a PlanetOS dataset

    Args:
        dataset (:obj:`string`): Dataset name ("merra2" or "era5")
        apikey (:obj:`string`): PlanetOS API key

    Returns:
        (:obj:`pandas.Timestamp`, :obj:`pandas.Timestamp`): Timestamps of first and last datetimes in PlanetOS dataset
    """

    url = "http://api.planetos.com/v1/datasets/" + _get_dataset_names(dataset) + "/subdatasets?"

    r = requests.get(url, params={"apikey": apikey})

    start_date = pd.to_datetime(r.json()["subdatasets"][0]["temporalCoverage"]["start"] * 1e6)
    end_date = pd.to_datetime(r.json()["subdatasets"][0]["temporalCoverage"]["end"] * 1e6)

    return start_date, end_date


def download_reanalysis_data_planetos(
    dataset,
    lat,
    lon,
    start_date=None,
    end_date=None,
    num_years=20,
    var_names=None,
    var_dict=None,
    calc_derived_vars=False,
    save_pathname=None,
    save_filename=None,
    apikey_file=None,
    **api_kwargs,
):
    """
    This function downloads ERA5 or MERRA2 reanalysis data for a specific lat/lon using the PlanetOS API, returning a
    dataframe and, optionally, saving the data as a csv file. Only the indicated variables are downloaded and the start
    and end datetimes of the data are determined based on the provided combination of start_date, end_date, and
    num_years arguments as follows:

    1. If start_date and end_date are both defined, they will be used as the final start and end dates but will be
       adjusted if they are outside of the start and end dates of the PlanetOS dataset
    2. If start_date and end_date are both undefined, the final end date will be the end of the last full month in
       the PlanetOS dataset and the start date will be num_years before the end date
    3. If only start_date is defined, the final end date will be the lesser of num_years after the start date or the
       last datetime of the PlanetOS dataset
    4. If only end_date is defined, the final start date will be the greater of num_years before the end date or the
       first dateitme of the PlanetOS dataset

    Args:
        dataset (:obj:`string`): Dataset name ("merra2" or "era5")
        lat (:obj:`float`): Latitude in WGS 84 spatial reference system (degrees)
        lon (:obj:`float`): Longitude in WGS 84 spatial reference system (degrees)
        start_date (:obj:`pandas.Timestamp` or :obj:`string`, optional): Desired start datetime of reanalysis data time
            series. Defaults to None.
        end_date (:obj:`pandas.Timestamp` or :obj:`string`, optional): Desired end datetime of reanalysis data time
            series. Defaults to None.
        num_years (:obj:`int`, optional): Desired number of years of reanalysis data. Only used if either
            start_date or end_date are undefined. Defaults to 20.
        var_names (:obj:`list`, optional): List of desired reanalysis variable names from PlanetOS data
            set. If undefined, default variables will be downloaded (U and V wind speeds, temperature, and surface air
            pressure) and variable names will be converted to standard OpenOA variable names. Defaults to None.
        var_dict (:obj:`dict`, optional): Optional dictionary mapping the desired reanalysis variable names from
            PlanetOS to custom variable names which will be used in the datarame that is returned. Note that if the
            argument var_names is undefined, the downloaded default variables will be converted to standard OpenOA
            variable names. Defaults to None.
        calc_derived_vars (:obj:`bool`, optional): Boolean that specifies whether wind speed, wind direction, and air
            density are computed from the downloaded reanalysis variables. Note that this requires the downloaded
            eastward wind speed, northward wind speed, temperature, and surface pressure to be renamed to the standard
            variable names "u_ms", "v_ms", "temperature_K", and "surf_pres_Pa". Defaults to False.
        save_pathname (:obj:`string`, optional): The path where the downloaded reanalysis data will be saved (if
            defined). Defaults to None.
        save_filename (:obj:`string`, optional): The file name used to save the downloaded reanalysis data (if
            defined). Defaults to None.
        apikey_file (:obj:`string`, optional): The combined path and file name where the PlanetOS API key is saved. If
            undefined, will assume there is a file called "APIKEY" saved in the operational_analysis.toolkits
            directory. Defaults to None.
        **api_kwargs: Optional additional keyword arguments used in the PlanetOS API request.

    Returns:
        :obj:`pandas.DataFrame`: A dataframe containing time series of the requested reanalysis variables
    """

    # Import PlanetOS apikey
    if apikey_file is None:
        apikey_file = Path(__file__).parent.resolve() / "APIKEY"
    apikey = open(apikey_file).read().strip()

    # Get start and end dates of dataset
    start_date_ds, end_date_ds = get_dataset_start_end_dates_planetos(dataset, apikey)

    # Depending on the start date, end date, and number of years specified, calculate the appropriate start and end
    # dates
    start_date, end_date = _get_start_end_dates_planetos(
        start_date, end_date, num_years, start_date_ds, end_date_ds
    )

    # Determine number of data points
    date_diff = end_date - start_date
    count = int(24 * date_diff.days + date_diff.seconds / 3600)

    # Get default variable names and dictionary if not defined
    if var_names is None:
        var_dict = _get_default_var_dicts_planetos(dataset)
        var_names = list(var_dict.keys())

    # Download data from PlanetOS
    url = "http://api.planetos.com/v1/datasets/" + _get_dataset_names(dataset) + "/point?"

    base_kwargs = {
        "apikey": apikey,
        "count": count,
        "lon": lon,
        "lat": lat,
        "vars": ", ".join(var_names),
        "time_start": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "time_end": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    r = requests.get(url, params={**base_kwargs, **api_kwargs})

    # convert to standard dataframe
    df = _convert_resp_to_df(r, var_names, var_dict)

    # compute derived variables if requested
    if calc_derived_vars:
        df["windspeed_ms"] = np.sqrt(df["u_ms"] ** 2 + df["v_ms"] ** 2)
        df["winddirection_deg"] = met.compute_wind_direction(df["u_ms"], df["v_ms"])
        df["rho_kgm-3"] = met.compute_air_density(df["temperature_K"], df["surf_pres_Pa"])

    if save_filename is not None:
        df.to_csv(Path(save_pathname).resolve() / f"{save_filename}.csv", index=False)

    return df
