"""
This module provides functions for downloading files, including reanalysis data

This module provides functions for downloading data, including long-term historical atmospheric
data from the MERRA2 and ERA5 reanalysis products and returning as pandas DataFrames and saving
data in csv files. Currently by default the module downloads monthly reanalysis data for a time
period of interest using NASA Goddard Earth Sciences Data and Information Services Center
(GES DISC) for MERRA2 and the Copernicus Climate Data Store (CDS) API for ERA5, but this can be
modified to get hourly data, and indeed other data sources available on GES DISC and CDS.

To use this module to download data users must first create user accounts. Instructions can be
found at https://disc.gsfc.nasa.gov/data-access#python-requests and
https://cds.climate.copernicus.eu/api-how-to

In addition you can download data directly from these source:

* Hourly MERRA2 data can be downloaded directly from NASA GES DISC by selecting the
  "Subset / Get Data" link on the following webpage:
  https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary. Specific dates, variables, and
  coordinates can be selected using the OPeNDAP or GES DISC Subsetter download methods.

* Hourly ERA5 data can be downloaded using either the CDS web interface or the CDS API, as
  explained here: https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5. Data for specific
  dates, variables, and coordinates can be downloaded using the CDS web interface via the "Download
  data" tab here:
  https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview.
  Instructions for using the CDS Toolbox API to download ERA5 data programatically is found here:
  https://cds.climate.copernicus.eu/toolbox/doc/how-to/1_how_to_retrieve_data/1_how_to_retrieve_data.html
  (note that the "reanalysis-era5-single-levels" dataset should generally be used).
"""

from __future__ import annotations

import re
import hashlib
import datetime
from pathlib import Path
from zipfile import ZipFile

import cdsapi
import pandas as pd
import xarray as xr
import requests
from tqdm import tqdm

from openoa.logging import logging


logger = logging.getLogger()


BYTES_MB = 1024 * 104


def download_file(url: str, outfile: str | Path) -> None:
    """
    Download a file from the web, based on its url, and save to the outfile.

    Args:
        url(:obj:`str`): Url of data to download.
        outfile(:obj:`str` | :obj:`Path`): File path to which the download is saved.

    Raises:
        HTTPError: If unable to access url.
        Exception: If the request failed for another reason.
    """

    outfile = Path(outfile).resolve()
    result = requests.get(url, stream=True)

    try:
        result.raise_for_status()
        try:
            with outfile.open("wb") as f:
                for chunk in tqdm(result.iter_content(chunk_size=BYTES_MB), desc="MB downloaded"):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Contents of {url} written to {outfile}")

        except Exception as e:
            logger.error(e)
            raise

    except requests.exceptions.HTTPError as eh:
        logger.error(eh)
        raise

    except Exception as e:
        logger.error(e)
        raise


def download_zenodo_data(record_id: int, outfile_path: str | Path) -> None:
    """
    Download data from Zenodo based on the Zenodo record_id.

    The following files will be saved to the asset data folder:

        1. record_details.json, which details the Zenodo api details.
        2. all files available for the record_id.

    Args:
        record_id(:obj:`int`): The Zenodo record id.
        outfile_path(:obj:`str` | :obj:`Path`): Path to save files to.

    """

    url_zenodo = r"https://zenodo.org/api/records/"
    r = requests.get(f"{url_zenodo}{record_id}")

    r_json = r.json()

    logger.info("======")
    logger.info("Title: " + r_json["metadata"]["title"])
    logger.info("Version: " + r_json["metadata"]["version"])
    logger.info("URL: " + r_json["links"]["latest_html"])
    logger.info("Record DOI: " + r_json["doi"])
    logger.info("License: " + r_json["metadata"]["license"]["id"])
    logger.info("======\n")

    # create outfile_path if it does not exist
    outfile_path = Path(outfile_path).resolve()
    if not outfile_path.exists():
        outfile_path.mkdir()

    # save record details to json file
    outfile = outfile_path / "record_details.json"

    with outfile.open("wb") as f:
        f.write(r.content)

    # download all files
    files = r_json["files"]
    for f in files:
        url_file = f["links"]["self"]

        outfile = outfile_path / (file_name := f["key"])

        # check if file exists
        if outfile.exists():
            # if it does check the checksum is correct
            with outfile.open("rb") as f_check:
                file_hash = hashlib.md5()
                while chunk := f_check.read(8192):
                    file_hash.update(chunk)

            if f["checksum"][4:] == file_hash.hexdigest():
                logger.info(f"File already exists: {file_name}")

            # download and unzip if the checksum isn't correct
            else:
                logger.info(f"Downloading: {file_name}")
                logger.info(f"File size: {f['size']/(BYTES_MB):,.2f} MB")

                download_file(url_file, outfile)

                logger.info(f"Saved to: {outfile}\n")

                if outfile.suffix == ".zip":
                    with ZipFile(outfile) as zipfile:
                        zipfile.extractall(outfile_path)

        # download and unzip if the file doesn't exist
        else:
            logger.info(f"\nDownloading: {file_name}")
            logger.info(f"File size: {f['size']/(BYTES_MB):,.2f} MB")

            download_file(url_file, outfile)

            logger.info(f"Saved to: {outfile}\n")

            if outfile.suffix == ".zip":
                with ZipFile(outfile) as zipfile:
                    zipfile.extractall(outfile_path)


def get_era5_monthly(
    lat: float,
    lon: float,
    save_pathname: str | Path,
    save_filename: str,
    start_date: str = "2000-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
    Get ERA5 data directly from the CDS service. This requires registration on the CDS service.
    See registration details at: https://cds.climate.copernicus.eu/api-how-to

    This function returns monthly ERA5 data from the "ERA5 monthly averaged data on single levels
    from 1959 to present" dataset. See further details regarding the dataset at:
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means.
    Only 10m wind speed, the temperature at 2m, and the surface pressure are downloaded here.

    As well as returning the data as a dataframe, the data is also saved as monthly NetCDF files and
    a csv file with the concatenated data. These are located in the "save_pathname" directory, with
    "save_filename" prefix. This allows future loading without download from the CDS service.

    Args:
        lat(:obj:`float`): Latitude in WGS 84 spatial reference system (decimal degrees).
        lon(:obj:`float`): Longitude in WGS 84 spatial reference system (decimal degrees).
        save_pathname(:obj:`str` | :obj:`Path`): The path where the downloaded reanalysis data will
            be saved.
        save_filename(:obj:`str`): The file name used to save the downloaded reanalysis data.
        start_date(:obj:`str`): The starting year and month that data is downloaded for. This
            should be provided as a string in the format "YYYY-MM". Defaults to "2000-01".
        end_date(:obj:`str`): The final year and month that data is downloaded for. This should be
            provided as a string in the format "YYYY-MM". Defaults to current year and most recent
            month with full data, accounting for the fact that the ERA5 monthly dataset is released
            around the the 6th of the month.

    Returns:
        df(:obj:`dataframe`): A dataframe containing time series of the requested reanalysis
            variables:
            1. windspeed_ms: the wind speed in m/s at 10m height.
            2. temperature_K: air temperature in Kelvin at 2m height.
            3. surf_pres_Pa: surface pressure in Pascals.

    Raises:
        ValueError: If the start_date is greater than the end_date.
        Exception: If unable to connect to the cdsapi client.
    """

    logger.info("Please note access to ERA5 data requires registration")
    logger.info("Please see: https://cds.climate.copernicus.eu/api-how-to")

    # set up cds-api client
    try:
        c = cdsapi.Client()
    except Exception as e:
        logger.error("Failed to make connection to cds")
        logger.error("Please see https://cds.climate.copernicus.eu/api-how-to for help")
        logger.error(e)
        raise

    # create save_pathname if it does not exist
    save_pathname = Path(save_pathname).resolve()
    if not save_pathname.exists():
        save_pathname.mkdir()

    # get the current date minus 37 days to find the most recent full month of data
    now = datetime.datetime.now() - datetime.timedelta(days=37)

    # assign end_year to current year if not provided by the user
    if end_date is None:
        end_date = f"{now.year}-{now.month:02}"

    # convert dates to datetime objects
    start_date = datetime.datetime.strptime(start_date, "%Y-%m")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m")

    # check that the start and end dates are the right way around
    if start_date > end_date:
        logger.error("The start_date should be less than or equal to the end_date")
        logger.error(f"start_date = {start_date.date()}, end_date = {end_date.date()}")
        raise ValueError("The start_date should be less than or equal to the end_date")

    # list all dates that will be downloaded
    dates = pd.date_range(start=start_date, end=end_date, freq="MS", inclusive="both")

    # get the data for the closest 9 nodes to the coordinates
    node_spacing = 0.250500001 * 1

    # See: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form
    # for formulating other requests from cds
    cds_dataset = "reanalysis-era5-single-levels-monthly-means"
    cds_request = {
        "product_type": "monthly_averaged_reanalysis",
        "format": "netcdf",
        "variable": [
            "10m_wind_speed",
            "2m_temperature",
            "surface_pressure",
        ],
        "year": None,
        "month": None,
        "time": ["00:00"],
        "area": [
            lat + node_spacing,
            lon - node_spacing,
            lat - node_spacing,
            lon + node_spacing,
        ],
    }

    # download the data
    for date in dates:
        outfile = save_pathname / f"{save_filename}_{date.year}{date.month:02}.nc"

        if not outfile.is_file():
            logger.info(f"Downloading ERA5: {outfile}")

            try:
                cds_request.update({"year": date.year, "month": date.month})
                c.retrieve(cds_dataset, cds_request, outfile)

            except Exception as e:
                logger.error(f"Failed to download ERA5: {outfile}")
                logger.error(e)

    # get the saved data
    ds_nc = xr.open_mfdataset(f"{save_pathname / f'{save_filename}*.nc'}")

    # rename variables to conform with OpenOA
    ds_nc = ds_nc.rename_vars(
        {"si10": "windspeed_ms", "t2m": "temperature_K", "sp": "surf_pres_Pa"}
    )

    # select the central node only for now
    if "expver" in ds_nc.dims:
        sel = ds_nc.sel(expver=1, latitude=lat, longitude=lon, method="nearest")
    else:
        sel = ds_nc.sel(latitude=lat, longitude=lon, method="nearest")

    # convert to a pandas dataframe
    df = sel.to_dataframe()

    # select required columns
    df = df[["windspeed_ms", "temperature_K", "surf_pres_Pa"]]

    # rename the index to match other datasets
    df.index.name = "datetime"

    # drop any empty rows
    df = df.dropna()

    # crop time series to only the selected time period
    df = df.loc[start_date:end_date]

    # save to csv for easy loading as required
    df.to_csv(save_pathname / f"{save_filename}.csv", index=True)

    return df


def get_merra2_monthly(
    lat: float,
    lon: float,
    save_pathname: str | Path,
    save_filename: str,
    start_date: str = "2000-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
    Get MERRA2 data directly from the NASA GES DISC service, which requires registration on the
    GES DISC service. See: https://disc.gsfc.nasa.gov/data-access#python-requests.

    This function returns monthly MERRA2 data from the "M2IMNXLFO" dataset. See further details
    regarding the dataset at: https://disc.gsfc.nasa.gov/datasets/M2IMNXLFO_5.12.4/summary.
    Only surface wind speed, temperature and surface pressure are downloaded here.

    As well as returning the data as a dataframe, the data is also saved as monthly NetCDF files
    and a csv file with the concatenated data. These are located in the "save_pathname" directory,
    with "save_filename" prefix. This allows future loading without download from the CDS service.

    Args:
        lat(:obj:`float`): Latitude in WGS 84 spatial reference system (decimal degrees).
        lon(:obj:`float`): Longitude in WGS 84 spatial reference system (decimal degrees).
        save_pathname(:obj:`str` | :obj:`Path`): The path where the downloaded reanalysis data will
            be saved.
        save_filename(:obj:`str`): The file name used to save the downloaded reanalysis data.
        start_date(:obj:`str`): The starting year and month that data is downloaded for. This
            should be provided as a string in the format "YYYY-MM". Defaults to "2000-01".
        end_date(:obj:`str`): The final year and month that data is downloaded for. This should be
            provided as a string in the format "YYYY-MM". Defaults to current year and most recent
            month.

    Returns:
        df(:obj:`dataframe`): A dataframe containing time series of the requested reanalysis
            variables:
            1. windspeed_ms: the surface wind speed in m/s.
            2. temperature_K: surface air temperature in Kelvin.
            3. surf_pres_Pa: surface pressure in Pascals.

    Raises:
        ValueError: If the start_year is greater than the end_year.
    """

    logger.info("Please note access to MERRA2 data requires registration")
    logger.info("Please see: https://disc.gsfc.nasa.gov/data-access#python-requests")

    # base url containing the monthly data set M2IMNXLFO
    base_url = r"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2_MONTHLY/M2IMNXLFO.5.12.4/"

    # create save_pathname if it does not exist
    save_pathname = Path(save_pathname).resolve()
    if not save_pathname.exists():
        save_pathname.mkdir()

    # get the current date minus 37 days to find the most recent full month of data
    now = datetime.datetime.now() - datetime.timedelta(days=37)

    # assign end_year to current year if not provided by the user
    if end_date is None:
        end_date = f"{now.year}-{now.month:02}"

    # convert dates to datetime objects
    start_date = datetime.datetime.strptime(start_date, "%Y-%m")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m")

    # check that the start and end dates are the right way around
    if start_date > end_date:
        logger.error("The start_date should be less than or equal to the end_date")
        logger.error(f"start_date = {start_date.date()}, end_date = {end_date.date()}")
        raise ValueError("The start_date should be less than or equal to the end_date")

    # list all years that will be downloaded
    years = list(range(start_date.year, end_date.year + 1, 1))

    # download the data
    for year in years:
        # get the file names from the GES DISC site for the year
        result = requests.get(f"{base_url}{year}")

        files = re.findall(r"(>MERRA2_\S+.nc4)", result.text)
        files = list(dict.fromkeys(files))
        files = [x[1:] for x in files]

        # coordinate indexes
        lat_i = ""
        lon_i = ""

        # download each of the files and save them
        for f in files:
            outfile = save_pathname / f"{save_filename}_{f.split('.')[-2]}.nc"

            if not outfile.is_file():
                # download one file for determining coordinate indicies
                if lat_i == "":
                    url = f"{base_url}{year}/{f}" + r".nc4?PS,SPEEDLML,TLML,time,lat,lon"
                    download_file(url, outfile)
                    ds_nc = xr.open_dataset(outfile)
                    ds_nc_idx = ds_nc.assign_coords(
                        lon_idx=("lon", range(ds_nc.dims["lon"])),
                        lat_idx=("lat", range(ds_nc.dims["lat"])),
                    )
                    sel = ds_nc_idx.sel(lat=lat, lon=lon, method="nearest")
                    lon_i = f"[{sel.lon_idx.values-1}:{sel.lon_idx.values+1}]"
                    lat_i = f"[{sel.lat_idx.values-1}:{sel.lat_idx.values+1}]"
                    ds_nc.close()
                    outfile.unlink()

                # download file with specified coordinates
                url = (
                    f"{base_url}{year}/{f}"
                    r".nc4?PS[0:0]"
                    f"{lat_i}{lon_i}"
                    f",SPEEDLML[0:0]{lat_i}{lon_i}"
                    f",TLML[0:0]{lat_i}{lon_i}"
                    f",time,lat{lat_i},lon{lon_i}"
                )

                download_file(url, outfile)

    # get the saved data
    ds_nc = xr.open_mfdataset(f"{save_pathname / f'{save_filename}*.nc'}")

    # rename variables to conform with OpenOA
    ds_nc = ds_nc.rename_vars(
        {"SPEEDLML": "windspeed_ms", "TLML": "temperature_K", "PS": "surf_pres_Pa"}
    )

    # select the central node only for now
    sel = ds_nc.sel(lat=lat, lon=lon, method="nearest")

    # convert to a pandas dataframe
    df = sel.to_dataframe()

    # select required columns
    df = df[["windspeed_ms", "temperature_K", "surf_pres_Pa"]]

    # rename the index to match other datasets
    df.index.name = "datetime"

    # drop any empty rows
    df = df.dropna()

    # crop time series to only the selected time period
    df = df.loc[start_date:end_date]

    # save to csv for easy loading as required
    df.to_csv(save_pathname / f"{save_filename}.csv", index=True)

    return df
