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


def download_file(url: str, outfile: str | Path) -> None:
    """
    Download a file from the web, based on its url.

    Args:
        url(:obj:`str`): Url of data to download.
        outfile(:obj:`str` | :obj:`Path`): File path to which the download is saved.

    Returns:
        Downloaded file saved to outfile.
    """

    outfile = Path(outfile).resolve()
    result = requests.get(url, stream=True)

    try:
        result.raise_for_status()
        try:
            with outfile.open("wb") as f:
                for chunk in tqdm(
                    result.iter_content(chunk_size=1024 * 1024), desc="MB downloaded"
                ):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Contents of {url} written to {outfile}")

        except IOError as ei:
            logger.error(f"Error writing to {outfile}")
            logger.error(ei)

    except requests.exceptions.HTTPError as eh:
        logger.error(eh)
    except requests.exceptions.RequestException as er:
        logger.error(er)


def download_zenodo_data(record_id: int, outfile_path: str | Path) -> None:
    """
    Download data from Zenodo based on the Zenodo record_id.

    Args:
        record_id(:obj:`int`): The Zenodo record id.
        outfile_path(:obj:`str` | :obj:`Path`): Path to save files to.

    Returns:
        Files saved to the asset data folder:

          1. record_details.json, which details the Zenodo api details.
          2. all files available for the record_id.
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
                logger.info(f"File size: {f['size']/(1024*1024):,.2f} MB")

                download_file(url_file, outfile)

                logger.info(f"Saved to: {outfile}\n")

                if outfile.suffix == ".zip":
                    with ZipFile(outfile) as zipfile:
                        zipfile.extractall(outfile_path)

        # download and unzip if the file doesn't exist
        else:
            logger.info(f"\nDownloading: {file_name}")
            logger.info(f"File size: {f['size']/(1024*1024):,.2f} MB")

            download_file(url_file, outfile)

            logger.info(f"Saved to: {outfile}\n")

            if outfile.suffix == ".zip":
                with ZipFile(outfile) as zipfile:
                    zipfile.extractall(outfile_path)


def get_era5(
    lat: float,
    lon: float,
    save_pathname: str | Path,
    save_filename: str,
) -> pd.DataFrame:
    """
    Get ERA5 data directly from the CDS service, which requires registration on the CDS service.
    See: https://cds.climate.copernicus.eu/api-how-to

    Monthly 10m height data is demonstrated here, as hourly data takes too long to download,
    but this could be amended and other CDS datasets also used (e.g. CERRA for Europe).

    Args:
        lat(:obj:`float`): Latitude in WGS 84 spatial reference system (decimal degrees).
        lon(:obj:`float`): Longitude in WGS 84 spatial reference system (decimal degrees).
        save_pathname(:obj:`str` | :obj:`Path`): The path where the downloaded reanalysis data will
            be saved.
        save_filename(:obj:`str`): The file name used to save the downloaded reanalysis data.

    Returns:
        df(:obj:`dataframe`): A dataframe containing time series of the requested reanalysis
            variables.
        Saved NetCDF annual ERA5 files.
        Saved ERA5 csv file.

    Raises:
        NameError: If unable to connect to the cdsapi client.
    """

    logger.info("Please note access to ERA5 data requires registration")
    logger.info("Please see: https://cds.climate.copernicus.eu/api-how-to")

    # set up cds-api client
    try:
        c = cdsapi.Client()
    except Exception as e:
        logger.error(f"Failed to make connection to cds: {e}")
        logger.error("Please see: https://cds.climate.copernicus.eu/api-how-to")
        raise NameError(e)

    # create save_pathname if it does not exist
    save_pathname = Path(save_pathname).resolve()
    if not save_pathname.exists():
        save_pathname.mkdir()

    # downloads all years from today back to the year 2000
    now = datetime.datetime.now()
    years = list(range(2000, now.year + 1, 1))

    # get the data for the closest 9 nodes to the coordinates
    node_spacing = 0.250500001 * 1

    # download the data
    for year in years:
        outfile = save_pathname / f"{save_filename}_{year}.nc"

        if year == now.year:
            months = list(range(1, now.month, 1))
        else:
            months = list(range(1, 12 + 1, 1))

        # See: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form
        # for formulating other requests from cds
        if not outfile.is_file() or year == now.year:
            logger.info(f"Downloading ERA5: {outfile}")

            try:
                c.retrieve(
                    "reanalysis-era5-single-levels-monthly-means",
                    {
                        "product_type": "monthly_averaged_reanalysis",
                        "format": "netcdf",
                        "variable": [
                            "10m_wind_speed",
                            "2m_temperature",
                            "surface_pressure",
                        ],
                        "year": year,
                        "month": months,
                        "time": ["00:00"],
                        "area": [
                            lat + node_spacing,
                            lon - node_spacing,
                            lat - node_spacing,
                            lon + node_spacing,
                        ],
                    },
                    outfile,
                )
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

    if save_filename is not None:
        df.to_csv(save_pathname / f"{save_filename}.csv", index=True)

    return df


def get_merra2(
    lat: float,
    lon: float,
    save_pathname: str | Path,
    save_filename: str,
) -> pd.DataFrame:
    """
    Get MERRA2 data directly from the NASA GES DISC service, which requires registration on the
    GES DISC service. See: https://disc.gsfc.nasa.gov/data-access#python-requests.

    Monthly 10m height data is demonstrated here, as hourly data takes too long to download, but
    could be amended and other GES DISC datasets also used (e.g. FLDAS).

    Args:
        lat(:obj:`float`): Latitude in WGS 84 spatial reference system (decimal degrees).
        lon(:obj:`float`): Longitude in WGS 84 spatial reference system (decimal degrees).
        save_pathname(:obj:`str` | :obj:`Path`): The path where the downloaded reanalysis data will
            be saved.
        save_filename(:obj:`str`): The file name used to save the downloaded reanalysis data.

    Returns:
        df(:obj:`dataframe`): A dataframe containing time series of the requested reanalysis
            variables.
        Saved NetCDF monthly MERRA2 files.
        Saved MERRA2 csv file.
    """

    logger.info("Please note access to MERRA2 data requires registration")
    logger.info("Please see: https://disc.gsfc.nasa.gov/data-access#python-requests")

    # base url containing the monthly data set M2IMNXLFO
    base_url = r"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2_MONTHLY/M2IMNXLFO.5.12.4/"

    # create save_pathname if it does not exist
    save_pathname = Path(save_pathname).resolve()
    if not save_pathname.exists():
        save_pathname.mkdir()

    now = datetime.datetime.now()
    years = list(range(2000, now.year + 1, 1))

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
        for file in files:
            outfile = save_pathname / f"{save_filename}_{file.split('.')[-2]}.nc"

            if not outfile.is_file():
                # download one file for determining coordinate indicies
                if lat_i == "":
                    url = f"{base_url}{year}/{file}" + r".nc4?PS,SPEEDLML,TLML,time,lat,lon"
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
                    f"{base_url}{year}/{file}"
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

    if save_filename is not None:
        df.to_csv(save_pathname / f"{save_filename}.csv", index=True)

    return df
