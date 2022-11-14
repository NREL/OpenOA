#################################################
# Data import script for Cubico Projects #
#################################################
"""
This is the import script for Cubico's Kelmarsh & Penmanshiel projects. Below
is a description of data quality for each data frame and an overview of the
steps taken to correct the raw data for use in the PRUF OA code.

1. SCADA dataframe
- 10-minute SCADA data for each of the four turbines in the project
- Power, wind speed, wind direction, nacelle position, wind vane, temperature,
  blade pitch

2. Meter data frame
- 10-minute performance data provided in energy units (kWh)

3. Curtailment data frame
- 10-minute availability and curtailment data in kwh

4. Reanalysis products
- Import MERRA-2 and ERA5 1-hour reanalysis data
- Import ERA-5 monthly reanalysis data at 10m height for wind
- Wind speed, wind direction, temperature, and density
"""

from __future__ import annotations
from distutils.log import error

import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import openoa.utils.unit_conversion as un
import openoa.utils.met_data_processing as met
from openoa.logging import logging
from openoa.plant import PlantData
from openoa.utils import filters, timeseries


logger = logging.getLogger()


import requests
import os
import hashlib

import json
import yaml

import cdsapi
import xarray as xr
import datetime


def download_file(url,outfile):
    """
    Download a file from the web based on its url

    Args:
        url (str): url of data to download
        outfile: file path to which the download is saved

    Returns:
        Downloaded file saved to outfile
    """
    
    result = requests.get(url,stream=True)
    
    try:
        result.raise_for_status()
        
        chunk_number = 0
        
        try:
            with open(outfile, "wb") as f:
                
                for chunk in result.iter_content(chunk_size=1024*1024):
                    
                    chunk_number = chunk_number + 1
                    
                    print(str(chunk_number) + " MB downloaded", end="\r")
                    
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        
            logger.info('Contents of '+url+' written to '+outfile)
            
        except:
            logger.error('Error writing to '+outfile)
            
           
    except:
        logger.error('Requests.get() returned an error code '+str(result.status_code))
        logger.error(url)


def download_zenodo_data(record_id,outfile_path):
    """
    Download data from zenodo based on the zenodo record_id
    
    Args:
        record_id (int): the Zenodo record id
        outfile_path: path to save files to

    Returns:
        Files saved to the asset data folder:
         1. record_details.json, which details the zenodo api details
         2. all files available for the record_id
    """
    
    
    url_zenodo = r"https://zenodo.org/api/records/"

    record_id = str(record_id)
    
    r = requests.get(url_zenodo + record_id)
    
    r_json = r.json()
    
    
    logger.info("======")
    logger.info("Title: " + r_json["metadata"]["title"])
    logger.info("Version: " + r_json["metadata"]["version"])
    logger.info("URL: " + r_json["links"]["latest_html"])
    logger.info("Record DOI: " + r_json["doi"])
    logger.info("License: " + r_json["metadata"]["license"]["id"])
    logger.info("======\n")
    
       
    # create outfile_path if it does not exist
    if not os.path.exists(outfile_path):
        os.makedirs(outfile_path)
    
    
    # save record details to json file
    outfile = outfile_path.joinpath("record_details.json")
    
    with open(outfile, "wb") as f:
        f.write(r.content)

        
    # download all files
    files = r_json["files"]
    for f in files:
        
        url_file = f["links"]["self"]
        
        file_name = f["key"]
                
        outfile = outfile_path.joinpath(file_name)
        
        
        # check if file exists
        if os.path.exists(outfile):
            
            
            # if it does check the checksum is correct
            with open(outfile, "rb") as f_check:
                file_hash = hashlib.md5()
                while chunk := f_check.read(8192):
                    file_hash.update(chunk)
        
            if f["checksum"][4:]==file_hash.hexdigest():
                logger.info("File already exists: " + file_name)
            
            
            # download and unzip if the checksum isn't correct
            else:
                
                logger.info("Downloading: " + file_name)
                logger.info("File size: " + str(round(f["size"]/(1024*1024),2)) + "MB")       

                download_file(url_file,outfile)

                logger.info("Saved to: " + str(outfile) + "\n")

                if outfile.endswith(".zip"):
                    with ZipFile(outfile) as zipfile:
                        zipfile.extractall(outfile_path)
        
        
        # download and unzip if the file doesn't exist
        else:
            
            logger.info("\nDownloading: " + file_name)
            logger.info("File size: " + str(round(f["size"]/(1024*1024),2)) + "MB")       

            download_file(url_file,outfile)

            logger.info("Saved to: " + str(outfile) + "\n")

            if outfile.endswith(".zip"):
                with ZipFile(outfile) as zipfile:
                    zipfile.extractall(outfile_path)


def download_asset_data(asset="kelmarsh",outfile_path="data/kelmarsh/"):
    """
    Simplify downloading of know open data assets from zenodo
    
    The record_id will need updating as new data versions come out
    but does mean we have control to avoid any backwards compatibility issues

    Args:
        asset (str): name of asset
        outfile_path: path to save files to

    Returns:
        Files saved to the outfile_path:
         1. record_details.json, which details the zenodo api details
         2. all files available for the record_id
    """
    
    if asset.lower() == "kelmarsh":
        record_id = 7212475
    elif asset.lower() == "penmanshiel":
        record_id = 5946808
    else:
        raise NameError("Zenodo record id undefined for: " + asset)
        
    download_zenodo_data(record_id,outfile_path)


def extract_all_data(path="data/kelmarsh/"):
    """
    Get all zip files in path and extract them

    Args:
        path (Path): path to zip files

    Returns:
        All zip files extracted into the path
    """

    logger.info("Extracting compressed data files")
    
    zipFiles = Path(path).rglob("*.zip")
    
    for file in zipFiles:
        with ZipFile(file) as zipfile:
            zipfile.extractall(path)


def get_scada_headers(SCADA_files):
    """
    Get just the headers from the SCADA files

    Args:
        SCADA_files (list of files): list of SCADA files

    Returns:
        SCADA_headers (pandas dataframe): containing details of all SCADA files
    """
    
    csv_params = {"index_col":0,"skiprows":2, "nrows":4, "delimiter":": ","header":None, "engine":"python"}

    SCADA_headers = pd.concat((pd.read_csv(f,**csv_params).rename(columns={1:f}) for f in SCADA_files),axis=1)

    SCADA_headers.index = SCADA_headers.index.str.replace("# ","")

    SCADA_headers = SCADA_headers.transpose()

    SCADA_headers = SCADA_headers.reset_index().rename(columns={"index":"File"})
    
    return SCADA_headers


def get_scada_df(SCADA_headers,usecolumns=None):
    """
    Extract the desired SCADA data
    
    Args:
        SCADA_headers (pandas dataframe): containing details of all SCADA files
        usecolumns: selection of columns to be imported from the SCADA files

    Returns:
        SCADA (pandas dataframe): dataframe with SCADA data
    """
    
    if usecolumns is None:
        usecolumns = ["# Date and time", 
            "Power (kW)", 
            "Wind speed (m/s)",
            "Wind direction (°)",
            "Nacelle position (°)",
            "Nacelle ambient temperature (°C)",
            "Blade angle (pitch position) A (°)"]

    csv_params = {"index_col":"# Date and time",
        "parse_dates":True,
        "skiprows":9,
        "usecols":usecolumns}

    SCADA_lst = list()
    for turbine in SCADA_headers["Turbine"].unique():
        SCADA_wt = pd.concat((pd.read_csv(f,**csv_params) for f in list(SCADA_headers.loc[SCADA_headers["Turbine"] == turbine]["File"])))
        SCADA_wt["Turbine"] = turbine
        SCADA_wt.index.names = ["Timestamp"]
        SCADA_lst.append(SCADA_wt.copy())

    SCADA = pd.concat(SCADA_lst)
    
    return SCADA


def get_curtailment_df(SCADA_headers):
    """
    Get the curtailment and availability data
    
    Args:
        SCADA_headers (pandas dataframe): containing details of all SCADA files

    Returns:
        curtailment_df (pandas dataframe): dataframe with curtailment data
    """
    
    # Curtailment data is available as a subset of the SCADA data
    usecolumns = ["# Date and time", 
        "Lost Production to Curtailment (Total) (kWh)", 
        "Lost Production to Downtime (kWh)", 
        "Energy Export (kWh)"]

    curtailment_df = get_scada_df(SCADA_headers,usecolumns)
    
    return curtailment_df


def get_meter_data(path="data/kelmarsh/"):
    """
    Get the PMU meter data

    Args:
        path (Path): path to meter data

    Returns:
        meter_df (pandas dataframe): dataframe with meter data
    """

    usecolumns = ["# Date and time","GMS Energy Export (kWh)"]

    csv_params = {"index_col":"# Date and time","parse_dates":True,"skiprows":10,"usecols":usecolumns}

    meter_files = list(Path(path).rglob("*PMU*.csv"))
    
    meter_df = pd.read_csv(meter_files[0],**csv_params)
    
    meter_df.index.names = ["Timestamp"]
    
    return meter_df


def get_era5(asset="penmanshiel",lat=55.864,lon=-2.352):
    """
    Get ERA5 data directly from the CDS service
    This requires registration on the CDS service
    See: https://cds.climate.copernicus.eu/api-how-to

    Monthly 10m height data is demonstrated here,
    as hourly data takes too long to download, but could be amended
    and other CDS datasets also used (e.g. CERRA for Europe)

    Args:
        asset (str): name of the asset.
        lat (float): latitude of the asset as decimal degrees
        lon (float): longitude of the asset as decimal degrees

    Returns:
        NetCDF annual ERA5 files saved to the asset data folder
        ERA5 csv file saved to the asset data folder
    """

    # set up cds-api client
    try:
        c = cdsapi.Client()
    except Exception as e:
        logger.error('Failed to make connection to cds: '+ str(e))
        logger.error('Please See: https://cds.climate.copernicus.eu/api-how-to')
        raise NameError(e)

    # the data is stored with the asset data
    outfile_path = r"data//"+asset+"/era5_monthly_10m//"

    # create outfile_path if it does not exist
    if not os.path.exists(outfile_path):
        os.makedirs(outfile_path)


    now = datetime.datetime.now()
    years = list(range(2000,now.year+1,1))

    # get the data for the closest 3 nodes to the coordinates
    node_spacing = 0.250500001*1

    # download the data
    for year in years:

        outfile = outfile_path+asset+"_ERA5_monthly_"+str(year)+".nc"

        if year == now.year:
            months = list(range(1,now.month,1))
        else:
            months = list(range(1,12+1,1))

        # See: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form
        # for formulating other requests from cds
        if not os.path.exists(outfile) or year==now.year:

            logger.info("Downloading ERA5 :" + outfile)

            c.retrieve(
                "reanalysis-era5-single-levels-monthly-means",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": [
                        "10m_wind_speed", "2m_temperature", "surface_pressure",
                    ],
                    "year": year,
                    "month": months,
                    "product_type": "monthly_averaged_reanalysis",
                    "time": [
                        "00:00"
                    ],
                    "area": [
                        lat+node_spacing, lon-node_spacing, 
                        lat-node_spacing, lon+node_spacing,
                    ],
                },
                outfile)
    
    # get the saved data
    ds_nc = xr.open_mfdataset(outfile_path+asset+"_ERA5_monthly_"+"*.nc")

    # renamce variables to conform with OpenOA
    ds_nc = ds_nc.rename_vars({"si10":"windspeed_ms","t2m":"temperature_K","sp":"surf_pres_Pa"})
    
    # select the central node only for now
    if 'expver' in ds_nc.dims:
        sel = ds_nc.sel(expver=1,latitude=lat,longitude=lon, method="nearest")
    else:
        sel = ds_nc.sel(latitude=lat,longitude=lon, method="nearest")   

    # convert to a pandas dataframe
    df = sel.to_dataframe()

    # select required columns
    df = df[["windspeed_ms","temperature_K","surf_pres_Pa"]]

    # rename the index to match other datasets
    df.index.name = "datetime"

    # drop any empty rows
    df = df.dropna()
    
    # export to csv for easy loading next time
    df.to_csv("data//"+asset+"//"+asset+"_era5_monthly_10m.csv")


def get_merra2(asset="penmanshiel",lat=55.864,lon=-2.352):
    """
    Get MERRA2 data directly from the NASA GES DISC service
    This requires registration on the GES DISC service
    See: https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Generate%20Earthdata%20Prerequisite%20Files

    Monthly 10m height data is demonstrated here,
    as hourly data takes too long to download, but could be amended
    and other GES DISC datasets also used (e.g. FLDAS)

    Args:
        asset (str): name of the asset.
        lat (float): latitude of the asset as decimal degrees
        lon (float): longitude of the asset as decimal degrees

    Returns:
        NetCDF monthly MERRA-2 files saved to the data folder
        MERRA-2 csv file saved to the asset data folder
    """
    
    base_url = r"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2_MONTHLY/M2IMNXLFO.5.12.4/"

    # the merra2 asset data is stored with the asset data
    outfile_path = r"data//"+asset+"//MERRA2_monthly_10m//"
    
    # create outfile_path if it does not exist
    if not os.path.exists(outfile_path):
        os.makedirs(outfile_path)


    now = datetime.datetime.now()
    years = list(range(2000,now.year+1,1))

    
    
    # download the data
    for year in years:
        
        # get the file names from the GES DISC site for the year
        result = requests.get(base_url+str(year))
        
        files = re.findall(r"(>MERRA2_\S+.nc4)", result.text)
        files = list(dict.fromkeys(files))
        files = [x[1:] for x in files]
        
        
        # coordinate indexes
        lat_i = ""
        lon_i = ""
        
        # download each of the files and save them
        for file in files:
            
            outfile = outfile_path+"/MERRA2_monthly_"+file.split(".")[-2]+".nc"

            if not os.path.isfile(outfile):

                # download one file for determining coordinate indicies
                if lat_i=="":
                    url = base_url+str(year)+"//"+file+r".nc4?PS,SPEEDLML,TLML,time,lat,lon"
                    download_file(url,outfile)
                    ds_nc = xr.open_dataset(outfile)
                    ds_nc_idx = ds_nc.assign_coords(lon_idx=("lon",range(ds_nc.dims['lon'])),lat_idx=("lat",range(ds_nc.dims['lat'])))
                    sel = ds_nc_idx.sel(lat=lat,lon=lon, method="nearest")
                    lon_i = "["+str(sel.lon_idx.values-1)+":"+str(sel.lon_idx.values+1)+"]"
                    lat_i = "["+str(sel.lat_idx.values-1)+":"+str(sel.lat_idx.values+1)+"]"
                    ds_nc.close()
                    os.remove(outfile) 
                    
                    
                url = base_url+str(year)+"//"+file+r".nc4?PS[0:0]"+lat_i+lon_i+",SPEEDLML[0:0]"+lat_i+lon_i+",TLML[0:0]"+lat_i+lon_i+",time,lat"+lat_i+",lon"+lon_i
                
                download_file(url,outfile)
                

                            
                    
    # get the saved data
    ds_nc = xr.open_mfdataset(outfile_path+"MERRA2_monthly_"+"*.nc")

    # renamce variables to conform with OpenOA
    ds_nc = ds_nc.rename_vars({"SPEEDLML":"windspeed_ms","TLML":"temperature_K","PS":"surf_pres_Pa"})
    
    # select the central node only for now
    sel = ds_nc.sel(lat=lat,lon=lon, method="nearest")   

    # convert to a pandas dataframe
    df = sel.to_dataframe()

    # select required columns
    df = df[["windspeed_ms","temperature_K","surf_pres_Pa"]]

    # rename the index to match other datasets
    df.index.name = "datetime"

    # drop any empty rows
    df = df.dropna()
    
    # export to csv for easy loading next time
    df.to_csv("data//"+asset+"//"+asset+"_MERRA2_monthly_10m.csv")
    
                    
                    
def prepare(asset="kelmarsh", return_value="plantdata"):
    """
    Do all loading and preparation of the data for this plant.

    Args:
    - asset (pandas.DataFrame): asset name, currently either kelmarsh or penmanshiel
    - return_value (str): "plantdata" will return a fully constructed PlantData object. "dataframes" will return a list of dataframes instead.

    Returns:
        Either PlantData object or Dataframes dependent upon return_value
    """

    # Set the path to store and access all the data
    path = Path("data//"+asset)

    # Download and extract data if necessary
    download_asset_data(asset=asset,outfile_path=path)


    ##############
    # ASSET DATA #
    ##############

    logger.info("Reading in the asset data")
    asset_df = pd.read_csv(path/(asset+"_WT_static.csv"))

    # Assign type to turbine for all assets
    asset_df["type"] = "turbine"


    ###################
    # SCADA DATA #
    ###################
    logger.info("Reading in the SCADA data")
    SCADA_files = Path(path).rglob("Turbine_Data*.csv")
    SCADA_headers = get_scada_headers(SCADA_files)
    scada_df = get_scada_df(SCADA_headers)
    scada_df = scada_df.reset_index()


    ##############
    # METER DATA #
    ##############
    logger.info("Reading in the meter data")
    meter_df = get_meter_data(path)
    meter_df = meter_df.reset_index()

    
    #####################################
    # Availability and Curtailment Data #
    #####################################
    logger.info("Reading in the curtailment data")
    curtail_df = get_curtailment_df(SCADA_headers)  # Load Availability and Curtail data
    curtail_df = curtail_df.reset_index()


    ###################
    # REANALYSIS DATA #
    ###################
    logger.info("Reading in the reanalysis data")

    # reanalysis datasets are held in a dictionary
    reanalysis_dict = dict()

    # MERRA2
    if os.path.exists(path/(asset+"_merra2.csv")):
        logger.info("Reading MERRA2")
        reanalysis_merra2_df = pd.read_csv(path/(asset+"_merra2.csv"))
        reanalysis_dict.update(dict(merra2=reanalysis_merra2_df))

    # ERA5
    if os.path.exists(path/(asset+"_era5.csv")):
        logger.info("Reading ERA5")
        reanalysis_era5_df = pd.read_csv(path/(asset+"_era5.csv"))
        reanalysis_dict.update(dict(era5=reanalysis_era5_df))


    # ERA5 monthly 10m
    logger.info("Loading ERA5 monthly")
    get_era5(asset=asset,lat=asset_df["Latitude"].mean(),lon=asset_df["Longitude"].mean())

    logger.info("Reading ERA5 monthly")
    reanalysis_era5_monthly_df = pd.read_csv(path/(asset+"_era5_monthly_10m.csv"))
    reanalysis_dict.update(dict(erai=reanalysis_era5_monthly_df))


    ###################
    # PLANT DATA #
    ###################

    # Create plant_meta.json
    asset_json = {
        
      "asset": {
        "elevation": "Elevation (m)",
        "hub_height": "Hub Height (m)",
        "id": "Title",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "rated_power": "Rated power (kW)",
        "rotor_diameter": "Rotor Diameter (m)"
      },
        
      "curtail": {
        "availability": "Lost Production to Downtime (kWh)",
        "curtailment": "Lost Production to Curtailment (Total) (kWh)",
        "frequency": "10T",
        "net_energy": "Energy Export (kWh)",
        "time": "Timestamp"
      },
        
      "latitude": str(asset_df["Latitude"].mean()),
      "longitude": str(asset_df["Longitude"].mean()),
      "capacity":str(asset_df["Rated power (kW)"].sum()/1000),
        
      "meter": {
        "energy": "GMS Energy Export (kWh)",
        "time": "Timestamp"
      },
        
      "reanalysis": {
        "era5": {
          "frequency": "H",
          "surface_pressure": "surf_pres_Pa",
          "temperature": "temperature_K",
          "time": "datetime",
          "windspeed_u": "u_ms",
          "windspeed_v": "v_ms",
          "windspeed":"windspeed_ms",
          "wind_direction":"winddirection_deg",
        },
          
        "merra2": {
          "frequency": "H",
          "surface_pressure": "surf_pres_Pa",
          "temperature": "temperature_K",
          "time": "datetime",
          "windspeed_u": "u_ms",
          "windspeed_v": "v_ms",
          "windspeed":"windspeed_ms",
          "wind_direction":"winddirection_deg",
        },

        "erai": {
          "frequency": "1MS",
          "surface_pressure": "surf_pres_Pa",
          "temperature": "temperature_K",
          "time": "datetime",
          "windspeed":"windspeed_ms",
        },
        
      },
        
      "scada": {
        "frequency": "10T",
        "id": "Turbine",
        "pitch": "Blade angle (pitch position) A (°)",
        "power": "Power (kW)",
        "temperature": "Nacelle ambient temperature (°C)",
        "time": "Timestamp",
        "wind_direction": "Wind direction (°)",
        "windspeed": "Wind speed (m/s)"
      }
    }

    with open(path.joinpath("plant_meta.json"), "w") as outfile:
        json.dump(asset_json, outfile, indent=2)
        
    with open(path.joinpath("plant_meta.yml"), "w") as outfile:
        yaml.dump(asset_json, outfile, default_flow_style=False)

    
    # Return the appropriate data format
    if return_value == "dataframes":
        return (
            scada_df,
            meter_df,
            curtail_df,
            asset_df,
            reanalysis_dict,
        )
    elif return_value == "plantdata":
        # Build and return PlantData
        plantdata = PlantData(
            analysis_type="MonteCarloAEP",  # Choosing a random type that doesn't fail validation
            metadata=path / "plant_meta.yml",
            scada=scada_df,
            meter=meter_df,
            curtail=curtail_df,
            asset=asset_df,
            reanalysis=reanalysis_dict,
        )

        return plantdata


if __name__ == "__main__":
    prepare()
