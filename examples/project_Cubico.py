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
- Import ERA-5 and MERRA-2 monthly reanalysis data at ground level
- Fields cover wind speed, wind direction, temperature, and air density
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

import openoa.utils.downloader as downloader

import os

import json
import yaml

import xarray as xr

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
        
    downloader.download_zenodo_data(record_id,outfile_path)


def extract_all_data(path="data/kelmarsh/"):
    """
    Get all zip files in path and extract them

    Args:
        path (str): path to zip files

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
        path (str): path to meter data

    Returns:
        meter_df (pandas dataframe): dataframe with meter data
    """

    usecolumns = ["# Date and time","GMS Energy Export (kWh)"]

    csv_params = {"index_col":"# Date and time","parse_dates":True,"skiprows":10,"usecols":usecolumns}

    meter_files = list(Path(path).rglob("*PMU*.csv"))
    
    meter_df = pd.read_csv(meter_files[0],**csv_params)
    
    meter_df.index.names = ["Timestamp"]
    
    return meter_df 
                    
                    
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
    path = "data//"+asset+"//"

    # Download and extract data if necessary
    download_asset_data(asset=asset,outfile_path=path)


    ##############
    # ASSET DATA #
    ##############

    logger.info("Reading in the asset data")
    asset_df = pd.read_csv(path+"//"+asset+"_WT_static.csv")

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
    if os.path.exists(path+"//"+asset+"_merra2.csv"):
        logger.info("Reading MERRA2")
        reanalysis_merra2_df = pd.read_csv(path+"//"+asset+"_merra2.csv")
        reanalysis_dict.update(dict(merra2=reanalysis_merra2_df))

    # ERA5
    if os.path.exists(path+"//"+asset+"_era5.csv"):
        logger.info("Reading ERA5")
        reanalysis_era5_df = pd.read_csv(path+"//"+asset+"_era5.csv")
        reanalysis_dict.update(dict(era5=reanalysis_era5_df))

    # ERA5 monthly 10m
    if not os.path.exists(path+"//era5_monthly_10m//"+asset+"_era5_monthly_10m.csv"):
        logger.info("Downloading ERA5 monthly")
        downloader.get_era5(lat=asset_df["Latitude"].mean(),
                            lon=asset_df["Longitude"].mean(),
                            save_pathname=path+"//era5_monthly_10m//",
                            save_filename=asset+"_era5_monthly_10m")

    logger.info("Reading ERA5 monthly")
    reanalysis_era5_monthly_df = pd.read_csv(path+"//era5_monthly_10m//"+asset+"_era5_monthly_10m.csv")
    reanalysis_dict.update(dict(era5_monthly=reanalysis_era5_monthly_df))

    # MERRA2 monthly 10m
    if not os.path.exists(path+"//_merra2_monthly_10m//"+asset+"_merra2_monthly_10m.csv"):
        logger.info("Downloading MERRA2 monthly")
        downloader.get_merra2(lat=asset_df["Latitude"].mean(),
                            lon=asset_df["Longitude"].mean(),
                            save_pathname=path+"//merra2_monthly_10m//",
                            save_filename=asset+"_merra2_monthly_10m")

    logger.info("Reading MERRA2 monthly")
    reanalysis_merra2_monthly_df = pd.read_csv(path+"//merra2_monthly_10m//"+asset+"_merra2_monthly_10m.csv")
    reanalysis_dict.update(dict(merra2_monthly=reanalysis_merra2_monthly_df))


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

        "era5_monthly": {
          "frequency": "1MS",
          "surface_pressure": "surf_pres_Pa",
          "temperature": "temperature_K",
          "time": "datetime",
          "windspeed":"windspeed_ms",
        },
        
        "merra2_monthly": {
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

    with open(path+"//plant_meta.json", "w") as outfile:
        json.dump(asset_json, outfile, indent=2)
        
    with open(path+"//plant_meta.yml", "w") as outfile:
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
            metadata=path+"//plant_meta.yml",
            scada=scada_df,
            meter=meter_df,
            curtail=curtail_df,
            asset=asset_df,
            reanalysis=reanalysis_dict,
        )

        return plantdata


if __name__ == "__main__":
    prepare()
