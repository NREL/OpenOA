"""
This module fetches metadata of wind farms
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import eia
import numpy as np
import pandas as pd

from openoa.utils import unit_conversion


if TYPE_CHECKING:
    from openoa import PlantData


def fetch_eia(
    api_key: str,
    plant_id: str,
    file_path: str | Path,
    plant_file: str | Path,
    plant_sheet: str | Path,
    wind_file: str | Path,
    wind_sheet: str | Path,
):
    """
    Read in EIA data of wind farm of interest:
     - from EIA API for monthly productions, return monthly net energy generation time series
     - from local Excel files for wind farm metadata, return dictionary of metadata

    Args:
        api_key(:obj:`str`): 32-character user-specific API key, obtained from EIA.
        plant_id(:obj:`str`): 5-character EIA power plant code.
        file_path(:obj:`str`): Directory with EIA metadata .xlsx files.
        plant_file(:obj:`str` | `Path`): Name of the plant metadata Excel file in :py:attr:`file_path`.
            Formerly hard-coded to: "2___Plant_Y2017.xlsx".
        plant_sheet(:obj:`str`): The name of the sheet containing the data in :py:attr:`plant_file`.
            Formerly hard-coded as "Plant".
        wind_file(:obj:`str` | `Path`): Name of the wind metadata Excel file in :py:attr:`file_path`.
            Formerly hard-coded to: ""3_2_Wind_Y2017.xlsx".
        wind_sheet(:obj:`str`): The name of the sheet containing the data in :py:attr:`plant_file`.
            Formerly hard-coded as "Operable".

    Returns:
        :obj:`pandas.Series`: monthly net energy generation in MWh
        :obj:`dictionary`: metadata of the wind farm with 'plant_id'

    """
    file_path = Path(file_path).resolve()
    # EIA metadata

    plant_var_list = [
        "City",
        "Latitude",
        "Longitude",
        "Balancing Authority Name",
        "Transmission or Distribution System Owner",
    ]

    wind_var_list = [
        "Utility Name",
        "Plant Name",
        "State",
        "County",
        "Nameplate Capacity (MW)",
        "Operating Month",
        "Operating Year",
        "Number of Turbines",
        "Predominant Turbine Manufacturer",
        "Predominant Turbine Model Number",
        "Turbine Hub Height (Feet)",
    ]

    def meta_dic_fn(metafile: str | Path, sheet: str, var_list: list[str]):
        all_plant = pd.read_excel(file_path / metafile, sheet_name=sheet, skiprows=1)

        eia_plant = all_plant.loc[
            all_plant["Plant Code"] == np.int64(plant_id)
        ]  # specific wind farm

        if eia_plant.shape[0] == 0:  # Couldn't locate EIA ID in database
            raise Exception("Plant ID not found in EIA database")

        eia_info = eia_plant[var_list]  # select column
        eia_info = eia_info.reset_index(drop=True)  # reset index to 0
        eia_dic = eia_info.T.to_dict()  # convert to dictionary
        out_dic = eia_dic[0]  # remove extra level of dictionary, "0" in this case

        return out_dic

    # file path with 2017 EIA metadata files
    plant_dict = meta_dic_fn(plant_file, plant_sheet, plant_var_list)
    wind_dict = meta_dic_fn(wind_file, wind_sheet, wind_var_list)

    # convert feet to meter and delete reference to feet
    hub_height = np.round(
        unit_conversion.convert_feet_to_meter(wind_dict["Turbine Hub Height (Feet)"])
    )
    wind_dict.update({"Turbine Hub Height (m)": hub_height})
    wind_dict.pop("Turbine Hub Height (Feet)", None)
    out_dic = plant_dict.copy()
    out_dic.update(wind_dict)  # append dictionary

    # EIA monthly energy production data

    api = eia.API(api_key)  # get data from EIA

    series_search_m = api.data_by_series(series="ELEC.PLANT.GEN.%s-ALL-ALL.M" % plant_id)
    eia_monthly = pd.DataFrame(series_search_m)  # net monthly energy generation of wind farm in MWh
    eia_monthly.columns = ["eia_monthly_mwh"]  # rename column
    eia_monthly = eia_monthly.set_index(
        pd.DatetimeIndex(eia_monthly.index)
    )  # convert to DatetimeIndex

    return eia_monthly, out_dic


def attach_eia_data(
    project: PlantData,
    api_key: str,
    plant_id: str,
    file_path: str | Path,
    plant_file: str | Path,
    plant_sheet: str | Path,
    wind_file: str | Path,
    wind_sheet: str | Path,
):
    """
    Assign EIA meta data to PlantData object, which is by default an empty dictionary.

    Args:
        project(:obj:`PlantData`): PlantData object for a particular project
        api_key(:obj:`str`): 32-character user-specific API key, obtained from EIA.
        plant_id(:obj:`str`): 5-character EIA power plant code.
        file_path(:obj:`str`): Directory with EIA metadata .xlsx files.
        plant_file(:obj:`str` | `Path`): Name of the plant metadata Excel file in :py:attr:`file_path`.
            Formerly hard-coded to: "2___Plant_Y2017.xlsx".
        plant_sheet(:obj:`str`): The name of the sheet containing the data in :py:attr:`plant_file`.
        wind_file(:obj:`str` | `Path`): Name of the wind metadata Excel file in :py:attr:`file_path`.
            Formerly hard-coded to: ""3_2_Wind_Y2017.xlsx".
        wind_sheet(:obj:`str`): The name of the sheet containing the data in :py:attr:`plant_file`.

    Returns:
        (None)
    """
    project.eia["api_key"] = api_key
    project.eia["data_dir"] = file_path
    project.eia["eia_id"] = plant_id
    project.eia["monthly_energy"], project.eia["meta_data"] = fetch_eia(
        api_key, plant_id, file_path, plant_file, plant_sheet, wind_file, wind_sheet
    )
