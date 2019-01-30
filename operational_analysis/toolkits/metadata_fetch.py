"""
This module fetches metadata of wind farms
"""

import eia
import pandas as pd
import numpy as np
from operational_analysis.toolkits import unit_conversion


def fetch_eia(api_key, plant_id, file_path):
    """
    Read in EIA data of wind farm of interest
    - from EIA API for monthly productions, return monthly net energy generation time series
    - from local Excel files for wind farm metadata, return dictionary of metadata

    Args:
        api_key(:obj:`string`): 32-character user-specific API key, obtained from EIA
        plant_id(:obj:`string`): 5-character EIA power plant code
        file_path(:obj:`string`): directory with EIA metadata .xlsx files in 2017

    Returns:
        :obj:`pandas.Series`: monthly net energy generation in MWh
        :obj:`dictionary`: metadata of the wind farm with 'plant_id'

    """

    # EIA metadata

    plant_var_list = ['City', 'Latitude', 'Longitude', 'Balancing Authority Name',
                      'Transmission or Distribution System Owner']

    wind_var_list = ['Utility Name', 'Plant Name', 'State', 'County', 'Nameplate Capacity (MW)',
                     'Operating Month', 'Operating Year', 'Number of Turbines',
                     'Predominant Turbine Manufacturer', 'Predominant Turbine Model Number',
                     'Turbine Hub Height (Feet)']

    def meta_dic_fn(metafile, sheet, var_list):
        all_plant = pd.read_excel(file_path + metafile, sheet_name=sheet, skiprows=1)

        eia_plant = all_plant.loc[all_plant['Plant Code'] == np.int(plant_id)]  # specific wind farm

        if (eia_plant.shape[0] == 0):  # Couldn't locate EIA ID in database
            raise Exception('Plant ID not found in EIA database')

        eia_info = eia_plant[var_list]  # select column
        eia_info = eia_info.reset_index(drop=True)  # reset index to 0
        eia_dic = eia_info.T.to_dict()  # convert to dictionary
        out_dic = eia_dic[0]  # remove extra level of dictionary, "0" in this case

        return (out_dic)

    # file path with 2017 EIA metadata files
    plant_dic = meta_dic_fn('2___Plant_Y2017.xlsx', 'Plant', plant_var_list)
    wind_dic = meta_dic_fn('3_2_Wind_Y2017.xlsx', 'Operable', wind_var_list)

    # convert feet to meter
    hubheight_meter = np.round(unit_conversion.convert_feet_to_meter(wind_dic['Turbine Hub Height (Feet)']))
    wind_dic.update({'Turbine Hub Height (m)': hubheight_meter})
    wind_dic.pop('Turbine Hub Height (Feet)', None)  # delete hub height in feet
    out_dic = plant_dic.copy()
    out_dic.update(wind_dic)  # append dictionary

    # EIA monthly energy production data

    api = eia.API(api_key)  # get data from EIA

    series_search_m = api.data_by_series(series='ELEC.PLANT.GEN.%s-ALL-ALL.M' % plant_id)
    eia_monthly = pd.DataFrame(series_search_m)  # net monthly energy generation of wind farm in MWh
    eia_monthly.columns = ['eia_monthly_mwh']  # rename column
    eia_monthly = eia_monthly.set_index(pd.DatetimeIndex(eia_monthly.index))  # convert to DatetimeIndex

    return eia_monthly, out_dic

def add_eia_meta_to_project(project, api_key, plant_id, file_path):
    """
    Assign EIA meta data to PlantData object.
    
    Args:
        project(:obj:`PlantData`): PlantData object for a particular project
        api_key(:obj:`string`): 32-character user-specific API key, obtained from EIA
        plant_id(:obj:`string`): 5-character EIA power plant code
        file_path(:obj:`string`): directory with EIA metadata .xlsx files

    Returns:
        (None)
    """
      
    project._eia = {}
    project._eia['api_key'] = api_key
    project._eia['data_dir'] = file_path
    project._eia['eia_id'] = plant_id
    project._eia['monthly_energy'], project._eia['meta_data'] = fetch_eia(api_key, plant_id, file_path)
    