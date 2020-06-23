######################################
# Data import script for Project Engie #
######################################
"""
This is the import script for project Engie. Below is a description of data quality for each data frame
and an overview of the steps taken to correct the raw data.

1. Turbine data frame

"""
from operational_analysis.types import PlantData

import numpy as np
import pandas as pd
import operational_analysis.toolkits.timeseries as ts
import operational_analysis.toolkits.unit_conversion as un
from operational_analysis.toolkits import filters

from operational_analysis import logged_method_call
from operational_analysis import logging

logger = logging.getLogger(__name__)

class Project_Engie(PlantData):
    """This class loads data for the Engie site into a PlantData object"""

    def __init__(self, path="./",
                 name="engie", engine="pandas"):

        super(Project_Engie, self).__init__(path, name, engine)

    def prepare(self):
        """
        Do all loading and preparation of the data for this plant.
        """     
        # Set time frequencies of data in minutes
        self._scada_freq = '10T' # 10-min 
        
        # Load meta data
        self._lat_lon = (48.4461, 5.5925)
        self._plant_capacity = 8.2 # MW
        self._num_turbines = 4
        self._turbine_capacity = 2.05 # MW
        
        ###################
        # SCADA DATA #
        ###################
        logger.info("Loading SCADA data")
        self._scada.load(self._path, "engie_scada", "csv")  # Load Scada data
        logger.info("SCADA data loaded")
        
        logger.info("Timestamp QC and conversion to UTC")
        # Get 'time' field in datetime format
        self._scada.df['time']=pd.to_datetime(self._scada.df['time'])

        # Convert local to UTC time, simple shift forward since no DST present in data
        self._scada.df['time_utc'] = self._scada.df['time'] + pd.Timedelta(hours = 0)
        
        # Remove duplicated timestamps and turbine id
        self._scada.df = self._scada.df[self._scada.df.duplicated(subset = ['time', 'ID']) == False]
        
        # Set time as index
        self._scada.df['time'] = self._scada.df['time_utc']
        self._scada.df.set_index('time',inplace=True,drop=False) # Set datetime as index
        
        logger.info("Correcting for out of range of power, wind speed, and wind direction variables")
        #Handle extrema values
        self._scada.df = self._scada.df[(self._scada.df["wmet_wdspd_avg"]>=0.0) & (self._scada.df["wmet_wdspd_avg"]<=40.0)]
        self._scada.df = self._scada.df[(self._scada.df["wtur_W_avg"]>=-1000.0) & (self._scada.df["wtur_W_avg"]<=2200.0)]
        self._scada.df = self._scada.df[(self._scada.df["wmet_wDir_avg"]>=0.0) & (self._scada.df["wmet_wDir_avg"]<=360.0)]            

        logger.info("Flagging unresponsive sensors")
        #Flag repeated values from frozen sensors
        temp_flag = filters.unresponsive_flag(self._scada.df["wmet_wdspd_avg"], 3)
        self._scada.df.loc[temp_flag, 'wmet_wdspd_avg'] = np.nan
        temp_flag = filters.unresponsive_flag(self._scada.df["wmet_wDir_avg"], 3)
        self._scada.df.loc[temp_flag, 'wmet_wDir_avg'] = np.nan
        
        # Put power in watts; note although the field name suggests 'watts', it was really reporting in kw
        self._scada.df["Power_W"] = self._scada.df["wtur_W_avg"] * 1000
        
        # Calculate energy
        self._scada.df['energy_kwh'] = un.convert_power_to_energy(self._scada.df["wtur_W_avg"], self._scada_freq)
        
        logger.info("Converting field names to IEC 61400-25 standard")
        #Map to -25 standards

        scada_map = {"time"                 : "time",
                     "ID"       : "id",
                     "Power_W"              : "wtur_W_avg",
                     "wmet_wdspd_avg"    : "wmet_wdspd_avg", 
                     "wmet_wDir_avg"    : "wmet_wDir_avg"
                     }

        self._scada.df.rename(scada_map, axis="columns", inplace=True)
        
        # Remove the fields we are not yet interested in
        self._scada.df.drop(['time_utc'], axis=1, inplace=True)