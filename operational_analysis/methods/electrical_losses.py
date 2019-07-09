# This class defines key analytical routines for calculating electrical losses for 
# a wind plant using operational data. Electrical loss is calculated per month and on 
# an average annual basis by comparing monthly energy production from the turbines
# and the revenue meter

import pandas as pd
import numpy as np

from operational_analysis import logged_method_call
from operational_analysis import logging

logger = logging.getLogger(__name__)

class ElectricalLosses(object):
    """
    A serial (Pandas-driven) implementation of calculating the average monthly and annual
    electrical losses at a wind plant. Energy output from the turbine SCADA meter and the 
    wind plant revenue meter are used to estimate electrical losses.
    
    """ 

    @logged_method_call
    def __init__(self, plant):
        """
        Initialize electrical losses class with input parameters

        Args:
         plant(:obj:`PlantData object`): PlantData object from which EYAGapAnalysis should draw data.

        """
        logger.info("Initializing Electrical Losses Object")
        
        self._plant = plant
        
        self._time_conversion = {'1T': 1.,
                                 '5T': 5.,
                                 '10T': 10.,
                                 '30T': 30.,
                                 '1H': 60.}
        
        self._min_per_hour = 60 # Mintues per hour converter
        self._hours_per_day= 24 # Hours per day converter
        
    @logged_method_call
    def run(self):
        """
        Run the electrical loss calculation in order by calling this function.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        self.process_scada()
        self.process_meter()
        self.calculate_electrical_losses()
       
    @logged_method_call
    def process_scada(self):
        """
        Calculate daily sum of turbine energy only for days when all turbines are reporting
        at all time steps.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        logger.info("Processing SCADA data")
        
        scada_df = self._plant._scada.df 
        
        # Sum up SCADA data power and energy and count number of entries        
        scada_sum = scada_df.groupby(scada_df.index)[['energy_kwh', 'wtur_W_avg']].sum()
        scada_sum['count'] = scada_df.groupby(scada_df.index)[['energy_kwh']].count()
        
        # Calculate daily sum of all turbine energy production and count number of entries
        self._scada_daily = scada_sum.resample('D')['energy_kwh'].sum().to_frame()
        self._scada_daily.columns = ['turbine_energy_kwh']
        self._scada_daily['count'] = scada_sum.resample('D')['count'].sum()
        
        # Specify expected count provided all turbines reporting
        expected_count = self._hours_per_day * self._min_per_hour / \
                         self._time_conversion[self._plant._scada_freq] * self._plant._num_turbines
        
        # Keep only data with all turbines reporting for every time step during the day
        self._scada_daily = self._scada_daily[self._scada_daily['count'] == expected_count]
       
    @logged_method_call
    def process_meter(self):
        """
        Calculate daily sum of meter energy only for days when meter data is reporting at all time steps.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        logger.info("Processing meter data")
        
        meter_df = self._plant._meter.df
        
        # Sum up meter data to daily
        self._meter_daily = meter_df.resample('D').sum()
        self._meter_daily['mcount'] = meter_df.resample('D')['energy_kwh'].count()
        
        # Specify expected count provided all timestamps reporting
        expected_mcount = self._hours_per_day * self._min_per_hour / \
                          self._time_conversion[self._plant._meter_freq]
        
        # Keep only data with all turbines reporting for every time step during the day
        self._meter_daily = self._meter_daily[self._meter_daily['mcount'] == expected_mcount]

    @logged_method_call
    def calculate_electrical_losses(self):                
        """
        Calculaate electrical losses based on the difference in the sum of turbine and metered energy over the
        compiled days.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        logger.info("Calculating electrical losses")
        
        # Merge the SCADA and meter data and drop NaN data
        merge_df = self._meter_daily.join(self._scada_daily)
        merge_df.dropna(inplace = True)
        
        # Calculate electrical loss from difference of sum of turbine and meter energy
        merge_sum = merge_df.sum(axis = 0)
        self._total_turbine_energy = merge_sum['turbine_energy_kwh']
        self._total_meter_energy = merge_sum['energy_kwh']
        self._electrical_losses = 1 - self._total_meter_energy/self._total_turbine_energy