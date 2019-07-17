# This class defines key analytical routines for calculating electrical losses for 
# a wind plant using operational data. Electrical loss is calculated per month and on 
# an average annual basis by comparing monthly energy production from the turbines
# and the revenue meter

from operational_analysis import logged_method_call
from operational_analysis import logging

logger = logging.getLogger(__name__)

class ElectricalLosses(object):
    """
    A serial (Pandas-driven) implementation of calculating the average monthly and annual
    electrical losses at a wind plant. Energy output from the turbine SCADA meter and the 
    wind plant revenue meter are used to estimate electrical losses.
    
    The approach is to first calculate daily sums of turbine and revenue meter energy over the 
    plant period of record. Only those days where all turbines and the revenue meter were 
    reporting for all timesteps are considered. Electrical loss is then the difference in 
    total turbine energy production and meter production over those concurrent days. 
    
    In the case that meter data is not provided on a daily or sub-daily basis (e.g. monthly), a
    different approach is implemented. The sum of daily turbine energy is corrected for any missing 
    reported energy data from the turbines based on the ratio of expected number of data counts per day 
    to the actual. Daily corrected sum of turbine energy is then summed on a monthly basis. Electrical 
    loss is then the difference between total corrected turbine energy production and meter production 
    over those concurrent months.   
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
                                 '1H': 60.,
                                 'D': 60 * 24}
        
        self._min_per_hour = 60 # Mintues per hour converter
        self._hours_per_day= 24 # Hours per day converter
        self._month_hours = [44640,40320,41760,43200] # number of hours in a month, separated by number of days
        
    @logged_method_call
    def run(self):
        """
        Run the electrical loss calculation in order by calling this function.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        # Process SCADA data to daily sums
        self.process_scada()
        
        # Process meter data to daily sums (if time frequency is less than monthly)
        self._monthly_meter = True # Keep track of reported meter data frequency
        
        if (self._plant._meter_freq != 'MS') & (self._plant._meter_freq != 'M') & (self._plant._meter_freq != '1MS'):
            self.process_meter()
            self._monthly_meter = False # Set to false if sub-monthly frequency
        
        # Calculate electrical losses
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
        scada_sum = scada_df.groupby(scada_df.index)[['energy_kwh']].sum()
        scada_sum['count'] = scada_df.groupby(scada_df.index)[['energy_kwh']].count()
        self._scada_sum = scada_sum
        
        # Calculate daily sum of all turbine energy production and count number of entries
        self._scada_daily = scada_sum.resample('D')['energy_kwh'].sum().to_frame()
        self._scada_daily.columns = ['turbine_energy_kwh']
        self._scada_daily['count'] = scada_sum.resample('D')['count'].sum()
        
        # Specify expected count provided all turbines reporting
        expected_count = self._hours_per_day * self._min_per_hour / \
                         self._time_conversion[self._plant._scada_freq] * self._plant._num_turbines

        # Correct sum of turbine energy for cases with missing reported data
        self._scada_daily['corrected_energy'] = self._scada_daily['turbine_energy_kwh'] * expected_count / \
                                                self._scada_daily['count']
                                                
        # Store daily SCADA data where all turbines reporting for every time step during the day
        self._scada_sub = self._scada_daily[self._scada_daily['count'] == expected_count]
       
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
        Calculate electrical losses based on the difference in the sum of turbine and metered energy over the
        compiled days.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        logger.info("Calculating electrical losses")
        
        meter_df = self._plant._meter.df
          
        # If monthly meter data, sum the corrected daily turbine energy to monthly and merge with meter
        if self._monthly_meter:
            scada_monthly = self._scada_daily.resample('MS')['corrected_energy'].sum().to_frame()
            scada_monthly.columns = ['turbine_energy_kwh']

            # Determine availability for each month represented
            scada_monthly['count'] = self._scada_sum.resample('MS')['count'].sum()
            scada_monthly['expected_count_monthly'] = 1 #initialized values for column

            #months with 31 days
            month_list = ((scada_monthly.index.month ==1) | \
            (scada_monthly.index.month == 3) | \
            (scada_monthly.index.month ==5) | \
            (scada_monthly.index.month ==7) | \
            (scada_monthly.index.month ==8) | \
            (scada_monthly.index.month ==10) | \
            (scada_monthly.index.month ==12))
            scada_monthly.loc[month_list, 'expected_count_monthly'] = \
            self._month_hours[0]* self._plant._num_turbines /self._time_conversion[self._plant._scada_freq]

            #February 
            month_list = (scada_monthly.index.month ==2)
            year_list = ((scada_monthly.index.year == 2000) | \
            (scada_monthly.index.year == 2004) | \
            (scada_monthly.index.year == 2008) | \
            (scada_monthly.index.year == 2012) | \
            (scada_monthly.index.year == 2016) | \
            (scada_monthly.index.year == 2020))

            #Non Leap year
            scada_monthly.loc[(month_list & year_list), 'expected_count_monthly'] = \
            self._month_hours[1]* self._plant._num_turbines /self._time_conversion[self._plant._scada_freq]
            
            #Leap year
            scada_monthly.loc[(month_list & (~year_list)), 'expected_count_monthly'] = \
            self._month_hours[2]* self._plant._num_turbines /self._time_conversion[self._plant._scada_freq]

            #months with 30 days
            month_list = ((scada_monthly.index.month ==4) | \
            (scada_monthly.index.month ==6) | \
            (scada_monthly.index.month ==9) | \
            (scada_monthly.index.month ==11))
            scada_monthly.loc[month_list, 'expected_count_monthly'] = \
            self._month_hours[3]* self._plant._num_turbines / self._time_conversion[self._plant._scada_freq]

            scada_monthly['perc'] = scada_monthly['count']/scada_monthly['expected_count_monthly']
            
            # Filter out months in which there was less than 95% of total running (all turbines at all timesteps)
            scada_monthly = scada_monthly.loc[scada_monthly['perc']>= .95, :]
            merge_df = meter_df.join(scada_monthly)
        
        # If sub-monthly meter data, merge the daily data for which all turbines are reporting at all timestamps
        else:
            # Note 'self._scada_sub' only contains full reported data
            merge_df = self._meter_daily.join(self._scada_sub)
            
        # Drop non-concurrent timestamps and get total sums over concurrent period of record
        merge_df.dropna(inplace = True)
        self._merge_df = merge_df
        merge_sum = merge_df.sum(axis = 0)
        
        # Calculate electrical loss from difference of sum of turbine and meter energy 
        self._total_turbine_energy = merge_sum['turbine_energy_kwh']
        self._total_meter_energy = merge_sum['energy_kwh']
        self._electrical_losses = 1 - self._total_meter_energy/self._total_turbine_energy