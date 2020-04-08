# This class defines key analytical routines for calculating electrical losses for 
# a wind plant using operational data. Electrical loss is calculated per month and on 
# an average annual basis by comparing monthly energy production from the turbines
# and the revenue meter

from operational_analysis import logged_method_call
from operational_analysis import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ElectricalLosses(object):
    """
    A serial (Pandas-driven) implementation of calculating the average monthly and annual
    electrical losses at a wind plant, and their uncertainty. 
    Energy output from the turbine SCADA meter and the wind plant revenue 
    meter are used to estimate electrical losses.
    
    The approach is to first calculate daily sums of turbine and revenue meter energy over the 
    plant period of record. Only those days where all turbines and the revenue meter were 
    reporting for all timesteps are considered. Electrical loss is then the difference in 
    total turbine energy production and meter production over those concurrent days. 
    
    A Monte Carlo approach is applied to sample revenue meter data and SCADA data
    with a 0.5% imposed uncertainty, and one filtering parameter is sampled too.
    The uncertainty in estimated electrical losses is quantified as standard deviation
    of the distribution of losses obtained from the MC sampling.
    
    In the case that meter data is not provided on a daily or sub-daily basis (e.g. monthly), a
    different approach is implemented. The sum of daily turbine energy is corrected for any missing 
    reported energy data from the turbines based on the ratio of expected number of data counts per day 
    to the actual. Daily corrected sum of turbine energy is then summed on a monthly basis. Electrical 
    loss is then the difference between total corrected turbine energy production and meter production 
    over those concurrent months.   
    """ 

    @logged_method_call
    def __init__(self, plant, num_sim, uncertainty_meter=0.005, uncertainty_scada=0.005,
                 uncertainty_correction_thresh=(0.9,0.995)):
        """
        Initialize electrical losses class with input parameters

        Args:
         plant(:obj:`PlantData object`): PlantData object from which EYAGapAnalysis should draw data.

        """
        logger.info("Initializing Electrical Losses Object")
        
        self._plant = plant
        self.num_sim = num_sim
        
        self._time_conversion = {'1T': 1.,
                                 '5T': 5.,
                                 '10T': 10.,
                                 '30T': 30.,
                                 '1H': 60.,
                                 'D': 60 * 24}
        
        self._min_per_hour = 60 # Mintues per hour converter
        self._hours_per_day= 24 # Hours per day converter
    
        # Define relevant uncertainties, to be applied in Monte Carlo sampling
        self.uncertainty_meter = np.float64(uncertainty_meter)
        self.uncertainty_scada = np.float64(uncertainty_scada)
        self.uncertainty_correction_thresh = np.array(uncertainty_correction_thresh, dtype=np.float64)  


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
        
        # Monte Carlo sampling
        self.setup_monte_carlo_inputs()
        
        # Calculate electrical losses, Monte Carlo approach
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
        self._scada_daily['perc'] = self._scada_daily['count']/expected_count
                                                
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
    def setup_monte_carlo_inputs(self):
        """
        Perform Monte Carlo sampling of parameters for UQ

        Args:
            (none)

        Returns:
            (None)
        """
        
        self._mc_metered_energy_fraction = np.random.normal(1, self.uncertainty_meter, self.num_sim)        
        self._mc_scada_data_fraction = np.random.normal(1, self.uncertainty_scada, self.num_sim)        
        self._mc_correction_thresh = np.random.randint(self.uncertainty_correction_thresh[0]*1000, self.uncertainty_correction_thresh[1]*1000,
                                                        self.num_sim) / 1000.  
        
    @logged_method_call
    def calculate_electrical_losses(self):                
        """
        Apply Monte Carlo approach to calculate electrical losses and their
        uncertainty based on the difference in the sum of turbine and metered 
        energy over the compiled days.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        logger.info("Calculating electrical losses")
        
        # Initialize variable to store results
        self._electrical_losses =  np.empty(self.num_sim)

        # Loop through number of simulations, calculate losses each time, store results
        for n in tqdm(np.arange(self.num_sim)):
        
            meter_df = self._plant._meter.df
              
            # If monthly meter data, sum the corrected daily turbine energy to monthly and merge with meter
            if self._monthly_meter:
                
                scada_monthly = self._scada_daily.resample('MS')['corrected_energy'].sum().to_frame()
                scada_monthly.columns = ['turbine_energy_kwh']
    
                # Determine availability for each month represented
                scada_monthly['count'] = self._scada_sum.resample('MS')['count'].sum()
                scada_monthly['expected_count_monthly'] = scada_monthly.index.daysinmonth * self._hours_per_day * self._min_per_hour / \
                             self._time_conversion[self._plant._scada_freq] * self._plant._num_turbines 
                scada_monthly['perc'] = scada_monthly['count']/scada_monthly['expected_count_monthly']
                                
                # Filter out months in which there was less than x% of total running (all turbines at all timesteps)
                self._correction_thresh = self._mc_correction_thresh[n]
                
                scada_monthly = scada_monthly.loc[scada_monthly['perc']>= self._correction_thresh, :]
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
            self.mc_total_turbine_energy = self._total_turbine_energy * self._mc_scada_data_fraction[n]
            self._total_meter_energy = merge_sum['energy_kwh']
            self.mc_total_meter_energy = self._total_meter_energy * self._mc_metered_energy_fraction[n]

            self._electrical_losses[n] = 1 - self.mc_total_meter_energy/self.mc_total_turbine_energy

        
        
        
        
      
        
        
        
        
        
        
        