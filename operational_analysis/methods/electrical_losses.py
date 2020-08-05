# This class defines key analytical routines for calculating electrical losses for 
# a wind plant using operational data. Electrical loss is calculated per month and on 
# an average annual basis by comparing monthly energy production from the turbines
# and the revenue meter

from operational_analysis import logged_method_call
from operational_analysis import logging
import numpy as np
from tqdm import tqdm
import pandas as pd

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
    def __init__(self, plant, UQ = False, num_sim = 20000):
        """
        Initialize electrical losses class with input parameters
        
        Args:
         plant(:obj:`PlantData object`): PlantData object from which EYAGapAnalysis should draw data.
         num_sim:(:obj:`int`): number of Monte Carlo simulations
         UQ:(:obj:`bool`): choice whether to perform (True) or not (False) uncertainty quantification                                      
        """
        logger.info("Initializing Electrical Losses Object")
        
        # Check that selected UQ is allowed
        if UQ == True:
            logger.info("Note: uncertainty quantification will be performed in the calculation")
            self.num_sim = num_sim 
        elif UQ == False:
            logger.info("Note: uncertainty quantification will NOT be performed in the calculation")
            self.num_sim = 1
        else:
            raise ValueError("UQ has to either be True (uncertainty quantification performed, default) or False (uncertainty quantification NOT performed)")
        self.UQ = UQ
        
        self._plant = plant
        
        self._min_per_hour = 60 # Mintues per hour converter
        self._hours_per_day= 24 # Hours per day converter
    
    @logged_method_call
    def run(self, uncertainty_meter=0.005, uncertainty_scada=0.005,
                 uncertainty_correction_thresh=0.95):
        """
        Run the electrical loss calculation in order by calling this function.
        
        Args:
         uncertainty_meter(:obj:`float`): uncertainty imposed to revenue meter data (for UQ = True case)
         uncertainty_scada(:obj:`float`): uncertainty imposed to scada data (for UQ = True case)
         uncertainty_correction_thresh(:obj:`tuple`): Data availability thresholds (fractions) 
                                                         under which months should be eliminated. 
                                                         This should be a tuple in the UQ = True case, 
                                                         a single value when UQ = False.
        Returns:
            (None)
        """
        # Define uncertainties and check types
        expected_type = float if self.UQ == False else tuple
        assert type(uncertainty_correction_thresh) == expected_type,  f"uncertainty_correction_thresh must be {expected_type} for UQ={self.UQ}"

        self.uncertainty_correction_thresh = np.array(uncertainty_correction_thresh, dtype=np.float64)  
        if self.UQ == True:
            self.uncertainty_meter = uncertainty_meter
            self.uncertainty_scada = uncertainty_scada
        
        # Process SCADA data to daily sums
        self.process_scada()
        
        # Process meter data to daily sums (if time frequency is less than monthly)
        self._monthly_meter = True # Keep track of reported meter data frequency
        
        if (self._plant._meter_freq != 'MS') & (self._plant._meter_freq != 'M') & (self._plant._meter_freq != '1MS'):
            self.process_meter()
            self._monthly_meter = False # Set to false if sub-monthly frequency
        
        # Setup Monte Carlo approach
        self.setup_inputs()
        
        # Calculate electrical losses, Monte Carlo approach
        self.calculate_electrical_losses()
 
    def setup_inputs(self):
        """
        Create and populate the data frame defining the simulation parameters.
        This data frame is stored as self._inputs

        Args:
            (None)
            
        Returns:
            (None)
        """
        if self.UQ == True:
            inputs = {
                "meter_data_fraction": np.random.normal(1, self.uncertainty_meter, self.num_sim),
                "scada_data_fraction": np.random.normal(1, self.uncertainty_scada, self.num_sim),
                "correction_threshold": np.random.randint(self.uncertainty_correction_thresh[0]*1000, self.uncertainty_correction_thresh[1]*1000,
                                                        self.num_sim) / 1000.  
            } 
            self._inputs = pd.DataFrame(inputs)

        if self.UQ == False:
            inputs = {
                "meter_data_fraction": 1,
                "scada_data_fraction": 1,
                "correction_threshold": self.uncertainty_correction_thresh,
            }
            self._inputs = pd.DataFrame(inputs,index=[0])

        self._electrical_losses = np.empty([self.num_sim,1]) 
        
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
                         (pd.to_timedelta(self._plant._scada_freq).seconds/60) * self._plant._num_turbines

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
                          (pd.to_timedelta(self._plant._meter_freq).seconds/60)
        
        # Keep only data with all turbines reporting for every time step during the day
        self._meter_daily = self._meter_daily[self._meter_daily['mcount'] == expected_mcount]
            
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

        # Loop through number of simulations, calculate losses each time, store results
        for n in tqdm(np.arange(self.num_sim)):
        
            self._run = self._inputs.loc[n]
            
            meter_df = self._plant._meter.df
              
            # If monthly meter data, sum the corrected daily turbine energy to monthly and merge with meter
            if self._monthly_meter:
                
                scada_monthly = self._scada_daily.resample('MS')['corrected_energy'].sum().to_frame()
                scada_monthly.columns = ['turbine_energy_kwh']
    
                # Determine availability for each month represented
                scada_monthly['count'] = self._scada_sum.resample('MS')['count'].sum()
                scada_monthly['expected_count_monthly'] = scada_monthly.index.daysinmonth * self._hours_per_day * self._min_per_hour / \
                             (pd.to_timedelta(self._plant._scada_freq).seconds/60) * self._plant._num_turbines 
                scada_monthly['perc'] = scada_monthly['count']/scada_monthly['expected_count_monthly']
                                
                # Filter out months in which there was less than x% of total running (all turbines at all timesteps)  
                scada_monthly = scada_monthly.loc[scada_monthly['perc']>= self._run.correction_threshold, :]
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
            self._total_turbine_energy = merge_sum['turbine_energy_kwh'] * self._run.scada_data_fraction
            self._total_meter_energy = merge_sum['energy_kwh'] * self._run.meter_data_fraction

            self._electrical_losses[n] = 1 - self._total_meter_energy/self._total_turbine_energy





