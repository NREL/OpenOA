# PrufPlantAnalysis
#
# This class defines key analytical routines for the PRUF/WRA Benchmarking
# standard operational assessment.
#
# The PrufPlantAnalysis object is a factory which instantiates either the Pandas, Dask, or Spark
# implementation depending on what the user prefers.
#
# The resulting object is loaded as a plugin into each PlantData object.

import random

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from operational_analysis.toolkits import met_data_processing as mt
from operational_analysis.toolkits import timeseries as tm
from operational_analysis.toolkits import unit_conversion as un
from operational_analysis.types import timeseries_table


class PlantAnalysis():
    """
    A serial (Pandas-driven) implementation of the benchmark PRUF operational
    analysis implementation. This module collects standard processing and
    analysis methods for estimating plant level operational AEP and uncertainty.

    The preprocessing should run in this order:

        1. Process revenue meter energy - creates monthly data frame, gets revenue meter on monthly basis, and adds data flag
        2. Process loss estimates - add monthly curtailment and availabilty losses to monthly data frame
        3. Process reanalysis data - add monthly density-corrected wind speeds from several reanalysis products to the monthly data frame
        4. Set up Monte Carlo - create the necessary Monte Carlo inputs to the OA process
        5. Run AEP Monte Carlo - run the OA process iteratively to get distribution of AEP results

    The end result is a distribution of AEP results which we use to assess expected AEP and associated uncertainty
    """

    def __init__(self, plant):
        """
        Create a pruf plant analysis timeseries object that is a Timeseries Table.

        Args:
         plant(:obj:`PlantData object`): PlantData object from which PlantAnalysis should draw data.

        """
        self._monthly = timeseries_table.TimeseriesTable.factory(plant._engine)
        self._plant = plant

        # Memo dictionaries help speed up computation
        self.long_term_sampling = {} # Combinations of long-term reanalysis data sampling
        self.outlier_filtering = {} # Combinations of outlier filter results

        # Define relevant uncertainties, data ranges and max thresholds to be applied in Monte Carlo sampling
        self.uncertainty_meter = np.float64(0.005)
        self.uncertainty_losses = np.float64(0.05)
        self.uncertainty_loss_max = np.array((10,20),dtype=np.float64)
        self.uncertainty_windiness = np.array((10,20),dtype=np.float64)
        self.uncertainty_outlier = np.array((2,3.1),dtype=np.float64)
        self.uncertainty_nan_energy = np.float64(0.01)

    def process_revenue_meter_energy(self):
        """
        Initial creation of monthly data frame:
            1. Populate monthly data frame with energy data summed from 10-min QC'd data
            2. For each monthly value, find percentage of NaN data used in creating it and flag if percentage is greater than 0

        Args:
            (None)

        Returns:
            (None)

        """
        df = getattr(self._plant,'meter').df # Get the meter data frame

        # Create the monthly data frame by summing meter energy into yr-mo
        self._monthly.df=(df.resample('MS')['energy_kwh'].sum()/1e6).to_frame() # Get monthly energy values in GWh
        self._monthly.df.rename(columns={"energy_kwh":"energy_gwh"},inplace=True) # Rename kWh to MWh

        # Determine how much 10-min data was missing for each year-month energy value. Flag accordigly if any is missing
        self._monthly.df['energy_nan_perc']=df.resample('MS')['energy_kwh'].apply(tm.percent_nan) # Get percentage of meter data that were NaN when summing to monthly
     
        # Create a column with expected number of days per month (to be used when normalizing to 30-days for regression)
        days_per_month = (pd.Series(self._monthly.df.index)).dt.daysinmonth
        days_per_month.index = self._monthly.df.index
        self._monthly.df['num_days_expected'] = days_per_month
        
        # Get actual number of days per month in the raw data (used when trimming beginning and end of monthly data frame)
        if (self._plant._meter_freq!='1MS') & (self._plant._meter_freq!='1M'): # If meter data has higher resolution than monthly
            self._monthly.df['num_days_actual'] = df.resample('MS')['energy_kwh'].apply(tm.num_days)
        else:
            self._monthly.df['num_days_actual'] = self._monthly.df['num_days_expected']

    def process_loss_estimates(self):
        """
        Append availability and curtailment losses to monthly data frame

        Args:
            (None)

        Returns:
            (None)

        """
        df = getattr(self._plant,'curtail').df
        
        curt_monthly=np.divide(df.resample('MS')[['availability_kwh', 'curtailment_kwh']].sum(), 1e6) # Get sum of avail and curt losses in GWh
        curt_monthly.rename(columns = {'availability_kwh': 'availability_gwh', 'curtailment_kwh': 'curtailment_gwh'}, inplace = True)

        # Merge with revenue meter monthly data
        self._monthly.df=self._monthly.df.join(curt_monthly)

        # Add gross energy field
        self._monthly.df['gross_energy_gwh']= un.compute_gross_energy(self._monthly.df['energy_gwh'], 
                                                                      self._monthly.df['availability_gwh'],
                                                                      self._monthly.df['curtailment_gwh'], 'energy', 'energy')
        
        # Calculate percentage-based losses
        self._monthly.df['availability_pct']= np.divide(self._monthly.df['availability_gwh'], self._monthly.df['gross_energy_gwh'])
        self._monthly.df['curtailment_pct']= np.divide(self._monthly.df['curtailment_gwh'], self._monthly.df['gross_energy_gwh'])
        
        self._monthly.df['avail_nan_perc']=df.resample('MS')['availability_kwh'].apply(tm.percent_nan) # Get percentage of 10-min meter data that were NaN when summing to monthly
        self._monthly.df['curt_nan_perc']=df.resample('MS')['curtailment_kwh'].apply(tm.percent_nan) # Get percentage of 10-min meter data that were NaN when summing to monthly
        
        self._monthly.df['nan_flag']=False # Set flag to false by default
        self._monthly.df.loc[(self._monthly.df['energy_nan_perc']>self.uncertainty_nan_energy) |
                             (self._monthly.df['avail_nan_perc']>self.uncertainty_nan_energy) |
                             (self._monthly.df['curt_nan_perc']>self.uncertainty_nan_energy), 'nan_flag']=True # If more than 1% of data are NaN, set flag to True
        
        # By default, assume all reported losses are representative of long-term operational
        self._monthly.df['availability_typical'] = True
        self._monthly.df['curtailment_typical'] = True
        
        # By default, assume combined availability and curtailment losses are below the threshold to be considered valid
        self._monthly.df['combined_loss_valid'] = True        

    def process_reanalysis_data(self):
        """
        Process reanalysis data for use in PRUF plant analysis
            - calculate density-corrected wind speed
            - get monthly average wind speeds
            - append monthly averages to monthly energy data frame

        Args:
            (None)

        Returns:
            (None)
        """

        self._reanalysis_monthly=pd.DataFrame(index=pd.date_range(start='1997-01-01',end='2020-01-01',freq='MS')) # Define empty data frame that spans past our period of interest

        # Now loop through the different reanalysis products, density-correct wind speeds, and take monthly averages
        for key,items in self._plant._reanalysis._product.iteritems():
            rean_df=self._plant._reanalysis._product[key].df
            rean_df['ws_dens_corr']=mt.air_density_adjusted_wind_speed(rean_df, 'windspeed_ms','rho_kgm-3') # Density correct wind speeds
            self._reanalysis_monthly[key]=rean_df.resample('MS')['ws_dens_corr'].mean()#.to_frame() # Get average wind speed by year-month

        self._monthly.df=self._monthly.df.join(self._reanalysis_monthly) # Merge monthly reanalysis data to monthly energy data frame

    def trim_monthly_df(self):
        """
        Remove first and/or last month of data if the raw data had an incomplete number of days

        Args:
            (None)

        Returns:
            (None)
        """
        for p in self._monthly.df.index[[0,-1]]: # Loop through 1st and last data entry
            if self._monthly.df.loc[p,'num_days_expected'] != self._monthly.df.loc[p,'num_days_actual']:
                self._monthly.df.drop(p,inplace=True) # Drop the row from data frame

    def calculate_long_term_losses(self):
        """
        This function calculates long-term availability and curtailment losses based on the reported monthly data,
        filtering for those data that are deemed representative of average plant performance

        Args:
            (None)

        Returns:
            (tuple):
                :obj:`float`: annualized monthly availability loss expressed as fraction
                :obj:`float`: annualized monthly curtailment loss expressed as fraction
        """
        df = self._monthly.df

        # isolate availabilty and curtailment values that are representative of average plant performance
        avail_valid = df.loc[df['availability_typical'],'availability_gwh'].to_frame()
        curt_valid = df.loc[df['curtailment_typical'],'curtailment_gwh'].to_frame()

        # Now get average percentage losses by month
        avail_long_term=avail_valid.groupby(avail_valid.index.month)['availability_gwh'].mean()
        curt_long_term=curt_valid.groupby(curt_valid.index.month)['curtailment_gwh'].mean()

        # Calculate long-term availbilty and curtailment losses
        self.long_term_losses = (avail_long_term, curt_long_term)
    
    def setup_monte_carlo_inputs(self,reanal_subset, num_sim):
        """
        Perform Monte Carlo sampling for reported monthly revenue meter energy, availability, and curtailment data, as well
        as reanalysis data

        Args:
            reanal_subset(:obj:`list`): list of str data indicating which reanalysis products to use in OA
            num_sim(:obj:`int`): number of simulations to perform

        Returns:
            (None)
        """        
        self._mc_slope = np.empty(num_sim,dtype=np.float64)
        self._mc_intercept = np.empty(num_sim,dtype=np.float64)
        self._mc_num_points = np.empty(num_sim,dtype=np.float64)
        
        self._mc_outlier_threshold = np.random.randint(self.uncertainty_outlier[0]*10,(self.uncertainty_outlier[1])*10,num_sim)/10.
        self._mc_metered_energy_fraction = np.random.normal(1,self.uncertainty_meter,num_sim)
        self._mc_loss_fraction = np.random.normal(1,self.uncertainty_losses,num_sim)
        self._mc_num_years_windiness = np.random.randint(self.uncertainty_windiness[0],self.uncertainty_windiness[1]+1,num_sim)
        self._mc_loss_threshold = np.random.randint(self.uncertainty_loss_max[0],self.uncertainty_loss_max[1]+1,num_sim)/100.

        reanal_list=np.repeat(reanal_subset,num_sim) # Create extra long list of renanalysis product names to sample from
        self._mc_reanalysis_product = np.asarray(random.sample(reanal_list,num_sim))
        
    def filter_outliers(self, reanal, outlier_thresh, comb_loss_thresh):
        '''
        This function filters outliers based on
            1. The reanalysis product 
            2. The Huber parameter which controls sensitivity of outlier detection in robust linear regression 
            3. The combined availability and curtailment loss criteria
        
        There are only 300 combinations of outlier removals:
        (3 reanalysis product x 10 outlier threshold values x 10 combined loss thresholds)
        
        Therefore, we use a memoized funciton to store the regression data in a dictionary for each combination as it
        comes up in the Monte Carlo simulation. This saves significant computational time in not having to run
        robust linear regression for each Monte Carlo iteration
        
        Args:
            reanal(:obj:`string`): The name of the reanalysis product
            outlier_thresh(:obj:`float`): The Huber parameter controlling sensitivity of outlier detection
            comb_loss_thresh(:obj:`float`): The combined availabilty and curtailment monthly loss threshold
        
        Returns:
            :obj:`pandas.DataFrame`: Filtered monthly data ready for linear regression
        '''
        # Check if valid data has already been calculated and stored. If so, just return it
        if (reanal, outlier_thresh, comb_loss_thresh) in self.outlier_filtering:
            valid_data = self.outlier_filtering[(reanal, outlier_thresh, comb_loss_thresh)]
            return valid_data
        
        # If valid data hasn't yet been stored in dictionary, determine the valid data
        df = self._monthly.df
        
        # First set of filters checking combined losses and if the Nan data flag was on
        df_sub = df.loc[((df['availability_pct'] + df['curtailment_pct']) < comb_loss_thresh) & (df['nan_flag'] == False)]
        
        # Now perform robust linear regression using Huber algorithm to flag outliers
        X = sm.add_constant(df_sub[reanal]) # Reanalysis data with constant column
        y = df_sub['gross_energy_gwh'] # Energy data 
        
        # Perform robust linear regression
        rlm = sm.RLM(y, X, M = sm.robust.norms.HuberT(outlier_thresh))
        rlm_results = rlm.fit()
        
        # Define valid data as points in which the Huber algorithm returned a value of 1
        valid_data = df_sub.loc[rlm_results.weights == 1, [reanal, 'energy_gwh', 'availability_gwh', 
                                                           'curtailment_gwh', 'num_days_expected']]
        
        # Update the dictionary
        self.outlier_filtering[(reanal, outlier_thresh, comb_loss_thresh)] = valid_data
        
        # Return result
        return valid_data

    def set_regression_data(self,n):
        """
        This will be called for each iteration of the Monte Carlo simulation and will do the following:
            1. Randomly sample monthly revenue meter, availabilty, and curtailment data based on specified uncertainties and correlations
            2. Randomly choose one reanalysis product (### Not yet implemented ###)
            3. Calculate gross energy from randomzied energy data
            4. Normalize gross energy to 30-day months
            5. Filter results to remove months with NaN data and with combined losses that exceed the Monte Carlo sampled max threhold
            6. Return the wind speed and normalized gross energy to be used in the regression relationship

        Args:
            n(:obj:`int`): The Monte Carlo iteration number

        Returns:
            :obj:`pandas.Series`: Monte-Carlo sampled wind speeds
            :obj:`pandas.Series`: Monte-Carlo sampled normalized gross energy

        """
        # Get data to use in regression based on filtering result
        reg_data = self.filter_outliers(self._mc_reanalysis_product[n], 
                                        self._mc_outlier_threshold[n], 
                                        self._mc_loss_threshold[n])
 
        # Now monte carlo sample the data
        mc_energy = reg_data['energy_gwh'] * self._mc_metered_energy_fraction[n] # Create new Monte-Carlo sampled data frame and sample energy data
        mc_availability = reg_data['availability_gwh'] * self._mc_loss_fraction[n] # Calculate MC-generated availability
        mc_curtailment = reg_data['curtailment_gwh'] * self._mc_loss_fraction[n] # Calculate MC-generated curtailment
        num_days_expected = reg_data['num_days_expected']
        
        # Calculate gorss energy and normalize to 30-days
        mc_gross_energy = mc_energy + mc_availability + mc_curtailment
        mc_gross_norm = mc_gross_energy*30/num_days_expected # Normalize gross energy to 30-day months
        
        # Set reanalysis product
        mc_wind_speed = reg_data[self._mc_reanalysis_product[n]] # Copy wind speed data to Monte Carlo data frame
        
        # Return values needed for linear regression
        return [mc_wind_speed, mc_gross_norm]  # Return randomly sampled wind speed and normalized gross energy
    

    def run_regression(self,n):
        """
        Run robust linear regression between Monte-Carlo generated monthly gross energy and wind speed
        Return Monte-Carlo sampled slope and intercept values (based on their covariance) and report
        the number of outliers based on the robust linear regression result. 

        Args:
            n(:obj:`int`): The Monte Carlo iteration number

        Returns:
            :obj:`float`: Monte-carlo sampled slope
            :obj:`float`: Monte-carlo sampled intercept

        """
        reg_data=self.set_regression_data(n) # Get regression data
        
        p,V = np.polyfit(reg_data[0], reg_data[1], 1, cov=True) # Perform linear regression, return slope, intercept, and variances
        mc_slope, mc_intercept = np.random.multivariate_normal(p,V,1).T # Generate MC-sampled slope and intercept from regression relationship                
        
        # Update Monte Carlo tracker fields
        self._mc_num_points[n] = len(reg_data[0])
        self._mc_slope[n] = np.float(mc_slope)
        self._mc_intercept[n] = np.float(mc_intercept)

        # Return slope and intercept values
        return np.float(mc_slope), np.float(mc_intercept)

    def run_AEP_monte_carlo(self, num_sim):
        """
        Loop through OA process a number of times and return array of AEP results each time

        Args:
           num_sim(:obj:`int`): Number of simulations to perform

        Returns:
            :obj:`numpy.ndarray` Array of AEP, long-term avail, long-term curtailment calculations
        """

        sim_results=pd.DataFrame(index=np.arange(num_sim),data={'aep_GWh':np.empty(num_sim),
                                                                     'avail_pct':np.empty(num_sim),
                                                                     'curt_pct':np.empty(num_sim)})

        num_days_lt=(31,28.25,31,30,31,30,31,31,30,31,30,31) # Define "long-term" or "typical" number of days per month

        # Loop through number of simulations, run regression each time, store AEP results
        for n in tqdm(np.arange(num_sim)):
            slope,intercept=self.run_regression(n) # Get slope, intercept from regression
            ws_lt=self.sample_long_term_reanalysis(self._mc_num_years_windiness[n], self._mc_reanalysis_product[n]) # Get long-term wind speeds
            gross_norm_lt=ws_lt.multiply(slope)+intercept # Get long-term normalized gross energy by applying regression result to long-term monthly wind speeds
            gross_lt=gross_norm_lt*num_days_lt/30 # Undo normalization to 30-day months
            [avail_lt_losses, curt_lt_losses] = self.sample_long_term_losses(n) # Get long-term availability and curtailment losses by month

            # Get total availabilty and curtailment losses per year in MWh
            avail_lt_gwh = avail_lt_losses.sum()
            curt_lt_gwh = curt_lt_losses.sum()

            # Assign AEP, long-term availability, and long-term curtailment to output data frame
            sim_results.loc[n,'aep_GWh'] = gross_lt.sum()-avail_lt_gwh
            sim_results.loc[n,'avail_pct'] = avail_lt_gwh/gross_lt.sum()
            sim_results.loc[n,'curt_pct'] = curt_lt_gwh/gross_lt.sum()

        # Return final output
        return sim_results

    def sample_long_term_reanalysis(self, n, r):
        """
        This function returns the windiness-corrected monthly wind speeds based on the Monte-Carlo generated sample of:
            1. The reanalysis product
            2. The number of years to use in the long-term correction
        
        A memoized approach is used here since there are a finite combination of long-term wind speeds:
            (3 reanalysis products x 10 different year combinations) = 30 total combinations
            
        This memoized approach saves significant computaitonal time in the Monte Carlo simulation

        Args:
           n(:obj:`integer`): The Monte Carlo Iteration Number
           r(:obj:`string`): The reanalysis product used for Monte Carlo sample 'n'

        Returns:
           :obj:`pandas.DataFrame`: the windiness-corrected or 'long-term' annualized monthly wind speeds

        """       
        # Check if valid data has already been calculated and stored. If so, just return it 
        if (r, n) in self.long_term_sampling:
            ws_monthly = self.long_term_sampling[(r, n)]
            return ws_monthly       
        
        # If data hasn't yet been computed, peform the calculation         
        ws_df=self._reanalysis_monthly[r].to_frame().dropna() # Drop NA values from monthly reanalysis data series
        ws_data=ws_df.tail(n*12) # Get last 'x' years of data from reanalysis product
        ws_monthly=ws_data.groupby(ws_data.index.month)[r].mean() # Get long-term annualized monthly wind speeds
        
        # Store result in dictionary
        self.long_term_sampling[(r,n)] = ws_monthly
            
        # Return result
        return ws_monthly
                
    def sample_long_term_losses(self,n):
        """
        This function calculates long-term availability and curtailment losses based on the Monte Carlo sampled historical
        availability and curtailment data

        Args:
            n(:obj:`integer`): The Monte Carlo iteration number

        Returns:
            :obj:`float`: annualized monthly availability loss expressed as fraction
            :obj:`float`: annualized monthly curtailment loss expressed as fraction
        """
        mc_avail = self.long_term_losses[0]*self._mc_loss_fraction[n]
        mc_curt = self.long_term_losses[1]*self._mc_loss_fraction[n]

        # Return availbilty and curtailment long-term monthly data
        return mc_avail, mc_curt
