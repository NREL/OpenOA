# This class defines key analytical routines for calculating long-term gross energy 
# for each turbine at a wind farm

import pandas as pd

from operational_analysis.toolkits import met_data_processing
from operational_analysis.toolkits import filters
from operational_analysis.toolkits.power_curve import functions

from operational_analysis import logged_method_call
from operational_analysis import logging

logger = logging.getLogger(__name__)

class TurbineLongTermGrossEnergy(object):
    """
    A serial (Pandas-driven) implementation of calculating long-term gross energy 
    for each turbine in a wind farm. This module collects standard processing and
    analysis methods for estimating this metric.

    The method proceeds as follows:

        1. Filter turbine data for normal operation
        2. Calculate daily means of wind speed, wind direction, and air density from reanalysis products
        3. Calculate daily sums of energy from each turbine
        4. Fit daily data (features are atmospheric variables, response is turbine power) using a
           generalized additive model (GAM)
        5. Apply model results to long-term atmospheric varaibles to calculate long term
           gross energy for each turbine

    The end result is a table of long-term gross energy values for each turbine in the wind farm. Note
    that this gross energy metric does not back out losses associated with waking or turbine performance.
    Rather, gross energy in this context is what turbine would have produced under normal operation 
    (i.e. excluding downtime and underperformance).

    Required schema of PlantData:
            _scada_freq
            Reanalysis: merra2, erai, ncep2 with columns ['time', 'u_ms', 'v_ms', 'windspeed_ms', 'rho_kgm-3']
            Scada: columns: ['time', 'id', 'windspeed_ms', 'power_kw', 'energy_kw']

    """

    @logged_method_call
    def __init__(self, plant, max_power_filter = 0.85):
        """
        Initialize turbine long-term gross energy analysis with data and parameters.

        Args:
         plant(:obj:`PlantData object`): PlantData object from which TurbineLongTermGrossEnergy should draw data.

        """
        logger.info("Initializing TurbineLongTermGrossEnergy Object")
        
        self._plant = plant  # Set plant as attribute of analysis object
        self._turbs = self._plant._scada.df['id'].unique() # Store turbine names
        
        self._max_power_filter = max_power_filter # Parameter used for bin-based filtering
        self._reanal = ['merra2', 'erai', 'ncep2'] # Reanalysis products to consider
        
        # Define several dictionaries to be populated within this method
        self._scada_dict = {}
        self._daily_reanal_dict = {}
        self._model_dict = {}
        self._model_results = {}
        self._turb_lt_gross = {}
        
        
        # Dictionary to convert time interval indicator into minutes
        self._time_conversion = {'10T': 10.,
                                 '5T': 5.,
                                 '1H': 60.}
        
        # Set number of 'valid' counts required when summing data to daily values
        self._num_valid_daily = 60. / self._time_conversion[self._plant._scada_freq] * 24
        
        # Initially sort the different turbine data into dictionary entries
        self.sort_scada_by_turbine()
        

    @logged_method_call
    def run(self):
        """
        Perform pre-processing of data into an internal representation for which the analysis can run more quickly.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        logger.info("Filtering turbine data")
        self.filter_turbine_data() # Filter turbine data
        
        logger.info("Processing reanalysis data to daily averages")
        self.setup_daily_reanalysis_data() # Setup daily reanalysis products
        
        logger.info("Setting up daily data for model fitting")
        self.setup_model_dict() # Setup daily data to be fit using the GAM
        
        logger.info("Fitting model data")
        self.fit_model() # Fit daily turbine energy to atmospheric data
        
        logger.info("Applying fitting results to calculate long-term gross energy")
        self.apply_model_to_lt() # Apply fitting result to long-term reanalysis data
     
        # Log the completion of the run
        logger.info("Run completed")
        
    def sort_scada_by_turbine(self):
        """
        Take raw SCADA data in plant object and sort into a dictionary using turbine IDs.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        df = self._plant._scada.df
        dic = self._scada_dict
        
        # Loop through turbine IDs
        for t in self._turbs:
            # Store relevant variables in dictionary
            dic[t] = df.loc[df['id'] == t, ['time', 'windspeed_ms', 'power_kw', 'energy_kwh']]            
            dic[t].set_index('time', inplace = True) # Set datetime as index
            dic[t].sort_index(inplace = True) # Sort data by time
        
    def filter_turbine_data(self):
        """
        Apply a set of filtering algorithms to the turbine wind speed vs power curve to flag
        data not representative of normal turbine operation
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        dic = self._scada_dict
        
        # Loop through turbines
        for t in self._turbs:
            max_bin = self._max_power_filter * dic[t].power_kw.max() # Set maximum range for using bin-filter
            
            # Apply range filter
            dic[t].loc[:,'flag_range'] = filters.range_flag(dic[t].loc[:, 'windspeed_ms'], below = 0, above = 40)
            
            # Apply frozen/unresponsive sensor filter
            dic[t].loc[:,'flag_frozen'] = filters.unresponsive_flag(dic[t].loc[:, 'windspeed_ms'], threshold = 3)
            
            # Apply window range filter
            dic[t].loc[:,'flag_window'] = filters.window_range_flag(window_col = dic[t].loc[:, 'windspeed_ms'], 
                                                                    window_start = 5., 
                                                                    window_end = 40, 
                                                                    value_col = dic[t].loc[:, 'power_kw'], 
                                                                    value_min = 20., 
                                                                    value_max = 2000.)
            
            # Apply bin-based filter
            dic[t].loc[:,'flag_bin'] = filters.bin_filter(bin_col = dic[t].loc[:, 'power_kw'], 
                                                          value_col = dic[t].loc[:, 'windspeed_ms'], 
                                                          bin_width = 100, 
                                                          threshold = 2., 
                                                          center_type = 'median', 
                                                          bin_min = 20., 
                                                          bin_max = max_bin, 
                                                          threshold_type = 'scalar', 
                                                          direction = 'all')
            
            # Create a 'final' flag which is true if any of the previous flags are true
            dic[t].loc[:, 'flag_final'] = (dic[t].loc[:, 'flag_range']) | \
                                          (dic[t].loc[:, 'flag_window']) | \
                                          (dic[t].loc[:, 'flag_bin']) | \
                                          (dic[t].loc[:, 'flag_frozen'])
            
    def plot_filtered_power_curves(self, save_folder, output_to_terminal = False):
        """
        Plot the raw and flagged power curve data and save to file.
        
        Args:
            save_folder('obj':'str'): The pathname to where figure files should be saved
            output_to_terminal('obj':'boolean'): Indicate whether or not to output figures to terminal
            
        Returns:
            (None)
        """
        import matplotlib.pyplot as plt
        dic = self._scada_dict
        
        # Loop through turbines
        for t in self._turbs:
            filt_df = dic[t].loc[dic[t]['flag_final']] # Filter only for valid data
            
            plt.figure(figsize = (6,5))
            plt.scatter(dic[t].windspeed_ms, dic[t].power_kw, s=1, label = 'Raw') # Plot all data
            plt.scatter(filt_df['windspeed_ms'], filt_df['power_kw'], s=1, label = 'Flagged') # Plot flagged data
            plt.xlim(0, 30)
            plt.xlabel('Wind speed (m/s)')
            plt.ylabel('Power (kW)')
            plt.title('Filtered power curve for Turbine %s' %t)
            plt.legend(loc = 'lower right')
            plt.savefig('%s/%s_filtered_pc.png' %(save_folder, t,), dpi = 200) # Save file
            
            # Output figure to terminal if desired
            if output_to_terminal:
                plt.show()
            
            plt.close()
    
    def setup_daily_reanalysis_data(self):
        """
        Process reanalysis data to daily means for later use in the GAM model
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        dic = self._daily_reanal_dict
        
        # Loop through reanalysis products
        for r in self._reanal:
            reanal = self._plant._reanalysis._product[r].df
            df_daily = reanal.resample('D')['u_ms', 'v_ms', 'windspeed_ms', 'rho_kgm-3'].mean() # Get daily means
            
            # Recalculate daily average wind direction
            df_daily['winddirection_deg'] = met_data_processing.compute_wind_direction(u = df_daily['u_ms'],
                                                                                       v = df_daily['v_ms'])
            dic[r] = df_daily # Assign result to dictionary
            
    def setup_model_dict(self):
        """
        Setup daily atmospheric variable averages and daily turbine energy sums for use
        in the GAM model
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        scada = self._scada_dict
        reanal = self._daily_reanal_dict
        mod = self._model_dict
        
        # Loop through turbines
        for t in self._turbs:
            scada_filt = scada[t].loc[scada[t]['flag_final'] == False] # Filter for valid data
            scada_daily = scada_filt.resample('D')['energy_kwh'].sum().to_frame() # Calculate daily energy sum
            scada_daily['count'] = scada_filt.resample('D')['energy_kwh'].count() # Count number of entries in sum
            
            # Discard daily sums if any sub-daily data was missing from that calculation
            daily_valid = scada_daily.loc[scada_daily['count'] == self._num_valid_daily]
            
            # Store the valid data to be used for fitting in a separate dictionary                          
            for r in self._reanal: # Loop through reanalysis products
                 mod[t, r] = daily_valid.join(reanal[r])         
      
    def fit_model(self):
        """
        Fit the daily turbine energy sum and atmospheric variable averages using a GAM model
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        mod_dict = self._model_dict
        mod_results = self._model_results
        
        
        for t in self._turbs: # Loop throuh turbines
            logger.info("Fitting turbine %s" %t)
            
            for r in self._reanal: # Loop through reanalysis products
                df = mod_dict[t, r]
                
                # Consider wind speed, wind direction, and air density as features
                mod_results[t, r] = functions.gam_3param(windspeed_column = df['windspeed_ms'],
                                                         winddir_column = df['winddirection_deg'],
                                                         airdens_column = df['rho_kgm-3'],
                                                         power_column=df['energy_kwh'])
        
    def apply_model_to_lt(self):
        """
        Apply model result to the long-term reanalysis data to calculate long-term
        gross energy for each turbine.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        turb_gross = self._turb_lt_gross
        mod_results = self._model_results
        
        # Create a data frame to store final results
        self._summary_results = pd.DataFrame(index = self._reanal, columns = self._turbs)
        
        for r in self._reanal: # Loop throuh reanalysis products
            daily_reanal = self._daily_reanal_dict[r]
            turb_gross[r] = pd.DataFrame(index = daily_reanal.index) # Set empty data frame to store results
            X_long_term = daily_reanal['windspeed_ms'], daily_reanal['winddirection_deg'], daily_reanal['rho_kgm-3']
            
            for t in self._turbs: # Loop through turbines
                turb_gross[r].loc[:, t] = mod_results[t, r](*X_long_term) # Apply GAM fit to long-term reanalysis data
            
            turb_annual = turb_gross[r].resample('AS').sum() # Calculate annual sums of energy from long-term estimate4
            self._summary_results.loc[r, :] = turb_annual.mean(axis = 0) # Store mean annual gross energy in data frame
