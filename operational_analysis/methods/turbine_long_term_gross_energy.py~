# This class defines key analytical routines for performing a 'gap-analysis'
# on EYA-estimated annual energy production (AEP) and that from operational data.
# Categories considered are availability, electrical losses, and long-term
# gross energy. The main output is a 'waterfall' plot linking the EYA-
# estimated and operational-estiamted AEP values. 

import pandas as pd
import numpy as np

from operational_analysis.toolkits import met_data_processing
from operational_analysis.toolkits import filters
from operational_analysis.toolkits.power_curve import functions
from operational_analysis.toolkits import imputing
from operational_analysis.toolkits import timeseries

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
            - _scada_freq
            - reanalysis products ['merra2', 'erai', 'ncep2'] with columns ['time', 'u_ms', 'v_ms', 'windspeed_ms', 'rho_kgm-3']
            - scada with columns: ['time', 'id', 'wmet_wdspd_avg', 'wtur_W_avg', 'energy_kwh']
    """

    @logged_method_call
    def __init__(self, plant):
        """
        Initialize turbine long-term gross energy analysis with data and parameters.
        Args:
         plant(:obj:`PlantData object`): PlantData object from which TurbineLongTermGrossEnergy should draw data.
        """
        logger.info("Initializing TurbineLongTermGrossEnergy Object")
        
        self._plant = plant  # Set plant as attribute of analysis object
        self._turbs = self._plant._scada.df['id'].unique() # Store turbine names
        
        # Get start and end of POR days in SCADA
        self._por_start = format(plant._scada.df.index.min(), '%Y-%m-%d')
        self._por_end = format(plant._scada.df.index.max(), '%Y-%m-%d')
        self._full_por = pd.date_range(self._por_start, self._por_end, freq = 'D')
        
        # Define several dictionaries and data frames to be populated within this method
        self._scada_dict = {}
        self._daily_reanal_dict = {}
        self._model_dict = {}
        self._model_results = {}
        self._turb_lt_gross = {}
        self._scada_daily_valid = pd.DataFrame()
              
        # Dictionary to convert time interval indicator into minutes
        self._time_conversion = {'10T': 10.,
                                 '5T': 5.,
                                 '1H': 60.}
        
        # Set number of 'valid' counts required when summing data to daily values
        self._num_valid_daily = 60. / self._time_conversion[self._plant._scada_freq] * 24
        

        
        # Initially sort the different turbine data into dictionary entries
        logger.info("Processing SCADA data into dictionaries by turbine (this can take a while)")
        self.sort_scada_by_turbine()
        

    @logged_method_call
    def run(self, reanal_subset = ['erai', 'ncep2', 'merra2'], max_power_filter = 0.85, 
            wind_bin_thresh = 2, correction_threshold = 0.90, enable_plotting = False,
            plot_dir = None):
        """
        Perform pre-processing of data into an internal representation for which the analysis can run more quickly.
        
        Args:
            reanal_subset(:obj:`list`): Which reanalysis products to use for long-term correction
            max_power_filter(:obj:`float`): Maximum power threshold (fraction) to which the bin filter 
                                            should be applied (default 0.85)
            wind_bin_thresh(:obj:`float`): The filter threshold for each bin (default is 2 m/s)
            correction_threshold(:obj:`float`): The threshold (fraction) above which daily scada energy data
                                                should be corrected (default is 0.90)
            enable_plotting(:obj:`boolean`): Indicate whether to output plots
            plot_dir(:obj:`string`): Location to save figures
            
        Returns:
            (None)
        """
        
        # Assign parameters as object attributes
        self._reanal = reanal_subset # Reanalysis data to consider in fitting
        self._max_power_filter = max_power_filter 
        self._wind_bin_thresh = wind_bin_thresh
        self._correction_threshold = correction_threshold
        
        logger.info("Filtering turbine data")
        self.filter_turbine_data() # Filter turbine data
        
        if enable_plotting:
            logger.info("Plotting filtered power curves")
            self.plot_filtered_power_curves(plot_dir) # Setup daily reanalysis products
        
        logger.info("Processing reanalysis data to daily averages")
        self.setup_daily_reanalysis_data() # Setup daily reanalysis products
        
        logger.info("Processing scada data to daily sums")
        self.filter_sum_impute_scada() # Setup daily reanalysis products
        
        logger.info("Setting up daily data for model fitting")
        self.setup_model_dict() # Setup daily data to be fit using the GAM
        
        logger.info("Fitting model data")
        self.fit_model() # Fit daily turbine energy to atmospheric data
        
        logger.info("Applying fitting results to calculate long-term gross energy")
        self.apply_model_to_lt() # Apply fitting result to long-term reanalysis data
        
        if enable_plotting:
            logger.info("Plotting daily fitted power curves")
            self.plot_daily_fitting_result(plot_dir) # Setup daily reanalysis products
     
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
            dic[t] = df.loc[df['id'] == t, ['wmet_wdspd_avg', 'wtur_W_avg', 'energy_kwh']]
            dic[t].sort_index(inplace=True)            
        
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
            turb_capac = dic[t].wtur_W_avg.max()
            max_bin = self._max_power_filter * turb_capac # Set maximum range for using bin-filter
            
            dic[t].dropna(subset = ['wmet_wdspd_avg', 'energy_kwh'], inplace = True) # Drop any data where scada wind speed or energy is NaN
            #print(dic[t].loc[np.isnan(dic[t]['wmet_wdspd_avg'])])
            
            # Flag turbine energy data less than zero
            dic[t].loc[:,'flag_neg'] = filters.range_flag(dic[t].loc[:, 'wtur_W_avg'], below = 0, above = turb_capac)
            # Apply range filter
            dic[t].loc[:,'flag_range'] = filters.range_flag(dic[t].loc[:, 'wmet_wdspd_avg'], below = 0, above = 40)
            # Apply frozen/unresponsive sensor filter
            dic[t].loc[:,'flag_frozen'] = filters.unresponsive_flag(dic[t].loc[:, 'wmet_wdspd_avg'], threshold = 3)
            # Apply window range filter
            dic[t].loc[:,'flag_window'] = filters.window_range_flag(window_col = dic[t].loc[:, 'wmet_wdspd_avg'], 
                                                                    window_start = 5., 
                                                                    window_end = 40, 
                                                                    value_col = dic[t].loc[:, 'wtur_W_avg'], 
                                                                    value_min =  0.02*turb_capac,
                                                                    value_max =  1.2*turb_capac) 
            # Apply bin-based filter
            dic[t].loc[:,'flag_bin'] = filters.bin_filter(bin_col = dic[t].loc[:, 'wtur_W_avg'], 
                                                          value_col = dic[t].loc[:, 'wmet_wdspd_avg'], 
                                                          bin_width = 0.06* turb_capac,
                                                          threshold = self._wind_bin_thresh, # wind bin thresh; 2.5 or so 
                                                          center_type = 'median', 
                                                          bin_min = 0.01* turb_capac,
                                                          bin_max = max_bin, 
                                                          threshold_type = 'scalar', 
                                                          direction = 'all')
            # Create a 'final' flag which is true if any of the previous flags are true
            dic[t].loc[:, 'flag_final'] = (dic[t].loc[:, 'flag_range']) | \
                                          (dic[t].loc[:, 'flag_window']) | \
                                          (dic[t].loc[:, 'flag_bin']) | \
                                          (dic[t].loc[:, 'flag_frozen'])
                       
            # Set negative turbine data to zero
            dic[t].loc[dic[t]['flag_neg'], 'wtur_W_avg'] = 0

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
            plt.scatter(dic[t].wmet_wdspd_avg, dic[t].wtur_W_avg, s=1, label = 'Raw') # Plot all data
            plt.scatter(filt_df['wmet_wdspd_avg'], filt_df['wtur_W_avg'], s=1, label = 'Flagged') # Plot flagged data
            plt.xlim(0, 30)
            plt.xlabel('Wind speed (m/s)')
            plt.ylabel('Power (W)')
            plt.title('Filtered power curve for Turbine %s' %t)
            plt.legend(loc = 'lower right')
            plt.savefig('%s/filtered_power_curve_%s.png' %(save_folder, t,), dpi = 200) # Save file
            
            # Output figure to terminal if desired
            if output_to_terminal:
                plt.show()
            
            plt.close()
    
    def plot_daily_fitting_result(self, save_folder, output_to_terminal = False):
        """
        Plot the raw and flagged power curve data and save to file.
        
        Args:
            save_folder('obj':'str'): The pathname to where figure files should be saved
            output_to_terminal('obj':'boolean'): Indicate whether or not to output figures to terminal
            
        Returns:
            (None)
        """
        import matplotlib.pyplot as plt
        
        mod_input = self._model_dict
        
        # Loop through turbines
        for t in self._turbs:
            for r in self._reanal:
                df = mod_input[(t, r)]
                daily_reanal = self._daily_reanal_dict[r]
                ws_daily = daily_reanal['windspeed_ms']
                
                df_imputed = df.loc[df['energy_kwh_corr'] != df['energy_imputed']]

            
                plt.figure(figsize = (6,5))
                plt.plot(ws_daily, self._turb_lt_gross[r].loc[:, t], 'r.', alpha = 0.1, label = 'Modeled')
                plt.plot(df['windspeed_ms'], df['energy_imputed'], '.', label= 'Input')
                plt.plot(df_imputed['windspeed_ms'], df_imputed['energy_imputed'], '.', label= 'Imputed')
                plt.xlabel('Wind speed (m/s)')
                plt.ylabel('Daily Energy (kWh)')
                plt.title('Daily SCADA Energy Fitting, Turbine %s' %t)
                plt.legend(loc = 'lower right')
                plt.savefig('%s/daily_power_curve_%s_%s.png' %(save_folder, r, t,), dpi = 200) # Save file
            
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
            reanal['u_ms'], reanal['v_ms'] = met_data_processing.compute_u_v_components(reanal['windspeed_ms'], reanal['winddirection_deg'])
            df_daily = reanal.resample('D')['u_ms', 'v_ms', 'windspeed_ms', 'rho_kgm-3'].mean() # Get daily means
            
            # Recalculate daily average wind direction
            df_daily['winddirection_deg'] = met_data_processing.compute_wind_direction(u = df_daily['u_ms'],
                                                                                       v = df_daily['v_ms'])
            dic[r] = df_daily # Assign result to dictionary
            
    def filter_sum_impute_scada(self):
        """
        Filter SCADA data for unflagged data, gather SCADA energy data into daily sums, and correct daily summed
        energy based on amount of missing data and a threshold limit. Finally impute missing data for each turbine
        based on reported energy data from other highly correlated turbines.
        threshold
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        scada = self._scada_dict
        
        num_thres = self._correction_threshold * self._num_valid_daily # Number of permissible reported timesteps
        
        # Loop through turbines
        for t in self._turbs:
            scada_filt = scada[t].loc[scada[t]['flag_final'] == False] # Filter for valid data
            scada_daily = scada_filt.resample('D')['energy_kwh'].sum().to_frame() # Calculate daily energy sum
            scada_daily['data_count'] = scada_filt.resample('D')['energy_kwh'].count() # Count number of entries in sum 
            scada_daily['perc_nan'] = scada_filt.resample('D')['energy_kwh'].apply(timeseries.percent_nan) # Count number of entries in sum 
            
            
            # Correct energy for missing data
            scada_daily['energy_kwh_corr'] = scada_daily ['energy_kwh'] * self._num_valid_daily/scada_daily['data_count']
            
            # Discard daily sums if less than 140 data counts (90% reported data) 
            scada_daily = scada_daily.loc[scada_daily['data_count'] >= num_thres]
            
            # Create temporary data frame that is gap filled and to be used for imputing
            temp_df = pd.DataFrame(index = self._full_por)
            temp_df['energy_kwh_corr'] = scada_daily['energy_kwh_corr'] # Corrected energy data
            temp_df['perc_nan'] = scada_daily['perc_nan'] # Corrected energy data
            temp_df['id'] = np.repeat(t, temp_df.shape[0]) # Index
            temp_df['day'] = temp_df.index # Day
            
            # Append turbine data into single data frame for imputing
            self._scada_daily_valid = self._scada_daily_valid.append(temp_df) # 
        
        # Reset index after all turbines has been combined
        self._scada_daily_valid.reset_index(inplace = True)
        
        # Impute missing days for each turbine
        self._scada_daily_valid['energy_imputed'] = imputing.impute_all_assets_by_correlation(self._scada_daily_valid, 
                                                                                              input_col = 'energy_kwh_corr',
                                                                                              ref_col = 'energy_kwh_corr',
                                                                                              align_col = 'day',
                                                                                              id_col = 'id')
        
        # Drop data that could not be imputed
        self._scada_daily_valid.dropna(subset = ['energy_imputed'], inplace = True)
  
    def setup_model_dict(self):
        """
        Setup daily atmospheric variable averages and daily turbine energy sums for use
        in the GAM model
        
        Args:
            (None)
            
        Returns:
            (None)
        """

        reanal = self._daily_reanal_dict
        mod = self._model_dict
        
        # Store the valid data to be used for fitting in a separate dictionary                          
        for t in self._turbs:    
            daily_valid = self._scada_daily_valid.loc[self._scada_daily_valid['id'] == t]
            daily_valid.set_index('day', inplace = True)
            for r in self._reanal: # Loop through reanalysis products
                mod[t, r] = daily_valid.join(reanal[r])
                mod[t, r].dropna(subset = ['energy_imputed', 'windspeed_ms'], inplace = True) # Drop any remaining NaNs (e.g., reanalysis does not cover fulll POR)
            
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
            for r in self._reanal: # Loop through reanalysis products
                df = mod_dict[t, r]
                # Consider wind speed, wind direction, and air density as features
                mod_results[t, r] = functions.gam_3param(windspeed_column = df['windspeed_ms'],
                                                         winddir_column = df['winddirection_deg'],
                                                         airdens_column = df['rho_kgm-3'],
                                                         power_column=df['energy_imputed']) 
        
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
                turb_gross[r].loc[turb_gross[r][t] < 0, t] = 0
            #turb_gross[r][turb_gross[r] < 0] = 0 # Set any predicted negative energy to zero
            turb_annual = turb_gross[r].resample('AS').sum() # Calculate annual sums of energy from long-term estimate4
            self._summary_results.loc[r, :] = turb_annual.mean(axis = 0) # Store mean annual gross energy in data frame
            self._plant_gross = self._summary_results.sum(axis=1).mean()