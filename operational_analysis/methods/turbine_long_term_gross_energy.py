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

from operational_analysis.toolkits import met_data_processing
from operational_analysis.toolkits import timeseries as tm
from operational_analysis.toolkits import unit_conversion as un
from operational_analysis.toolkits import filters
from operational_analysis.types import timeseries_table
from operational_analysis.toolkits.power_curve import functions

from operational_analysis import logged_method_call
from operational_analysis import logging

logger = logging.getLogger(__name__)


class TurbineLongTermGrossEnergy(object):
    """
    
    """

    @logged_method_call
    def __init__(self, plant, max_power_filter = 0.85):
        """
        Initialize turbine long-term gross energy analysis with data and parameters.

        Args:
         plant(:obj:`PlantData object`): PlantData object from which PlantAnalysis should draw data.

        """
        logger.info("Initializing TurbineLongTermGrossEnergy Object")

        #self._monthly = timeseries_table.TimeseriesTable.factory(plant._engine)
        
        self._plant = plant  # defined at runtime
        self._turbs = self._plant._scada.df['id'].unique()
        
        self._max_power_filter = max_power_filter
        self._reanal = ['merra2', 'erai', 'ncep2']
        
        self._scada_dict = {}
        self._daily_reanal_dict = {}
        self._model_dict = {}
        self._model_results = {}
        self._turb_lt_gross = {}
        
        self.sort_scada_by_turbine()
        
        # Convert time interval indicator into mintues
        self._time_conversion = {'10T': 10.,
                                 '5T': 5.,
                                 '1H': 60.}
        
        self._num_valid_daily = 60. / self._time_conversion[self._plant._scada_freq] * 24

    @logged_method_call
    def run(self):
        """
        Perform pre-processing of data into an internal representation for which the analysis can run more quickly.

        :return: None
        
        
        """
        self.filter_turbine_data()
        self.setup_daily_reanalysis_data()
        self.setup_model_dict()
        self.fit_model()
        self.apply_model_to_lt()
     
        # Log the completion of the run
        logger.info("Run completed")
        
    def sort_scada_by_turbine(self):
        
        df = self._plant._scada.df
        dic = self._scada_dict
        
        for t in self._turbs:
            dic[t] = df.loc[df['id'] == t, ['time', 'windspeed_ms', 'power_kw', 'energy_kwh']]            
            dic[t].set_index('time', inplace = True)
            dic[t].sort_index(inplace = True)
        
    def filter_turbine_data(self):
        dic = self._scada_dict
        for t in self._turbs:
            max_bin = self._max_power_filter * dic[t].power_kw.max()
            dic[t].loc[:,'flag_range'] = filters.range_flag(dic[t].loc[:, 'windspeed_ms'], below = 0, above = 40)
            dic[t].loc[:,'flag_frozen'] = filters.unresponsive_flag(dic[t].loc[:, 'windspeed_ms'], threshold = 3)
            dic[t].loc[:,'flag_window'] = filters.window_range_flag(window_col = dic[t].loc[:, 'windspeed_ms'], 
                                                                    window_start = 5., 
                                                                    window_end = 40, 
                                                                    value_col = dic[t].loc[:, 'power_kw'], 
                                                                    value_min = 20., 
                                                                    value_max = 2000.)
            dic[t].loc[:,'flag_bin'] = filters.bin_filter(bin_col = dic[t].loc[:, 'power_kw'], 
                                                          value_col = dic[t].loc[:, 'windspeed_ms'], 
                                                          bin_width = 100, 
                                                          threshold = 2., 
                                                          center_type = 'median', 
                                                          bin_min = 20., 
                                                          bin_max = max_bin, 
                                                          threshold_type = 'scalar', 
                                                          direction = 'all')
            dic[t].loc[:, 'flag_final'] = (dic[t].loc[:, 'flag_range']) | \
                                          (dic[t].loc[:, 'flag_window']) | \
                                          (dic[t].loc[:, 'flag_bin']) | \
                                          (dic[t].loc[:, 'flag_frozen'])
            
    def plot_filtered_power_curves(self, save_folder, output_to_terminal = False):
        import matplotlib.pyplot as plt
        dic = self._scada_dict
        
        for t in self._turbs:
            filt_df = dic[t].loc[dic[t]['flag_final']]
            plt.figure(figsize = (6,5))
            plt.scatter(dic[t].windspeed_ms, dic[t].power_kw, s=1, label = 'Raw')
            plt.scatter(filt_df['windspeed_ms'], filt_df['power_kw'], s=1, label = 'Flagged')
            plt.xlim(0, 30)
            plt.xlabel('Wind speed (m/s)')
            plt.ylabel('Power (kW)')
            plt.title('Filtered power curve for Turbine %s' %t)
            plt.legend(loc = 'lower right')
            plt.savefig('%s/%s_filtered_pc.png' %(save_folder, t,), dpi = 200)
            
            
            if output_to_terminal:
                plt.show()
            
            plt.close()
    
    def setup_daily_reanalysis_data(self):
        '''
        Probably just create a dictionary of data frames for each reanalysis product?
        '''
        dic = self._daily_reanal_dict
        for r in self._reanal:
            reanal = self._plant._reanalysis._product[r].df
            df_daily = reanal.resample('D')['u_ms', 'v_ms', 'windspeed_ms', 'rho_kgm-3'].mean()
            df_daily['winddirection_deg'] = met_data_processing.compute_wind_direction(u = df_daily['u_ms'],
                                                                                       v = df_daily['v_ms'])
            dic[r] = df_daily
            
    def setup_model_dict(self):
        '''
        Get daily sum of scada energy, keep only records with full coverage, merge with reanalysis
        '''
        scada = self._scada_dict
        reanal = self._daily_reanal_dict
        mod = self._model_dict
        
        for t in self._turbs: # Loop throuh turbines
            scada_filt = scada[t].loc[scada[t]['flag_final'] == False]
            scada_daily = scada_filt.resample('D')['energy_kwh'].sum().to_frame()
            scada_daily['count'] = scada_filt.resample('D')['energy_kwh'].count()
            daily_valid = scada_daily.loc[scada_daily['count'] == self._num_valid_daily]
                                      
            for r in self._reanal: # Loop through reanalysis products
                 mod[t, r] = daily_valid.join(reanal[r])         
      
    def fit_model(self):
        '''
        Fit model to a 20-spline GAM for now (hyperoptimization to come)
        '''
        mod_dict = self._model_dict
        mod_results = self._model_results
        
        for t in self._turbs: # Loop throuh turbines
            for r in self._reanal: # Loop through reanalysis products
                logger.info("Fitting turbine %s and reanalysis product %s" %(t, r,))
                df = mod_dict[t, r]
                mod_results[t, r] = functions.gam(windspeed_column = df['windspeed_ms'], 
                                                  winddir_column = df['winddirection_deg'],
                                                  airdens_column = df['rho_kgm-3'],
                                                  power_column=df['energy_kwh'])
        
    def apply_model_to_lt(self):
        '''
        Fit model to a 20-spline GAM for now (hyperoptimization to come)
        '''
        turb_gross = self._turb_lt_gross
        mod_results = self._model_results
        
        self._summary_results = pd.DataFrame(index = self._reanal, columns = self._turbs)
        
        for r in self._reanal: # Loop throuh turbines
            daily_reanal = self._daily_reanal_dict[r]
            turb_gross[r] = pd.DataFrame(index = daily_reanal.index)
            X_long_term = daily_reanal[['windspeed_ms', 'winddirection_deg', 'rho_kgm-3']]
            for t in self._turbs: # Loop through reanalysis products
                turb_gross[r].loc[:, t] = mod_results[t, r](X_long_term)
            
            turb_annual = turb_gross[r].resample('AS').sum()
            self._summary_results.loc[r, :] = turb_annual.mean(axis = 0)        