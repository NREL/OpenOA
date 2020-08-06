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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from operational_analysis.toolkits import met_data_processing as mt
from operational_analysis.toolkits import timeseries as tm
from operational_analysis.toolkits.machine_learning_setup import MachineLearningSetup
from operational_analysis.toolkits import unit_conversion as un
from operational_analysis.toolkits import filters
from operational_analysis.types import timeseries_table
from operational_analysis import logged_method_call
from operational_analysis import logging

logger = logging.getLogger(__name__)

class MonteCarloAEP(object):
    """
    A serial (Pandas-driven) implementation of the benchmark PRUF operational
    analysis implementation. This module collects standard processing and
    analysis methods for estimating plant level operational AEP and uncertainty.
    
    The preprocessing should run in this order:

        1. Process revenue meter energy - creates monthly/daily data frame, gets revenue meter on monthly/daily basis, and adds
           data flag

        2. Process loss estimates - add monthly/daily curtailment and availabilty losses to monthly/daily data frame
        
        3. Process reanalysis data - add monthly/daily density-corrected wind speeds, temperature (if used) and wind direction (if used)
           from several reanalysis products to the monthly data frame
        
        4. Set up Monte Carlo - create the necessary Monte Carlo inputs to the OA process
        
        5. Run AEP Monte Carlo - run the OA process iteratively to get distribution of AEP results

    The end result is a distribution of AEP results which we use to assess expected AEP and associated uncertainty
    """

    @logged_method_call
    def __init__(self, plant, reanal_products=["merra2", "ncep2", "erai"], uncertainty_meter=0.005, uncertainty_losses=0.05,
                 uncertainty_windiness=(10, 20), uncertainty_loss_max=(10, 20),
                 uncertainty_nan_energy=0.01, time_resolution = 'M', reg_model = 'lin', ml_setup_kwargs={},
                 reg_temperature = False, reg_winddirection = False):
        """
        Initialize APE_MC analysis with data and parameters.
        
        Args:
         plant(:obj:`PlantData object`): PlantData object from which PlantAnalysis should draw data.
         reanal_products(obj:`list`) : List of reanalysis products to use for Monte Carlo sampling. Defaults to ["merra2", "ncep2", "erai"].
         uncertainty_meter(:obj:`float`): uncertainty on revenue meter data
         uncertainty_losses(:obj:`float`): uncertainty on long-term losses
         uncertainty_windiness(:obj:`tuple`): number of years to use for the windiness correction
         uncertainty_loss_max(:obj:`tuple`): threshold for the combined availabilty and curtailment monthly loss threshold
         uncertainty_nan_energy(:obj:`float`): threshold to flag days/months based on NaNs
         time_resolution(:obj:`string`): whether to perform the AEP calculation at monthly ('M') or daily ('D') time resolution
         reg_model(:obj:`string`): which model to use for the regression ('lin' for linear, 'gam', 'gbm', 'etr'). At monthly time resolution only linear regression is allowed because of the reduced number of data points.
         ml_setup_kwargs(:obj:`kwargs`): keyword arguments to MachineLearningSetup class
         reg_temperature(:obj:`bool`): whether to include temperature (True) or not (False) as regression input
         reg_winddirection(:obj:`bool`): whether to include wind direction (True) or not (False) as regression input
        """
        logger.info("Initializing MonteCarloAEP Analysis Object")

        self._aggregate = timeseries_table.TimeseriesTable.factory(plant._engine)
        self._plant = plant  # defined at runtime
        self._reanal_products = reanal_products # set of reanalysis products to use
    
        # Memo dictionaries help speed up computation
        self.outlier_filtering = {}  # Combinations of outlier filter results
        self.long_term_sampling = {}  # Combinations of long-term reanalysis data sampling
        self.long_term_sampling_sd = {}  # Combinations of long-term reanalysis data sampling stdev of wind speed
        
        # Define relevant uncertainties, data ranges and max thresholds to be applied in Monte Carlo sampling
        self.uncertainty_meter = np.float64(uncertainty_meter)
        self.uncertainty_losses = np.float64(uncertainty_losses)
        self.uncertainty_windiness = np.array(uncertainty_windiness, dtype=np.float64)
        self.uncertainty_loss_max = np.array(uncertainty_loss_max, dtype=np.float64)
        self.uncertainty_nan_energy = np.float64(uncertainty_nan_energy)
        
        # Check that selected time resolution is allowed
        if time_resolution not in ['M','D']:
            raise ValueError("time_res has to either be M (monthly, default) or D (daily)")
        self.time_resolution = time_resolution
        self._resample_freq = {"M": 'MS', "D": 'D'}[self.time_resolution]
        self._hours_in_res = {"M": 30*24, "D": 1*24}[self.time_resolution]
        self._calendar_samples = {"M": 12, "D": 365}[self.time_resolution]
        self.num_days_lt = (31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

        # Check that choices for regression inputs are allowed
        if reg_temperature not in [True, False]:
            raise ValueError("reg_temperature has to either be True (if temperature is considered in the regression), or False (if temperature is omitted")
        if reg_winddirection not in [True, False]:
            raise ValueError("reg_winddirection has to either be True (if wind direction is considered in the regression), or False (if wind direction is omitted")
        self.reg_winddirection = reg_winddirection
        self.reg_temperature = reg_temperature

        # Build list of regression variables
        self._rean_vars = []
        if self.reg_temperature:
            self._rean_vars += ["temperature_K"]
        if self.reg_winddirection:
            self._rean_vars += ["u_ms", "v_ms"]
        
        # Check that selected regression model is allowed
        if reg_model not in ['lin', 'gbm','etr','gam']:
            raise ValueError("reg_model has to either be lin (Linear regression, default), gbm (Gradient boosting model), etr (Extra trees regressor) or gam (Generalized additive model)")
        self.reg_model = reg_model
        self.ml_setup_kwargs = ml_setup_kwargs
        
        # Monthly data can only use robust linear regression because of limited number of data
        if (time_resolution == 'M') & (reg_model != 'lin'):
            raise ValueError("For monthly time resolution, only linear regression is allowed!")

        # Run preprocessing step                                                                                                                                                                              
        self.calculate_aggregate_dataframe()

        # Store start and end of period of record
        self._start_por = self._aggregate.df.index.min()
        self._end_por = self._aggregate.df.index.max()
        
        # Create a data frame to store monthly/daily reanalysis data over plant period of record
        self._reanalysis_por = self._aggregate.df.loc[(self._aggregate.df.index >= self._start_por) & \
                                                    (self._aggregate.df.index <= self._end_por)]
        
        self._reanalysis_por_avg = self.groupby_time_res(self._reanalysis_por)
            
    @logged_method_call
    def run(self, num_sim, reanal_subset=None):
        """
        Perform pre-processing of data into an internal representation for which the analysis can run more quickly.
        
        Args:
            reanal_subset(:obj:`list`): list of str data indicating which reanalysis products to use in OA
            num_sim(:obj:`int`): number of simulations to perform        
         
        Returns:
            None
        """
        self.num_sim = num_sim
        
        if reanal_subset is None:
            self.reanal_subset = self._reanal_products
        else:
            self.reanal_subset = reanal_subset
                
        # Write parameters of run to the log file
        logged_self_params = ["uncertainty_meter", "uncertainty_losses", "uncertainty_loss_max", "uncertainty_windiness",
                              "uncertainty_nan_energy", "num_sim", "reanal_subset"]
        logged_params = {name: getattr(self, name) for name in logged_self_params}
        logger.info("Running with parameters: {}".format(logged_params))

        # Start the computation
        self.calculate_long_term_losses()
        
        self.setup_monte_carlo_inputs()
        
        self.results = self.run_AEP_monte_carlo()

        # Log the completion of the run
        logger.info("Run completed")

    def plot_reanalysis_normalized_rolling_monthly_windspeed(self):
        """
        Make a plot of annual average wind speeds from reanalysis data to show general trends for each
        Highlight the period of record for plant data
        
        Returns:
            matplotlib.pyplot object
        """
        import matplotlib.pyplot as plt
        project = self._plant

        # Define parameters needed for plot
        min_val = 1  # Default parameter providing y-axis minimum for shaded plant POR region
        max_val = 1  # Default parameter providing y-axis maximum for shaded plant POR region
        por_start = self._aggregate.df.index[0]  # Start of plant POR
        por_end = self._aggregate.df.index[-1]  # End of plant POR

        plt.figure(figsize=(14, 6))
        for key in self._reanal_products:
            rean_df = project._reanalysis._product[key].df  # Set reanalysis product
            ann_mo_ws = rean_df.resample('MS')['ws_dens_corr'].mean().to_frame()  # Take monthly average wind speed
            ann_roll = ann_mo_ws.rolling(12).mean()  # Calculate rolling 12-month average
            ann_roll_norm = ann_roll['ws_dens_corr'] / ann_roll[
                'ws_dens_corr'].mean()  # Normalize rolling 12-month average

            # Update min_val and max_val depending on range of data
            if ann_roll_norm.min() < min_val:
                min_val = ann_roll_norm.min()
            if ann_roll_norm.max() > max_val:
                max_val = ann_roll_norm.max()

            # Plot wind speed
            plt.plot(ann_roll_norm, label=key)

        # Plot dotted line at y=1 (i.e. average wind speed)
        plt.plot((ann_roll.index[0], ann_roll.index[-1]), (1, 1), 'k--')

        # Fill in plant POR region
        plt.fill_between([por_start, por_end], [min_val, min_val], [max_val, max_val], alpha=0.1, label='Plant POR')

        # Final touches to plot
        plt.xlabel('Year')
        plt.ylabel('Normalized wind speed')
        plt.legend()
        plt.tight_layout()
        return plt

    def plot_reanalysis_gross_energy_data(self, outlier_thres):
        """
        Make a plot of normalized 30-day gross energy vs wind speed for each reanalysis product, include R2 measure
        
        Args:
            outlier_thres (float): outlier threshold (typical range of 1 to 4) which adjusts outlier sensitivity detection
        
        Returns:
            matplotlib.pyplot object
        """
        import matplotlib.pyplot as plt
        valid_aggregate = self._aggregate.df
        plt.figure(figsize=(9, 9))

        # Loop through each reanalysis product and make a scatterplot of monthly wind speed vs plant energy
        for p in np.arange(0, len(list(self._reanal_products))):
            col_name = self._reanal_products[p]  # Reanalysis column in monthly data frame

            x = sm.add_constant(valid_aggregate[col_name])  # Define 'x'-values (constant needed for regression function)
            if self.time_resolution == 'M':
                y = valid_aggregate['gross_energy_gwh'] * 30 / valid_aggregate[
                    'num_days_expected']  # Normalize energy data to 30-days
            elif self.time_resolution == 'D':
                y = valid_aggregate['gross_energy_gwh']
                
            rlm = sm.RLM(y, x, M=sm.robust.norms.HuberT(
                t=outlier_thres))  # Robust linear regression with HuberT algorithm (threshold equal to 2)
            rlm_results = rlm.fit()

            r2 = np.corrcoef(x.loc[rlm_results.weights == 1, col_name], y[rlm_results.weights == 1])[
                0, 1]  # Get R2 from valid data

            # Plot results
            plt.subplot(2, 2, p + 1)
            plt.plot(x.loc[rlm_results.weights != 1, col_name], y[rlm_results.weights != 1], 'rx', label='Outlier')
            plt.plot(x.loc[rlm_results.weights == 1, col_name], y[rlm_results.weights == 1], '.', label='Valid data')
            plt.title(col_name + ', R2=' + str(np.round(r2, 3)))
            plt.xlabel('Wind speed (m/s)')
            if self.time_resolution == 'M':
                plt.ylabel('30-day normalized gross energy (GWh)')
            elif self.time_resolution == 'D':
                plt.ylabel('Daily gross energy (GWh)')
        plt.tight_layout()
        return plt

    def plot_result_aep_distributions(self):
        """
        Plot a distribution of AEP values from the Monte-Carlo OA method
        
        Returns:
            matplotlib.pyplot object
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 12))

        sim_results = self.results

        ax = fig.add_subplot(2, 2, 1)
        ax.hist(sim_results['aep_GWh'], 40, density=1)
        ax.text(0.05, 0.9, 'AEP mean = ' + str(np.round(sim_results['aep_GWh'].mean(), 1)) + ' GWh/yr',
                transform=ax.transAxes)
        ax.text(0.05, 0.8, 'AEP unc = ' + str(
            np.round(sim_results['aep_GWh'].std() / sim_results['aep_GWh'].mean() * 100, 1)) + "%",
                transform=ax.transAxes)
        plt.xlabel('AEP (GWh/yr)')

        ax = fig.add_subplot(2, 2, 2)
        ax.hist(sim_results['avail_pct'] * 100, 40, density=1)
        ax.text(0.05, 0.9, 'Mean = ' + str(np.round((sim_results['avail_pct'].mean()) * 100, 1)) + ' %',
                transform=ax.transAxes)
        plt.xlabel('Availability Loss (%)')

        ax = fig.add_subplot(2, 2, 3)
        ax.hist(sim_results['curt_pct'] * 100, 40, density=1)
        ax.text(0.05, 0.9, 'Mean: ' + str(np.round((sim_results['curt_pct'].mean()) * 100, 2)) + ' %',
                transform=ax.transAxes)
        plt.xlabel('Curtailment Loss (%)')
        plt.tight_layout()
        return plt

    def plot_aep_boxplot(self, param, lab):
        """                                                                                                                                                                                        
        Plot box plots of AEP results sliced by a specified Monte Carlo parameter                                                                                                                  
        
        Args:                                                                                                                                                                                      
           param(:obj:`list`): The Monte Carlo parameter on which to split the AEP results
           lab(:obj:`str`): The name to use for the parameter when producing the figure
        
        Returns:                                                                                                                                                                                   
            (none)                                                                                                                                                                               
        """

        import matplotlib.pyplot as plt
        sim_results = self.results

        tmp_df=pd.DataFrame(data={'aep': sim_results.aep_GWh, 'param': param})
        tmp_df.boxplot(column='aep',by='param',figsize=(8,6))
        plt.ylabel('AEP (GWh/yr)')
        plt.xlabel(lab)
        plt.title('AEP estimates by %s' % lab)
        plt.suptitle("")
        plt.tight_layout()
        return plt

    def plot_aggregate_plant_data_timeseries(self):
        """
        Plot timeseries of monthly/daily gross energy, availability and curtailment
        
        Returns:
            matplotlib.pyplot object
        """
        import matplotlib.pyplot as plt
        valid_aggregate = self._aggregate.df

        plt.figure(figsize=(12, 9))

        # Gross energy
        plt.subplot(2, 1, 1)
        plt.plot(valid_aggregate.gross_energy_gwh, '.-')
        plt.grid('on')
        plt.xlabel('Year')
        plt.ylabel('Gross energy (GWh)')

        # Availability and curtailment
        plt.subplot(2, 1, 2)
        plt.plot(valid_aggregate.availability_pct * 100, '.-', label='Availability')
        plt.plot(valid_aggregate.curtailment_pct * 100, '.-', label='Curtailment')
        plt.grid('on')
        plt.xlabel('Year')
        plt.ylabel('Loss (%)')
        plt.legend()
        
        plt.tight_layout()
        return plt

    @logged_method_call
    def groupby_time_res(self, df):
        """
        Group pandas dataframe based on the time resolution chosen in the calculation.
        
        Args:
            df(:obj:`dataframe`): dataframe that needs to be grouped based on time resolution used
         
        Returns:
            None
        """
     
        if self.time_resolution == 'M':
            df_grouped = df.groupby(df.index.month).mean()
        elif self.time_resolution == 'D':
            df_grouped = df.groupby([(df.index.month),(df.index.day)]).mean()
         
        return df_grouped

    @logged_method_call
    def calculate_aggregate_dataframe(self):
        """
        Perform pre-processing of the plant data to produce a monthly/daily data frame to be used in AEP analysis.
        
        Args:
            (None)
        
        Returns:
            (None)
        """

        # Average to monthly/daily, quantify NaN data
        self.process_revenue_meter_energy()

        # Average to monthly/daily, quantify NaN data, merge with revenue meter energy data
        self.process_loss_estimates()

        # Density correct wind speeds, process temperature and wind direction, average to monthly/daily
        self.process_reanalysis_data()

        # Remove first and last reporting months if only partial month reported
        # (only for monthly time resolution calculations)
        if self.time_resolution == 'M':
           self.trim_monthly_df()

        # Drop any data that have NaN gross energy values or NaN reanalysis data
        self._aggregate.df = self._aggregate.df.dropna(subset=['gross_energy_gwh']
                                 +[product for product in self._reanal_products])
                
    @logged_method_call
    def process_revenue_meter_energy(self):
        """
        Initial creation of monthly data frame:

            1. Populate monthly/daily data frame with energy data summed from 10-min QC'd data
            2. For each monthly/daily value, find percentage of NaN data used in creating it and flag if percentage is
               greater than 0
            
        Args:
            (None)

        Returns:
            (None)
        """
        df = getattr(self._plant, 'meter').df  # Get the meter data frame

        # Create the monthly/daily data frame by summing meter energy
        self._aggregate.df = (df.resample(self._resample_freq)['energy_kwh'].sum() / 1e6).to_frame()  # Get monthly energy values in GWh
        self._aggregate.df.rename(columns={"energy_kwh": "energy_gwh"}, inplace=True)  # Rename kWh to MWh

        # Determine how much 10-min data was missing for each year-month/daily energy value. Flag accordigly if any is missing
        self._aggregate.df['energy_nan_perc'] = df.resample(self._resample_freq)['energy_kwh'].apply(
            tm.percent_nan)  # Get percentage of meter data that were NaN when summing to monthly/daily

        if self.time_resolution == 'M':
            # Create a column with expected number of days per month (to be used when normalizing to 30-days for regression)
            days_per_month = (pd.Series(self._aggregate.df.index)).dt.daysinmonth
            days_per_month.index = self._aggregate.df.index
            self._aggregate.df['num_days_expected'] = days_per_month
    
            # Get actual number of days per month in the raw data
            # (used when trimming beginning and end of monthly data frame)
            # If meter data has higher resolution than monthly
            if (self._plant._meter_freq == '1MS') | (self._plant._meter_freq == '1M'):
                self._aggregate.df['num_days_actual'] = self._aggregate.df['num_days_expected']            
            else:
                self._aggregate.df['num_days_actual'] = df.resample('MS')['energy_kwh'].apply(tm.num_days)
                

    @logged_method_call
    def process_loss_estimates(self):
        """
        Append availability and curtailment losses to monthly data frame
        
        Args:
            (None)
        
        Returns:
            (None)
        """
        df = getattr(self._plant, 'curtail').df

        curt_aggregate = np.divide(df.resample(self._resample_freq)[['availability_kwh', 'curtailment_kwh']].sum(),
                                 1e6)  # Get sum of avail and curt losses in GWh
   
        curt_aggregate.rename(columns={'availability_kwh': 'availability_gwh', 'curtailment_kwh': 'curtailment_gwh'},
                            inplace=True)  
        # Merge with revenue meter monthly/daily data
        self._aggregate.df = self._aggregate.df.join(curt_aggregate)

        # Add gross energy field
        self._aggregate.df['gross_energy_gwh'] = un.compute_gross_energy(self._aggregate.df['energy_gwh'],
                                                                       self._aggregate.df['availability_gwh'],
                                                                       self._aggregate.df['curtailment_gwh'], 'energy',
                                                                       'energy')

        # Calculate percentage-based losses
        self._aggregate.df['availability_pct'] = np.divide(self._aggregate.df['availability_gwh'],
                                                         self._aggregate.df['gross_energy_gwh'])
        self._aggregate.df['curtailment_pct'] = np.divide(self._aggregate.df['curtailment_gwh'],
                                                        self._aggregate.df['gross_energy_gwh'])

        self._aggregate.df['avail_nan_perc'] = df.resample(self._resample_freq)['availability_kwh'].apply(
            tm.percent_nan)  # Get percentage of 10-min meter data that were NaN when summing to monthly/daily
        self._aggregate.df['curt_nan_perc'] = df.resample(self._resample_freq)['curtailment_kwh'].apply(
            tm.percent_nan)  # Get percentage of 10-min meter data that were NaN when summing to monthly/daily

        self._aggregate.df['nan_flag'] = False  # Set flag to false by default
        self._aggregate.df.loc[(self._aggregate.df['energy_nan_perc'] > self.uncertainty_nan_energy) |
                             (self._aggregate.df['avail_nan_perc'] > self.uncertainty_nan_energy) |
                             (self._aggregate.df['curt_nan_perc'] > self.uncertainty_nan_energy), 'nan_flag'] \
            = True  # If more than 1% of data are NaN, set flag to True

        # By default, assume all reported losses are representative of long-term operational
        self._aggregate.df['availability_typical'] = True
        self._aggregate.df['curtailment_typical'] = True

        # By default, assume combined availability and curtailment losses are below the threshold to be considered valid
        self._aggregate.df['combined_loss_valid'] = True
        
    @logged_method_call
    def process_reanalysis_data(self):
        """
        Process reanalysis data for use in PRUF plant analysis:

            - calculate density-corrected wind speed and wind components
            - get monthly/daily average wind speeds and components
            - calculate monthly/daily average wind direction
            - calculate monthly/daily average temperature
            - append monthly/daily averages to monthly/daily energy data frame
        
        Args:
            (None)
        
        Returns:
            (None)
        """
            
        # Define empty data frame that spans past our period of interest
        self._reanalysis_aggregate = pd.DataFrame(index=pd.date_range(start='1997-01-01', end='2020-01-01',
                                                                    freq=self._resample_freq), dtype=float)

        # Now loop through the different reanalysis products, density-correct wind speeds, and take monthly averages
        for key in self._reanal_products:
            rean_df = self._plant._reanalysis._product[key].df
            rean_df['ws_dens_corr'] = mt.air_density_adjusted_wind_speed(rean_df['windspeed_ms'],
                                                                         rean_df['rho_kgm-3'])  # Density correct wind speeds
            self._reanalysis_aggregate[key] = rean_df.resample(self._resample_freq)['ws_dens_corr'].mean()  # .to_frame() # Get average wind speed by year-month
            
            
            if self.reg_winddirection | self.reg_temperature:
                namescol = [key + '_' + var for var in self._rean_vars]
                self._reanalysis_aggregate[namescol] = rean_df[self._rean_vars].resample(self._resample_freq).mean()

            if self.reg_winddirection: # if wind direction is considered as regression variable
                self._reanalysis_aggregate[key + '_wd'] = np.pi-(np.arctan2(-self._reanalysis_aggregate[key + '_u_ms'],self._reanalysis_aggregate[key + '_v_ms'])) # Calculate wind direction
        
        self._aggregate.df = self._aggregate.df.join(
                self._reanalysis_aggregate)  # Merge monthly reanalysis data to monthly energy data frame
                                       
    @logged_method_call
    def trim_monthly_df(self):
        """
        Remove first and/or last month of data if the raw data had an incomplete number of days
        
        Args:
            (None)
        
        Returns:
            (None)
        """
        for p in self._aggregate.df.index[[0, -1]]:  # Loop through 1st and last data entry
            if self._aggregate.df.loc[p, 'num_days_expected'] != self._aggregate.df.loc[p, 'num_days_actual']:
                self._aggregate.df.drop(p, inplace=True)  # Drop the row from data frame

    @logged_method_call
    def calculate_long_term_losses(self):
        """
        This function calculates long-term availability and curtailment losses based on the reported data,
        filtering for those data that are deemed representative of average plant performance
        
        Args:
            (None)
        
        Returns:
            (tuple):
                :obj:`float`: long-term annual availability loss expressed as fraction
                :obj:`float`: long-term annual curtailment loss expressed as fraction
        """
        df = self._aggregate.df
        
        days_year_lt = 365.25 # Number of days per long-term year (accounting for leap year every 4 years)

        # isolate availabilty and curtailment values that are representative of average plant performance
        avail_valid = df.loc[df['availability_typical'],'availability_pct'].to_frame()
        curt_valid = df.loc[df['curtailment_typical'],'curtailment_pct'].to_frame()
        
        # Now get average percentage losses by month or day
        avail_long_term = self.groupby_time_res(avail_valid)['availability_pct']
        curt_long_term = self.groupby_time_res(curt_valid)['curtailment_pct']
    
        # Ensure there are 12 or 365 data points in long-term average. If not, throw an exception:
        if (avail_long_term.shape[0] < self._calendar_samples):
                raise Exception('Not all calendar days/months represented in long-term availability calculation')
        if (curt_long_term.shape[0] < self._calendar_samples):
                raise Exception('Not all calendar days/months represented in long-term curtailment calculation')

        if self.time_resolution == 'M':
            # Weight monthly totals by number of days in month
            avail_long_term *= self.num_days_lt
            curt_long_term *= self.num_days_lt

        avail_annual = avail_long_term.sum()/days_year_lt
        curt_annual = curt_long_term.sum()/days_year_lt

        self.long_term_losses = (avail_annual, curt_annual)

    def setup_monte_carlo_inputs(self):
        """
        Create and populate the data frame defining the simulation parameters.
        This data frame is stored as self._inputs
        
        Args:
            (None)
            
        Returns:
            (None)
        """
    
        # Create extra long list of renanalysis product names to sample from
        reanal_list = list(np.repeat(self.reanal_subset, self.num_sim))
        
        inputs = {
                "reanalysis_product": np.asarray(random.sample(reanal_list, self.num_sim)),
                "metered_energy_fraction": np.random.normal(1, self.uncertainty_meter, self.num_sim),
                "loss_fraction": np.random.normal(1, self.uncertainty_losses, self.num_sim),
                "num_years_windiness": np.random.randint(self.uncertainty_windiness[0],self.uncertainty_windiness[1] + 1, self.num_sim),
                "loss_threshold":np.random.randint(self.uncertainty_loss_max[0], self.uncertainty_loss_max[1] + 1, self.num_sim) / 100.,
            }
        
        self._inputs = pd.DataFrame(inputs)

    @logged_method_call
    def filter_outliers(self, n):
        """
        This function filters outliers based on a combination of range filter, unresponsive sensor filter, 
        and window filter.
        We use a memoized funciton to store the regression data in a dictionary for each combination as it
        comes up in the Monte Carlo simulation. This saves significant computational time in not having to run
        robust linear regression for each Monte Carlo iteration
        
        Args:
            n(:obj:`float`): Monte Carlo iteration
        
        Returns:
            :obj:`pandas.DataFrame`: Filtered monthly/daily data ready for linear regression
        """
        
        reanal = self._run.reanalysis_product
        
        # Check if valid data has already been calculated and stored. If so, just return it
        if (reanal, self._run.loss_threshold) in self.outlier_filtering:
            valid_data = self.outlier_filtering[(reanal, self._run.loss_threshold)]
            return valid_data

        # If valid data hasn't yet been stored in dictionary, determine the valid data
        df = self._aggregate.df
                
        # First set of filters checking combined losses and if the Nan data flag was on
        df_sub = df.loc[
            ((df['availability_pct'] + df['curtailment_pct']) < self._run.loss_threshold) & (df['nan_flag'] == False),:]
        
        # Set maximum range for using bin-filter, convert from MW to GWh
        plant_capac = self._plant._plant_capacity/1000 * self._hours_in_res

        # Apply range filter to wind speed
        df_sub = df_sub.assign(flag_range=filters.range_flag(df_sub[reanal], below = 0, above = 40))
        # Apply frozen/unresponsive sensor filter
        df_sub.loc[:,'flag_frozen'] = filters.unresponsive_flag(df_sub[reanal], threshold = 3)
        # Apply window range filter
        df_sub.loc[:,'flag_window'] = filters.window_range_flag(window_col = df_sub[reanal], 
                                                                    window_start = 5., 
                                                                    window_end = 40,
                                                                    value_col = df_sub['energy_gwh'], 
                                                                    value_min =  0.02*plant_capac,
                                                                    value_max =  1.2*plant_capac) 
        
        # Create a 'final' flag which is true if any of the previous flags are true
        df_sub.loc[:,'flag_final'] = (df_sub.loc[:, 'flag_range']) | (df_sub.loc[:, 'flag_frozen']) | \
                                          (df_sub.loc[:, 'flag_window']) 
                
        # Define valid data
        valid_data = df_sub.loc[df_sub.loc[:, 'flag_final'] == False, [reanal, 
                                                               'energy_gwh', 'availability_gwh',
                                                               'curtailment_gwh']]
        if self.reg_winddirection:
            valid_data_to_add = df_sub.loc[df_sub.loc[:, 'flag_final'] == False, [reanal + '_wd',
                                                               reanal + '_u_ms', reanal + '_v_ms']]
            valid_data = pd.concat([valid_data, valid_data_to_add], axis=1)
            
        if self.reg_temperature:
            valid_data_to_add = df_sub.loc[df_sub.loc[:, 'flag_final'] == False, [reanal + '_temperature_K']]
            valid_data = pd.concat([valid_data, valid_data_to_add], axis=1)
            
        if self.time_resolution == 'M':   
            valid_data_to_add = df_sub.loc[df_sub.loc[:, 'flag_final'] == False, ['num_days_expected']]
            valid_data = pd.concat([valid_data, valid_data_to_add], axis=1)     
                      
        # Update the dictionary
        self.outlier_filtering[(reanal, self._run.loss_threshold)] = valid_data

        # Return result
        return valid_data

    @logged_method_call
    def set_regression_data(self, n):
        """
        This will be called for each iteration of the Monte Carlo simulation and will do the following:
            
            1. Randomly sample monthly/daily revenue meter, availabilty, and curtailment data based on specified uncertainties
               and correlations
            2. Randomly choose one reanalysis product
            3. Calculate gross energy from randomzied energy data
            4. Normalize gross energy to 30-day months
            5. Filter results to remove months/days with NaN data and with combined losses that exceed the Monte Carlo
               sampled max threhold
            6. Return the wind speed and normalized gross energy to be used in the regression relationship
        
        Args:
            n(:obj:`int`): The Monte Carlo iteration number
        
        Returns:
            :obj:`pandas.Series`: Monte-Carlo sampled wind speeds and other variables (temperature, wind direction) if used in the regression
            :obj:`pandas.Series`: Monte-Carlo sampled normalized gross energy
        """
        # Get data to use in regression based on filtering result
        reg_data = self.filter_outliers(n)

        # Now monte carlo sample the data
        mc_energy = reg_data['energy_gwh'] * self._run.metered_energy_fraction # Create new Monte-Carlo sampled data frame and sample energy data
        mc_availability = reg_data['availability_gwh'] * self._run.loss_fraction # Calculate MC-generated availability
        mc_curtailment = reg_data['curtailment_gwh'] * self._run.loss_fraction # Calculate MC-generated curtailment

        # Calculate gorss energy and normalize to 30-days
        mc_gross_energy = mc_energy + mc_availability + mc_curtailment
        if self.time_resolution == 'M':
            num_days_expected = reg_data['num_days_expected']
            mc_gross_norm = mc_gross_energy * 30 / num_days_expected  # Normalize gross energy to 30-day months
        elif self.time_resolution == 'D':   
            mc_gross_norm = mc_gross_energy
            
        # Set reanalysis product
        reg_inputs = reg_data[self._run.reanalysis_product]  # Copy wind speed data to Monte Carlo data frame
        
        if self.reg_temperature: # if temperature is considered as regression variable
            mc_temperature = reg_data[self._run.reanalysis_product + "_temperature_K"]  # Copy temperature data to Monte Carlo data frame
            reg_inputs = pd.concat([reg_inputs,mc_temperature], axis = 1)
            
        if self.reg_winddirection: # if wind direction is considered as regression variable
            mc_wind_direction = reg_data[self._run.reanalysis_product + "_wd"]  # Copy wind direction data to Monte Carlo data frame
            reg_inputs = pd.concat([reg_inputs,np.sin(mc_wind_direction)], axis = 1)
            reg_inputs = pd.concat([reg_inputs,np.cos(mc_wind_direction)], axis = 1)
   
        reg_inputs = pd.concat([reg_inputs,mc_gross_norm], axis = 1)
        # Return values needed for regression
        return reg_inputs  # Return randomly sampled wind speed, wind direction, temperature and normalized gross energy

    @logged_method_call
    def run_regression(self, n):
        """
        Run robust linear regression between Monte-Carlo generated monthly/daily gross energy, 
        wind speed, temperature and wind direction (if used)
        
        Args:
            n(:obj:`int`): The Monte Carlo iteration number
        
        Returns:
            :obj:`?`: trained regression model
        """
        reg_data = self.set_regression_data(n)  # Get regression data
        
        # Randomly select 80% of the data to perform regression and incorporate some regression uncertainty
        reg_data = np.array(reg_data.sample(frac = 0.8))
        
        # Update Monte Carlo tracker fields
        self._mc_num_points[n] = np.shape(reg_data)[0]
        
        # Linear regression
        if self.reg_model == 'lin':
            reg = LinearRegression().fit(np.array(reg_data[:,0:-1]), reg_data[:,-1])
            predicted_y = reg.predict(np.array(reg_data[:,0:-1]))

            self._mc_slope[n,:] = reg.coef_
            self._mc_intercept[n] = np.float(reg.intercept_)
            
            self._r2_score[n] = r2_score(reg_data[:,-1], predicted_y)
            self._mse_score[n] = mean_squared_error(reg_data[:,-1], predicted_y)
            return reg
        # Machine learning models
        else: 
            ml = MachineLearningSetup(self.reg_model, **self.ml_setup_kwargs)
            ml.hyper_optimize(np.array(reg_data[:,0:-1]), reg_data[:,-1], n_iter_search = 5, report = False, cv = KFold(n_splits = 2))
            predicted_y = ml.random_search.predict(np.array(reg_data[:,0:-1]))
            
            self._r2_score[n] = r2_score(reg_data[:,-1], predicted_y)
            self._mse_score[n] = mean_squared_error(reg_data[:,-1], predicted_y)
            return ml.random_search

    @logged_method_call
    def run_AEP_monte_carlo(self):
        """
        Loop through OA process a number of times and return array of AEP results each time
        
        Returns:
            :obj:`numpy.ndarray` Array of AEP, long-term avail, long-term curtailment calculations
        """

        num_sim = self.num_sim

        # Initialize arrays to store metrics and results
        self._mc_num_points = np.empty(num_sim, dtype=np.float64)
        self._r2_score = np.empty(num_sim, dtype=np.float64)
        self._mse_score = np.empty(num_sim, dtype=np.float64)
        
        num_vars = 1
        if self.reg_winddirection:
            num_vars = num_vars + 2
        if self.reg_temperature:
            num_vars = num_vars + 1
                
        if self.reg_model == 'lin':
            self._mc_intercept = np.empty(num_sim, dtype=np.float64) 
            self._mc_slope = np.empty([num_sim,num_vars], dtype=np.float64)

        aep_GWh = np.empty(num_sim)
        avail_pct =  np.empty(num_sim)
        curt_pct =  np.empty(num_sim)
        lt_por_ratio =  np.empty(num_sim)

        # Loop through number of simulations, run regression each time, store AEP results
        for n in tqdm(np.arange(num_sim)):
            
            self._run = self._inputs.loc[n]
            
            # Run regression
            fitted_model = self.run_regression(n)
 
            # Get long-term regression inputs
            [reg_inputs_lt_noIAV, reg_lt_stdev] = self.sample_long_term_reanalysis()

            # Apply IAV to long-term regression inputs
            reg_inputs_lt = self.apply_IAV(reg_inputs_lt_noIAV, reg_lt_stdev)
            
            # Get long-term normalized gross energy by applying regression result to long-term monthly wind speeds
            inputs = np.array(reg_inputs_lt)
            if num_vars == 1:
                inputs = inputs.reshape(-1,1)
            gross_lt = fitted_model.predict(inputs)

            # Get POR gross energy by applying regression result to POR regression inputs                                                                        
            reg_inputs_por = [self._reanalysis_por_avg[self._run.reanalysis_product]]
            if self.reg_temperature:
                reg_inputs_por += [self._reanalysis_por_avg[self._run.reanalysis_product + '_temperature_K']]
            if self.reg_winddirection:
                reg_inputs_por += [np.sin(self._reanalysis_por_avg[self._run.reanalysis_product + '_wd'])]
                reg_inputs_por += [np.cos(self._reanalysis_por_avg[self._run.reanalysis_product + '_wd'])]
            gross_por = fitted_model.predict(np.array(pd.concat(reg_inputs_por, axis = 1)))

            if self.time_resolution == 'M': # Undo normalization to 30-day months
                gross_lt = gross_lt*self.num_days_lt/30 
                gross_por = gross_por*self.num_days_lt/30
            
            # Get long-term availability and curtailment losses by month
            [avail_lt_losses, curt_lt_losses] = self.sample_long_term_losses()  

            # Assign AEP, long-term availability, and long-term curtailment to output data frame
            aep_GWh[n] = gross_lt.sum() * (1 - avail_lt_losses)
            avail_pct[n] = avail_lt_losses
            curt_pct[n] = curt_lt_losses
            lt_por_ratio[n] = gross_lt.sum() / gross_por.sum()
            
        # Return final output
        sim_results = pd.DataFrame(index=np.arange(num_sim), data={'aep_GWh': aep_GWh,                                                                                                        
                                                                   'avail_pct': avail_pct,                                                                                                      
                                                                   'curt_pct': curt_pct,                                                                                                    
                                                                  'lt_por_ratio': lt_por_ratio})      
        return sim_results

    @logged_method_call
    def sample_long_term_reanalysis(self):
        """
        This function returns the windiness-corrected monthly wind speeds based on the Monte-Carlo generated sample of:
            
            1. The reanalysis product
            2. The number of years to use in the long-term correction
        
        Args:
           (None)

        Returns:
           :obj:`pandas.DataFrame`: the windiness-corrected or 'long-term' annualized monthly/daily wind speeds
           :obj:`pandas.DataFrame`: Long-term wind speed stdev, from which IAV will be derived
        """
        # Check if valid data has already been calculated and stored. If so, just return it
        if (self._run.reanalysis_product,self. _run.num_years_windiness) in self.long_term_sampling:
            long_term_reg_inputs = self.long_term_sampling[(self._run.reanalysis_product,self. _run.num_years_windiness)]
            ws_aggregate_sd = self.long_term_sampling_sd[(self._run.reanalysis_product,self. _run.num_years_windiness)]
            return [long_term_reg_inputs.copy(),ws_aggregate_sd.copy()]
        
        # Sample long-term wind speed values
        ws_df = self._reanalysis_aggregate[self._run.reanalysis_product].to_frame().dropna()  # Drop NA values from monthly/daily reanalysis data series
        ws_data = ws_df.tail(self. _run.num_years_windiness * self._calendar_samples)  # Get last 'x' years of data from reanalysis product
        ws_aggregate = self.groupby_time_res(ws_data)[self._run.reanalysis_product] # Get long-term annualized monthly/daily wind speeds
        ws_aggregate.reset_index(drop=True, inplace=True) # TODO: This rectifies a 0-indexing issue. Needs refactoring.
        long_term_reg_inputs = pd.DataFrame(ws_aggregate)
        # Wind speed stdev for later IAV calculation
        ws_aggregate_sd = ws_data.groupby(ws_data.index.month)[self._run.reanalysis_product].std() # Get long-term annualized stdev of daily wind speeds            
        
        #import pdb
        #pdb.set_trace()

        # Temperature and wind direction
        namescol = [self._run.reanalysis_product + '_' + var for var in self._rean_vars]
        # TODO: The result of this call has a multi-index, which is blown away with "reset_index" to match the index of mc_ws_aggregate. Needs refactoring.
        long_term_temp = self._reanalysis_aggregate[namescol].dropna().tail(self. _run.num_years_windiness * self._calendar_samples).apply(self.groupby_time_res).reset_index(drop=True)
        if self.reg_temperature:
            long_term_reg_inputs = pd.concat([long_term_reg_inputs, long_term_temp[self._run.reanalysis_product + '_temperature_K']], axis=1)
        if self.reg_winddirection:
            wd_aggregate = np.pi-np.arctan2(-long_term_temp[self._run.reanalysis_product + '_u_ms'],long_term_temp[self._run.reanalysis_product + '_v_ms']) # Calculate wind direction
            long_term_reg_inputs = pd.concat([long_term_reg_inputs, np.sin(wd_aggregate), np.cos(wd_aggregate)], axis=1)
        
        # Store result in dictionary
        self.long_term_sampling[(self._run.reanalysis_product, self. _run.num_years_windiness)] = long_term_reg_inputs
        self.long_term_sampling_sd[(self._run.reanalysis_product, self. _run.num_years_windiness)] = ws_aggregate_sd
        
        # Return result            
        return [long_term_reg_inputs.copy(),ws_aggregate_sd.copy()]

    @logged_method_call
    def apply_IAV(self, long_term_reg_inputs, ws_aggregate_sd_IAV):
        """
        This function calculates and apply IAV for the long-term wind speed data
        
        Args:
           long_term_reg_inputs(:obj:`pandas.DataFrame`): Long-term regression inputs
           ws_aggregate_sd(:obj:`pandas.DataFrame`): Long-term wind speed stdev, from which IAV will be derived
        Returns:
           :obj:`pandas.DataFrame`: the IAV-sampled monthly/daily regression inputs
        """
    
        if self.time_resolution == 'D':
            ws_aggregate_sd_IAV = np.repeat(ws_aggregate_sd_IAV, np.ceil(self.num_days_lt).astype(int))
        ws_aggregate_sd_IAV.reset_index(drop=True,inplace=True) # TODO: This rectifies a 0-indexing issue. Needs refactoring.
        iav = np.random.normal(1, ws_aggregate_sd_IAV/long_term_reg_inputs[self._run.reanalysis_product])        
        long_term_reg_inputs_IAV = long_term_reg_inputs.copy()
        long_term_reg_inputs_IAV[self._run.reanalysis_product] = long_term_reg_inputs_IAV[self._run.reanalysis_product] * iav
        return long_term_reg_inputs_IAV

    @logged_method_call
    def sample_long_term_losses(self):
        """
        This function calculates long-term availability and curtailment losses based on the Monte Carlo sampled
        historical availability and curtailment data
        
        Args:
            n(:obj:`integer`): The Monte Carlo iteration number
        
        Returns:
            :obj:`float`: annualized monthly/daily availability loss expressed as fraction
            :obj:`float`: annualized monthly/daily curtailment loss expressed as fraction
        """
        mc_avail = self.long_term_losses[0] * self._run.loss_fraction
        mc_curt = self.long_term_losses[1] * self._run.loss_fraction

        # Return availbilty and curtailment long-term monthly/daily data
        return mc_avail, mc_curt