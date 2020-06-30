import unittest
import numpy as np
import pandas as pd
from numpy import testing as nptest

from operational_analysis.methods import plant_analysis
from examples.project_ENGIE import Project_Engie

class TestPandasPrufPlantAnalysis(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)

        # Set up data to use for testing (ENGIE example plant)
        self.project = Project_Engie('./examples/data/la_haute_borne')
        self.project.prepare()

    # Test inputs to the regression model, at monthly time resolution
    def test_plant_analysis(self):
        
        # ____________________________________________________________________
        # Test inputs to the regression model, at monthly time resolution
        self.analysis = plant_analysis.MonteCarloAEP(self.project, 
                                                      reanal_products=['merra2', 'era5'],
                                                      time_resolution = 'M',
                                                      reg_temperature = True, 
                                                      reg_winddirection = True)
        df = self.analysis._aggregate.df

        # Check the pre-processing functions
        self.check_process_revenue_meter_energy_monthly(df)
        self.check_process_loss_estimates_monthly(df)
        self.check_process_reanalysis_data_monthly(df)

        # ____________________________________________________________________
        # Test inputs to the regression model, at daily time resolution
        self.analysis = plant_analysis.MonteCarloAEP(self.project, 
                                                      reanal_products=['merra2', 'era5'],
                                                      time_resolution = 'D',
                                                      reg_temperature = True, 
                                                      reg_winddirection = True)
        df = self.analysis._aggregate.df
        # Check the pre-processing functions
        self.check_process_revenue_meter_energy_daily(df)
        self.check_process_loss_estimates_daily(df)
        self.check_process_reanalysis_data_daily(df)
        
        # ____________________________________________________________________
        # Test linear regression model, at monthly time resolution
        self.analysis = plant_analysis.MonteCarloAEP(self.project, 
                                                      reanal_products=['merra2', 'era5'],
                                                      time_resolution = 'M',
                                                      reg_model = 'lin',
                                                      reg_temperature = False, 
                                                      reg_winddirection = False)
        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=30)
        sim_results = self.analysis.results
        self.check_simulation_results_lin_monthly(sim_results)

        # ____________________________________________________________________
        # Test linear regression model, at daily time resolution
        self.analysis = plant_analysis.MonteCarloAEP(self.project, 
                                                      reanal_products=['merra2', 'era5'],
                                                      time_resolution = 'D',
                                                      reg_model = 'lin',
                                                      reg_temperature = False, 
                                                      reg_winddirection = False)
        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=30)
        sim_results = self.analysis.results
        self.check_simulation_results_lin_daily(sim_results)

        # ____________________________________________________________________
        # Test GAM regression model (can be used at daily time resolution only)
        self.analysis = plant_analysis.MonteCarloAEP(self.project, 
                                                      reanal_products=['merra2', 'era5'],
                                                      time_resolution = 'D',
                                                      reg_model = 'gam',
                                                      reg_temperature = False, 
                                                      reg_winddirection = True)
        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=10)
        sim_results = self.analysis.results
        self.check_simulation_results_gam_daily(sim_results)

        # ____________________________________________________________________
        # Test GBM regression model (can be used at daily time resolution only)
        self.analysis = plant_analysis.MonteCarloAEP(self.project, 
                                                      reanal_products=['merra2', 'era5'],
                                                      time_resolution = 'D',
                                                      reg_model = 'gbm',
                                                      reg_temperature = True, 
                                                      reg_winddirection = True)
        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=10)
        sim_results = self.analysis.results
        self.check_simulation_results_gbm_daily(sim_results)

        # ____________________________________________________________________
        # Test ETR regression model (can be used at daily time resolution only)
        self.analysis = plant_analysis.MonteCarloAEP(self.project, 
                                                      reanal_products=['merra2', 'era5'],
                                                      time_resolution = 'D',
                                                      reg_model = 'etr',
                                                      reg_temperature = False, 
                                                      reg_winddirection = False)
        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=10)
        sim_results = self.analysis.results
        self.check_simulation_results_etr_daily(sim_results)

    def check_process_revenue_meter_energy_monthly(self, df):
        # Energy Nan flags are all zero
        nptest.assert_array_equal(df['energy_nan_perc'].values, np.repeat(0.0, df.shape[0]))

        # Expected number of days per month are equal to number of actual days
        nptest.assert_array_equal(df['num_days_expected'], df['num_days_actual'])

        # Check a few energy values
        expected_gwh = pd.Series([0.692400, 1.471730, 0.580035])
        actual_gwh = df.loc[pd.to_datetime(['2014-06-01', '2014-12-01', '2015-10-01']), 'energy_gwh']
        nptest.assert_array_almost_equal(expected_gwh, actual_gwh)

    def check_process_loss_estimates_monthly(self, df):
        # Availablity, curtailment nan fields both 0, NaN flag is all False
        nptest.assert_array_equal(df['avail_nan_perc'].values, np.repeat(0.0, df.shape[0]))
        nptest.assert_array_equal(df['curt_nan_perc'].values, np.repeat(0.0, df.shape[0]))
        nptest.assert_array_equal(df['nan_flag'].values, np.repeat(False, df.shape[0]))
        
        # Check a few reported availabilty and curtailment values
        expected_avail_gwh = pd.Series([0.029417, 0.021005, 0.000444])
        expected_curt_gwh = pd.Series([0.013250, 0.000000, 0.000000])
        expected_avail_pct = pd.Series([0.040019, 0.014071, 0.000765])
        expected_curt_pct = pd.Series([0.018026, 0.000000, 0.000000])
        
        date_ind = pd.to_datetime(['2014-06-01', '2014-12-01', '2015-10-01'])

        nptest.assert_array_almost_equal(expected_avail_gwh, df.loc[date_ind, 'availability_gwh'])
        nptest.assert_array_almost_equal(expected_curt_gwh, df.loc[date_ind, 'curtailment_gwh'])
        nptest.assert_array_almost_equal(expected_avail_pct, df.loc[date_ind, 'availability_pct'])
        nptest.assert_array_almost_equal(expected_curt_pct, df.loc[date_ind, 'curtailment_pct'])

    def check_process_reanalysis_data_monthly(self, df):
        # Check a few wind speed values
        expected_merra2 = pd.Series([5.43, 6.87, 5.03])
        expected_era5 = pd.Series([5.21, 6.72, 5.24])

        date_ind = pd.to_datetime(['2014-06-01', '2014-12-01', '2015-10-01'])

        nptest.assert_array_almost_equal(expected_merra2, df.loc[date_ind, 'merra2'], decimal = 2)
        nptest.assert_array_almost_equal(expected_era5, df.loc[date_ind, 'era5'], decimal = 2)
        
        # Check a few wind direction values
        expected_merra2_wd = pd.Series([11.7, 250.9, 123.7])
        expected_era5_wd = pd.Series([23.4, 253.1, 121.3])

        date_ind = pd.to_datetime(['2014-06-01', '2014-12-01', '2015-10-01'])

        nptest.assert_array_almost_equal(expected_merra2_wd*2*np.pi/360, df.loc[date_ind, 'merra2_wd'], decimal = 1)
        nptest.assert_array_almost_equal(expected_era5_wd*2*np.pi/360, df.loc[date_ind, 'era5_wd'], decimal = 1)
        
        # Check a few temperature values
        expected_merra2_temp = pd.Series([289.9, 275.3, 281.7])
        expected_era5_temp = pd.Series([290.8, 276.6, 282.7])

        date_ind = pd.to_datetime(['2014-06-01', '2014-12-01', '2015-10-01'])

        nptest.assert_array_almost_equal(expected_merra2_temp, df.loc[date_ind, 'merra2_temperature_K'],decimal = 1)
        nptest.assert_array_almost_equal(expected_era5_temp, df.loc[date_ind, 'era5_temperature_K'], decimal = 1)       

    def check_process_revenue_meter_energy_daily(self, df):
        # Energy Nan flags are all zero
        nptest.assert_array_equal(df['energy_nan_perc'].values, np.repeat(0.0, df.shape[0]))

        # Check a few energy values
        expected_gwh = pd.Series([0.0848525, 0.0253657, 0.0668642])
        actual_gwh = df.loc[pd.to_datetime(['2014-01-02', '2014-10-12', '2015-12-28']), 'energy_gwh']
        nptest.assert_array_almost_equal(expected_gwh, actual_gwh)

    def check_process_loss_estimates_daily(self, df):
        # Availablity, curtailment nan fields both 0, NaN flag is all False
        nptest.assert_array_equal(df['avail_nan_perc'].values, np.repeat(0.0, df.shape[0]))
        nptest.assert_array_equal(df['curt_nan_perc'].values, np.repeat(0.0, df.shape[0]))
        nptest.assert_array_equal(df['nan_flag'].values, np.repeat(False, df.shape[0]))
        
        # Check a few reported availabilty and curtailment values
        expected_avail_gwh = pd.Series([0.0000483644, 0.000000, 0.000000])
        expected_curt_gwh = pd.Series([0.000000, 0.00019581, 0.000000])
        expected_avail_pct = pd.Series([0.000569658, 0.000000, 0.000000])
        expected_curt_pct = pd.Series([0.000000, 0.00766034, 0.000000])
        
        date_ind = pd.to_datetime(['2014-01-02', '2014-10-12', '2015-12-28'])

        nptest.assert_array_almost_equal(expected_avail_gwh, df.loc[date_ind, 'availability_gwh'])
        nptest.assert_array_almost_equal(expected_curt_gwh, df.loc[date_ind, 'curtailment_gwh'])
        nptest.assert_array_almost_equal(expected_avail_pct, df.loc[date_ind, 'availability_pct'])
        nptest.assert_array_almost_equal(expected_curt_pct, df.loc[date_ind, 'curtailment_pct'])

    def check_process_reanalysis_data_daily(self, df):
        # Check a few wind speed values
        expected_merra2 = pd.Series([11.02, 7.04, 8.42])
        expected_era5 = pd.Series([10.48, 7.71, 9.61])

        date_ind = pd.to_datetime(['2014-01-02', '2014-10-12', '2015-12-28'])

        nptest.assert_array_almost_equal(expected_merra2, df.loc[date_ind, 'merra2'],decimal = 2)
        nptest.assert_array_almost_equal(expected_era5, df.loc[date_ind, 'era5'], decimal = 2)
        
        # Check a few wind direction values
        expected_merra2_wd = pd.Series([213.8, 129.1, 170.4])
        expected_era5_wd = pd.Series([212.2, 127.8, 170.3])

        date_ind = pd.to_datetime(['2014-01-02', '2014-10-12', '2015-12-28'])

        nptest.assert_array_almost_equal(expected_merra2_wd*2*np.pi/360, df.loc[date_ind, 'merra2_wd'], decimal = 1)
        nptest.assert_array_almost_equal(expected_era5_wd*2*np.pi/360, df.loc[date_ind, 'era5_wd'], decimal = 1)
        
        # Check a few temperature values
        expected_merra2_temp = pd.Series([279.7, 285.7, 278.2])
        expected_era5_temp = pd.Series([281.1, 285.8, 280.4])

        date_ind = pd.to_datetime(['2014-01-02', '2014-10-12', '2015-12-28'])

        nptest.assert_array_almost_equal(expected_merra2_temp, df.loc[date_ind, 'merra2_temperature_K'], decimal = 1)
        nptest.assert_array_almost_equal(expected_era5_temp, df.loc[date_ind, 'era5_temperature_K'], decimal = 1)       
 
    def check_simulation_results_lin_monthly(self, s):
        # Make sure AEP results are consistent to one decimal place
        expected_results = [12.41, 8.29, 1.30, 3.57, 0.09, 3.57]

        calculated_results = [s.aep_GWh.mean(),
                              s.aep_GWh.std() / s.aep_GWh.mean() * 100,
                              s.avail_pct.mean() * 100,
                              s.avail_pct.std() / s.avail_pct.mean() * 100,
                              s.curt_pct.mean() * 100,
                              s.curt_pct.std() / s.curt_pct.mean() * 100, ]

        nptest.assert_array_almost_equal(expected_results, calculated_results, decimal=0)

    def check_simulation_results_lin_daily(self, s):
        # Make sure AEP results are consistent to one decimal place
        expected_results = [12.31, 4.76, 1.36, 4.90, 0.09, 4.91]

        calculated_results = [s.aep_GWh.mean(),
                              s.aep_GWh.std() / s.aep_GWh.mean() * 100,
                              s.avail_pct.mean() * 100,
                              s.avail_pct.std() / s.avail_pct.mean() * 100,
                              s.curt_pct.mean() * 100,
                              s.curt_pct.std() / s.curt_pct.mean() * 100, ]

        nptest.assert_array_almost_equal(expected_results, calculated_results, decimal=0)

    def check_simulation_results_gam_daily(self, s):
        # Make sure AEP results are consistent to one decimal place
        expected_results = [12.68, 4.50, 1.36, 4.44, 0.087, 4.44]

        calculated_results = [s.aep_GWh.mean(),
                              s.aep_GWh.std() / s.aep_GWh.mean() * 100,
                              s.avail_pct.mean() * 100,
                              s.avail_pct.std() / s.avail_pct.mean() * 100,
                              s.curt_pct.mean() * 100,
                              s.curt_pct.std() / s.curt_pct.mean() * 100, ]

        nptest.assert_array_almost_equal(expected_results, calculated_results, decimal=-1)
        
    def check_simulation_results_gbm_daily(self, s):
        # Make sure AEP results are consistent to one decimal place
        expected_results = [12.82, 3.84, 1.35, 5.17, 0.09, 5.17]

        calculated_results = [s.aep_GWh.mean(),
                              s.aep_GWh.std() / s.aep_GWh.mean() * 100,
                              s.avail_pct.mean() * 100,
                              s.avail_pct.std() / s.avail_pct.mean() * 100,
                              s.curt_pct.mean() * 100,
                              s.curt_pct.std() / s.curt_pct.mean() * 100, ]

        nptest.assert_array_almost_equal(expected_results, calculated_results, decimal=-1)

    def check_simulation_results_etr_daily(self, s):
        # Make sure AEP results are consistent to one decimal place
        expected_results = [12.56, 3.99, 1.35, 5.10, 0.09, 5.10]

        calculated_results = [s.aep_GWh.mean(),
                              s.aep_GWh.std() / s.aep_GWh.mean() * 100,
                              s.avail_pct.mean() * 100,
                              s.avail_pct.std() / s.avail_pct.mean() * 100,
                              s.curt_pct.mean() * 100,
                              s.curt_pct.std() / s.curt_pct.mean() * 100, ]

        nptest.assert_array_almost_equal(expected_results, calculated_results, decimal=-1)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
