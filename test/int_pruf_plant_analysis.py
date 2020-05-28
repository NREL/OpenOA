import unittest

import numpy as np
import pandas as pd
from numpy import testing as nptest

from operational_analysis.types import plant
from operational_analysis.methods import plant_analysis
from examples.operational_AEP_analysis.project_EIA import Project_EIA


class TestPandasPrufPlantAnalysis(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

        # Set up data to use for testing (EIA example plant)
        self.project = Project_EIA('./examples/operational_AEP_analysis/data')
        self.project.prepare()
        self.analysis = plant_analysis.MonteCarloAEP(self.project)

    def test_validate_data(self):
        self.assertTrue(self.project.validate(), "Failed to validate PlantData from schema")


    def test_plant_analysis(self):

        df = self.analysis._monthly.df

        # Check the pre-processing functions
        self.check_process_revenue_meter_energy(df)
        self.check_process_loss_estimates(df)
        self.check_process_reanalysis_data(df)

        # Check outlier filtering
        self.check_filter_outliers()

        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=2000, reanal_subset=['ncep2', 'merra2', 'erai'])

        sim_results = self.analysis.results
        self.check_simulation_results(sim_results)

    def check_process_revenue_meter_energy(self, df):
        # Energy Nan flags are all zero
        nptest.assert_array_equal(df['energy_nan_perc'].to_numpy(), np.repeat(0.0, df.shape[0]))

        # Expected number of days per month are equal to number of actual days
        nptest.assert_array_equal(df['num_days_expected'], df['num_days_actual'])

        # Check a few energy values
        expected_gwh = pd.Series([6.765, 5.945907, 8.576])
        actual_gwh = df.loc[pd.to_datetime(['2003-12-01', '2010-05-01', '2015-01-01']), 'energy_gwh']
        nptest.assert_array_almost_equal(expected_gwh, actual_gwh)

    def check_process_loss_estimates(self, df):
        # Availablity, curtailment nan fields both 0, NaN flag is all False
        nptest.assert_array_equal(df['avail_nan_perc'].to_numpy(), np.repeat(0.0, df.shape[0]))
        nptest.assert_array_equal(df['curt_nan_perc'].to_numpy(), np.repeat(0.0, df.shape[0]))
        nptest.assert_array_equal(df['nan_flag'].to_numpy(), np.repeat(False, df.shape[0]))

        # Check a few reported availabilty and curtailment values
        expected_avail_gwh = pd.Series([0.236601, 0.161961, 0.724330])
        expected_curt_gwh = pd.Series([0.122979, 0.042608, 0.234614])
        expected_avail_pct = pd.Series([0.033209, 0.026333, 0.075966])
        expected_curt_pct = pd.Series([0.017261, 0.006928, 0.024606])

        date_ind = pd.to_datetime(['2003-12-01', '2010-05-01', '2015-01-01'])

        nptest.assert_array_almost_equal(expected_avail_gwh, df.loc[date_ind, 'availability_gwh'])
        nptest.assert_array_almost_equal(expected_curt_gwh, df.loc[date_ind, 'curtailment_gwh'])
        nptest.assert_array_almost_equal(expected_avail_pct, df.loc[date_ind, 'availability_pct'])
        nptest.assert_array_almost_equal(expected_curt_pct, df.loc[date_ind, 'curtailment_pct'])

    def check_process_reanalysis_data(self, df):
        # Check a few reported availabilty and curtailment values
        expected_merra2 = pd.Series([5.782525, 4.400884, 5.938899])
        expected_ncep2 = pd.Series([6.287746, 4.576479, 6.191463])
        expected_erai = pd.Series([6.081313, 4.315952, 5.738753])

        date_ind = pd.to_datetime(['2003-12-01', '2010-05-01', '2015-01-01'])

        nptest.assert_array_almost_equal(expected_merra2, df.loc[date_ind, 'merra2'])
        nptest.assert_array_almost_equal(expected_erai, df.loc[date_ind, 'erai'])
        nptest.assert_array_almost_equal(expected_ncep2, df.loc[date_ind, 'ncep2'])

    def check_filter_outliers(self):
        # Run a few combinations of outlier criteria, count number of data points remaining
        filt_params = {'a': ('merra2', 2, 0.15, 130),
                       'b': ('merra2', 3, 0.15, 144),
                       'c': ('merra2', 2, 0.25, 135),
                       'd': ('ncep2', 2, 0.25, 136),
                       'e': ('erai', 3, 0.25, 146)}

        for key, values in filt_params.items():
            nptest.assert_equal((self.analysis.filter_outliers(values[0], values[1], values[2])).shape[0], values[3])

    def check_simulation_results(self, s):
        # Make sure AEP results are consistent to one decimal place
        expected_results = [81.90, 1.66, 6.41, 5.01, 2.09, 5.01]

        calculated_results = [s.aep_GWh.mean(),
                              s.aep_GWh.std() / s.aep_GWh.mean() * 100,
                              s.avail_pct.mean() * 100,
                              s.avail_pct.std() / s.avail_pct.mean() * 100,
                              s.curt_pct.mean() * 100,
                              s.curt_pct.std() / s.curt_pct.mean() * 100, ]

        nptest.assert_array_almost_equal(expected_results, calculated_results, decimal=1)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
