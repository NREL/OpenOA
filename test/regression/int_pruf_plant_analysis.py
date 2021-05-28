import random
import unittest

import numpy as np
import pandas as pd
from numpy import testing as nptest
from examples.project_ENGIE import Project_Engie

from operational_analysis.methods import plant_analysis


def reset_prng():
    np.random.seed(42)
    random.seed(42)


class TestPandasPrufPlantAnalysis(unittest.TestCase):
    def setUp(self):
        reset_prng()
        # Set up data to use for testing (ENGIE example plant)
        self.project = Project_Engie("./examples/data/la_haute_borne")
        self.project.prepare()

    # Test inputs to the regression model, at monthly time resolution
    def test_monthly_inputs(self):
        reset_prng()
        # ____________________________________________________________________
        # Test inputs to the regression model, at monthly time resolution
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project,
            reanal_products=["merra2", "era5"],
            time_resolution="M",
            reg_temperature=True,
            reg_winddirection=True,
        )
        df = self.analysis._aggregate.df

        # Check the pre-processing functions
        self.check_process_revenue_meter_energy_monthly(df)
        self.check_process_loss_estimates_monthly(df)
        self.check_process_reanalysis_data_monthly(df)

    def test_monthly_lin(self):
        reset_prng()
        # ____________________________________________________________________
        # Test linear regression model, at monthly time resolution
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project,
            reanal_products=["merra2", "era5"],
            time_resolution="M",
            reg_model="lin",
            reg_temperature=False,
            reg_winddirection=False,
        )
        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=10)
        sim_results = self.analysis.results
        self.check_simulation_results_lin_monthly(sim_results)

    # Test inputs to the regression model, at daily time resolution
    def test_daily_inputs(self):
        reset_prng()
        # ____________________________________________________________________
        # Test inputs to the regression model, at monthly time resolution
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project,
            reanal_products=["merra2", "era5"],
            time_resolution="D",
            reg_temperature=True,
            reg_winddirection=True,
        )
        df = self.analysis._aggregate.df

        # Check the pre-processing functions
        self.check_process_revenue_meter_energy_daily(df)
        self.check_process_loss_estimates_daily(df)
        self.check_process_reanalysis_data_daily(df)

    def test_daily_gam(self):
        reset_prng()
        # ____________________________________________________________________
        # Test GAM regression model (can be used at daily time resolution only)
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project,
            reanal_products=["merra2", "era5"],
            time_resolution="D",
            reg_model="gam",
            reg_temperature=True,
            reg_winddirection=True,
        )
        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=5)
        sim_results = self.analysis.results
        self.check_simulation_results_gam_daily(sim_results)

    def test_daily_gbm(self):
        reset_prng()
        # ____________________________________________________________________
        # Test GBM regression model (can be used at daily time resolution only)
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project,
            reanal_products=["era5"],
            time_resolution="D",
            reg_model="gbm",
            reg_temperature=True,
            reg_winddirection=False,
        )
        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=5)
        sim_results = self.analysis.results
        self.check_simulation_results_gbm_daily(sim_results)

    def test_daily_etr(self):
        reset_prng()
        # ____________________________________________________________________
        # Test ETR regression model (can be used at daily time resolution only)
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project,
            reanal_products=["merra2"],
            time_resolution="D",
            reg_model="etr",
            reg_temperature=False,
            reg_winddirection=False,
        )
        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=5)
        sim_results = self.analysis.results
        self.check_simulation_results_etr_daily(sim_results)

    def check_process_revenue_meter_energy_monthly(self, df):
        # Energy Nan flags are all zero
        nptest.assert_array_equal(df["energy_nan_perc"].values, np.repeat(0.0, df.shape[0]))

        # Expected number of days per month are equal to number of actual days
        nptest.assert_array_equal(df["num_days_expected"], df["num_days_actual"])

        # Check a few energy values
        expected_gwh = pd.Series([0.692400, 1.471730, 0.580035])
        actual_gwh = df.loc[
            pd.to_datetime(["2014-06-01", "2014-12-01", "2015-10-01"]), "energy_gwh"
        ]
        nptest.assert_array_almost_equal(expected_gwh, actual_gwh)

    def check_process_loss_estimates_monthly(self, df):
        # Availablity, curtailment nan fields both 0, NaN flag is all False
        nptest.assert_array_equal(df["avail_nan_perc"].values, np.repeat(0.0, df.shape[0]))
        nptest.assert_array_equal(df["curt_nan_perc"].values, np.repeat(0.0, df.shape[0]))
        nptest.assert_array_equal(df["nan_flag"].values, np.repeat(False, df.shape[0]))

        # Check a few reported availabilty and curtailment values
        expected_avail_gwh = pd.Series([0.029417, 0.021005, 0.000444])
        expected_curt_gwh = pd.Series([0.013250, 0.000000, 0.000000])
        expected_avail_pct = pd.Series([0.040019, 0.014071, 0.000765])
        expected_curt_pct = pd.Series([0.018026, 0.000000, 0.000000])

        date_ind = pd.to_datetime(["2014-06-01", "2014-12-01", "2015-10-01"])

        nptest.assert_array_almost_equal(expected_avail_gwh, df.loc[date_ind, "availability_gwh"])
        nptest.assert_array_almost_equal(expected_curt_gwh, df.loc[date_ind, "curtailment_gwh"])
        nptest.assert_array_almost_equal(expected_avail_pct, df.loc[date_ind, "availability_pct"])
        nptest.assert_array_almost_equal(expected_curt_pct, df.loc[date_ind, "curtailment_pct"])

    def check_process_reanalysis_data_monthly(self, df):

        expected = {
            "merra2": [5.42523278, 6.86883337, 5.02690892],
            "era5": [5.20508049, 6.71586744, 5.23824611],
            "merra2_wd": [11.74700241, 250.90081133, 123.70142025],
            "era5_wd": [23.4291153, 253.14150601, 121.25886916],
            "merra2_temperature_K": [289.87128364, 275.26493716, 281.72562887],
            "era5_temperature_K": [290.82110632, 276.62490053, 282.71629935],
        }

        date_ind = pd.to_datetime(["2014-06-01", "2014-12-01", "2015-10-01"])
        computed = {key: df.loc[date_ind, key].to_numpy() for key in expected.keys()}

        print(computed)

        for key in expected.keys():
            nptest.assert_array_almost_equal(expected[key], computed[key])

    def check_process_revenue_meter_energy_daily(self, df):
        # Energy Nan flags are all zero
        nptest.assert_array_equal(df["energy_nan_perc"].values, np.repeat(0.0, df.shape[0]))

        # Check a few energy values
        expected_gwh = pd.Series([0.0848525, 0.0253657, 0.0668642])
        actual_gwh = df.loc[
            pd.to_datetime(["2014-01-02", "2014-10-12", "2015-12-28"]), "energy_gwh"
        ]
        nptest.assert_array_almost_equal(expected_gwh, actual_gwh)

    def check_process_loss_estimates_daily(self, df):
        # Availablity, curtailment nan fields both 0, NaN flag is all False
        nptest.assert_array_equal(df["avail_nan_perc"].values, np.repeat(0.0, df.shape[0]))
        nptest.assert_array_equal(df["curt_nan_perc"].values, np.repeat(0.0, df.shape[0]))
        nptest.assert_array_equal(df["nan_flag"].values, np.repeat(False, df.shape[0]))

        # Check a few reported availabilty and curtailment values
        expected_avail_gwh = pd.Series([0.0000483644, 0.000000, 0.000000])
        expected_curt_gwh = pd.Series([0.000000, 0.00019581, 0.000000])
        expected_avail_pct = pd.Series([0.000569658, 0.000000, 0.000000])
        expected_curt_pct = pd.Series([0.000000, 0.00766034, 0.000000])

        date_ind = pd.to_datetime(["2014-01-02", "2014-10-12", "2015-12-28"])

        nptest.assert_array_almost_equal(expected_avail_gwh, df.loc[date_ind, "availability_gwh"])
        nptest.assert_array_almost_equal(expected_curt_gwh, df.loc[date_ind, "curtailment_gwh"])
        nptest.assert_array_almost_equal(expected_avail_pct, df.loc[date_ind, "availability_pct"])
        nptest.assert_array_almost_equal(expected_curt_pct, df.loc[date_ind, "curtailment_pct"])

    def check_process_reanalysis_data_daily(self, df):

        expected = {
            "merra2": np.array([11.02459231, 7.04306896, 8.41880152]),
            "era5": np.array([10.47942319, 7.71069617, 9.60864791]),
            "merra2_wd": np.array([213.81683361, 129.08053181, 170.39815032]),
            "era5_wd": np.array([212.23854097, 127.75317448, 170.33488958]),
            "merra2_temperature_K": np.array([279.67922333, 285.69317833, 278.15574917]),
            "era5_temperature_K": np.array([281.14880642, 285.81961816, 280.42017656]),
        }

        date_ind = pd.to_datetime(["2014-01-02", "2014-10-12", "2015-12-28"])
        computed = {key: df.loc[date_ind, key].to_numpy() for key in expected.keys()}

        print(computed)

        for key in expected.keys():
            nptest.assert_array_almost_equal(expected[key], computed[key])

    def check_simulation_results_lin_monthly(self, s):
        # Make sure AEP results are consistent to one decimal place
        expected_results = [11.401602, 9.789065, 1.131574, 4.766565, 0.059858, 4.847703]

        calculated_results = [
            s.aep_GWh.mean(),
            s.aep_GWh.std() / s.aep_GWh.mean() * 100,
            s.avail_pct.mean() * 100,
            s.avail_pct.std() / s.avail_pct.mean() * 100,
            s.curt_pct.mean() * 100,
            s.curt_pct.std() / s.curt_pct.mean() * 100,
        ]

        nptest.assert_array_almost_equal(expected_results, calculated_results)

    def check_simulation_results_gam_daily(self, s):
        # Make sure AEP results are consistent to one decimal place
        expected_results = [12.807144, 3.959101, 1.320519, 6.294529, 0.049507, 8.152235]

        calculated_results = [
            s.aep_GWh.mean(),
            s.aep_GWh.std() / s.aep_GWh.mean() * 100,
            s.avail_pct.mean() * 100,
            s.avail_pct.std() / s.avail_pct.mean() * 100,
            s.curt_pct.mean() * 100,
            s.curt_pct.std() / s.curt_pct.mean() * 100,
        ]

        nptest.assert_array_almost_equal(expected_results, calculated_results)

    def check_simulation_results_gbm_daily(self, s):
        # Make sure AEP results are consistent to one decimal place
        expected_results = [12.794527, 10.609839, 1.298789, 4.849577, 0.050383, 8.620032]

        calculated_results = [
            s.aep_GWh.mean(),
            s.aep_GWh.std() / s.aep_GWh.mean() * 100,
            s.avail_pct.mean() * 100,
            s.avail_pct.std() / s.avail_pct.mean() * 100,
            s.curt_pct.mean() * 100,
            s.curt_pct.std() / s.curt_pct.mean() * 100,
        ]

        nptest.assert_array_almost_equal(expected_results, calculated_results)

    def check_simulation_results_etr_daily(self, s):
        # Make sure AEP results are consistent to one decimal place
        expected_results = [12.938536, 8.561798, 1.334704, 5.336189, 0.057874, 11.339976]

        calculated_results = [
            s.aep_GWh.mean(),
            s.aep_GWh.std() / s.aep_GWh.mean() * 100,
            s.avail_pct.mean() * 100,
            s.avail_pct.std() / s.avail_pct.mean() * 100,
            s.curt_pct.mean() * 100,
            s.curt_pct.std() / s.curt_pct.mean() * 100,
        ]

        nptest.assert_array_almost_equal(expected_results, calculated_results)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
