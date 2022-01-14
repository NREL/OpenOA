import random
import unittest
from test import example_data_path_str

import numpy as np
import pandas as pd
import pytest
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
        self.project = Project_Engie(example_data_path_str)
        self.project.prepare()

        # Set up a new project with modified reanalysis start and end dates
        self.project_rean = Project_Engie(example_data_path_str)
        self.project_rean.prepare()
        self.project_rean._reanalysis._product[
            "merra2"
        ].df = self.project_rean._reanalysis._product["merra2"].df.loc[
            self.project_rean._reanalysis._product["merra2"].df.index <= "2019-04-15 12:30"
        ]
        self.project_rean._reanalysis._product["era5"].df = self.project_rean._reanalysis._product[
            "era5"
        ].df.loc[self.project_rean._reanalysis._product["era5"].df.index >= "1999-01-15 12:00"]

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
        df_rean = self.analysis._reanalysis_aggregate

        # Check the pre-processing functions
        self.check_process_revenue_meter_energy_monthly(df)
        self.check_process_loss_estimates_monthly(df)
        self.check_process_reanalysis_data_monthly(df, df_rean)

    # Test reanalysis start and end dates depending on time resolution and end date argument
    def test_reanalysis_aggregate_monthly(self):
        reset_prng()
        # ____________________________________________________________________
        # Test default aggregate reanalysis values and date range, at monthly time resolution
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project_rean, reanal_products=["merra2", "era5"], time_resolution="M"
        )
        df_rean = self.analysis._reanalysis_aggregate

        # check that date range is truncated correctly
        assert (df_rean.index[0], df_rean.index[-1]) == (
            pd.to_datetime("1999-02-01"),
            pd.to_datetime("2019-03-01"),
        )

        # Check wind speed values at start and end dates
        expected = {"merra2": [7.584891, 8.679547], "era5": [7.241081, 8.188632]}
        computed = {
            key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy()
            for key in expected.keys()
        }

        for key in expected.keys():
            nptest.assert_array_almost_equal(expected[key], computed[key])

        # ____________________________________________________________________
        # Check for invalid user-defined end dates
        # Date range doesn't include full 20 years
        with pytest.raises(ValueError):
            self.analysis = plant_analysis.MonteCarloAEP(
                self.project_rean,
                reanal_products=["merra2", "era5"],
                time_resolution="M",
                end_date_lt="2018-12-15 12:00",
            )

        # End date out of bounds, monthly
        with pytest.raises(ValueError):
            self.analysis = plant_analysis.MonteCarloAEP(
                self.project_rean,
                reanal_products=["merra2", "era5"],
                time_resolution="M",
                end_date_lt="2019-04-15 13:00",
            )

        # ____________________________________________________________________
        # Test aggregate reanalysis values and date range with user-defined end date, at monthly time resolution
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project_rean,
            reanal_products=["merra2", "era5"],
            time_resolution="M",
            end_date_lt="2019-02-10 12:00",
        )
        df_rean = self.analysis._reanalysis_aggregate

        # check that date range is truncated correctly
        assert (df_rean.index[0], df_rean.index[-1]) == (
            pd.to_datetime("1999-02-01"),
            pd.to_datetime("2019-02-01"),
        )

        # Check wind speed values at start and end dates
        expected = {"merra2": [7.584891, 6.529796], "era5": [7.241081, 6.644804]}
        computed = {
            key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy()
            for key in expected.keys()
        }

        for key in expected.keys():
            nptest.assert_array_almost_equal(expected[key], computed[key])

    def test_reanalysis_aggregate_daily(self):
        reset_prng()
        # ____________________________________________________________________
        # Test default aggregate reanalysis values and date range, at daily time resolution
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project_rean, reanal_products=["merra2", "era5"], time_resolution="D"
        )
        df_rean = self.analysis._reanalysis_aggregate

        # check that date range is truncated correctly
        assert (df_rean.index[0], df_rean.index[-1]) == (
            pd.to_datetime("1999-01-16"),
            pd.to_datetime("2019-03-31"),
        )

        # Check wind speed values at start and end dates
        expected = {"merra2": [12.868168, 5.152958], "era5": [12.461761, 5.238968]}
        computed = {
            key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy()
            for key in expected.keys()
        }

        for key in expected.keys():
            nptest.assert_array_almost_equal(expected[key], computed[key])

        # ____________________________________________________________________
        # Check for invalid user-defined end dates
        # Date range doesn't include full 20 years
        with pytest.raises(ValueError):
            self.analysis = plant_analysis.MonteCarloAEP(
                self.project_rean,
                reanal_products=["merra2", "era5"],
                time_resolution="D",
                end_date_lt="2019-01-14 23:00",
            )

        # End date out of bounds, daily
        with pytest.raises(ValueError):
            self.analysis = plant_analysis.MonteCarloAEP(
                self.project_rean,
                reanal_products=["merra2", "era5"],
                time_resolution="D",
                end_date_lt="2019-04-15 13:00",
            )

        # ____________________________________________________________________
        # Test aggregate reanalysis values and date range with user-defined end date, at daily time resolution
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project_rean,
            reanal_products=["merra2", "era5"],
            time_resolution="D",
            end_date_lt="2019-02-10 12:00",
        )
        df_rean = self.analysis._reanalysis_aggregate

        # check that date range is truncated correctly
        assert (df_rean.index[0], df_rean.index[-1]) == (
            pd.to_datetime("1999-01-16"),
            pd.to_datetime("2019-02-10"),
        )

        # Check wind speed values at start and end dates
        expected = {"merra2": [12.868168, 14.571084], "era5": [12.461761, 14.045798]}
        computed = {
            key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy()
            for key in expected.keys()
        }

        for key in expected.keys():
            nptest.assert_array_almost_equal(expected[key], computed[key])

    def test_reanalysis_aggregate_hourly(self):
        reset_prng()
        # ____________________________________________________________________
        # Test default aggregate reanalysis values and date range, at hourly time resolution
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project_rean, reanal_products=["merra2", "era5"], time_resolution="H"
        )
        df_rean = self.analysis._reanalysis_aggregate

        # check that date range is truncated correctly
        assert (df_rean.index[0], df_rean.index[-1]) == (
            pd.to_datetime("1999-01-15 12:00"),
            pd.to_datetime("2019-03-31 23:00"),
        )

        # Check wind speed values at start and end dates
        expected = {"merra2": [10.509840, 9.096710], "era5": [9.202639, 9.486806]}
        computed = {
            key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy()
            for key in expected.keys()
        }

        for key in expected.keys():
            nptest.assert_array_almost_equal(expected[key], computed[key])

        # ____________________________________________________________________
        # Check for invalid user-defined end dates
        # Date range doesn't include full 20 years
        with pytest.raises(ValueError):
            self.analysis = plant_analysis.MonteCarloAEP(
                self.project_rean,
                reanal_products=["merra2", "era5"],
                time_resolution="H",
                end_date_lt="2019-01-15 10:00",
            )

        # End date out of bounds, hourly
        with pytest.raises(ValueError):
            self.analysis = plant_analysis.MonteCarloAEP(
                self.project_rean,
                reanal_products=["merra2", "era5"],
                time_resolution="H",
                end_date_lt="2019-04-15 13:00",
            )

        # ____________________________________________________________________
        # Test aggregate reanalysis values and date range with user-defined end date, at hourly time resolution
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project_rean,
            reanal_products=["merra2", "era5"],
            time_resolution="H",
            end_date_lt="2019-02-10 12:00",
        )
        df_rean = self.analysis._reanalysis_aggregate

        # check that date range is truncated correctly
        assert (df_rean.index[0], df_rean.index[-1]) == (
            pd.to_datetime("1999-01-15 12:00"),
            pd.to_datetime("2019-02-10 12:00"),
        )

        # Check wind speed values at start and end dates
        expected = {"merra2": [10.509840, 16.985526], "era5": [9.202639, 15.608469]}
        computed = {
            key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy()
            for key in expected.keys()
        }

        for key in expected.keys():
            nptest.assert_array_almost_equal(expected[key], computed[key])

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

    def test_daily_gam_outliers(self):
        reset_prng()
        # ____________________________________________________________________
        # Test GAM regression model (can be used at daily time resolution only)
        self.analysis = plant_analysis.MonteCarloAEP(
            self.project,
            reanal_products=["merra2", "era5"],
            time_resolution="D",
            outlier_detection=True,
            reg_model="gam",
            reg_temperature=True,
            reg_winddirection=True,
        )
        # Run Monte Carlo AEP analysis, confirm the results are consistent
        self.analysis.run(num_sim=5)
        sim_results = self.analysis.results
        self.check_simulation_results_gam_daily_outliers(sim_results)

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

    def check_process_reanalysis_data_monthly(self, df, df_rean):

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

        # check that date range is truncated correctly
        assert (df_rean.index[0], df_rean.index[-1]) == (
            pd.to_datetime("1999-01-01 00:00:00"),
            pd.to_datetime("2019-04-01 00:00:00"),
        )

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
        # Make sure AEP results are consistent to six decimal places
        expected_results = [11.284629, 10.801102, 1.130812, 4.287147, 0.061666, 5.365357]

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
        # Make sure AEP results are consistent to six decimal places
        expected_results = [12.781636, 4.428628, 1.323524, 6.135826, 0.049333, 8.462634]

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
        # Make sure AEP results are consistent to six decimal places
        expected_results = [12.957132, 14.928604, 1.310955, 5.248099, 0.049947, 9.21482]

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
        # Make sure AEP results are consistent to six decimal places
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

    def check_simulation_results_gam_daily_outliers(self, s):
        # Make sure AEP results are consistent to six decimal places
        expected_results = [13.46498, 8.248985, 1.324029, 5.966637, 0.048054, 8.851849]

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
