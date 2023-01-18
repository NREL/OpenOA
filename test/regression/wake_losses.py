import random
import unittest

import numpy as np
import pandas as pd
import pytest
from numpy import testing as nptest
from examples import project_ENGIE, example_data_path_str

from openoa.analysis import wake_losses


def reset_prng():
    np.random.seed(42)
    random.seed(42)


class TestWakeLosses(unittest.TestCase):
    def setUp(self):
        """
        Python Unittest setUp method.
        Load data from disk into PlantData objects and prepare the data for testing the WakeLosses method.
        """
        reset_prng()

        # Set up data to use for testing (ENGIE example plant)
        self.project = project_ENGIE.prepare(example_data_path_str)
        self.project.analysis_type.append("WakeLosses")
        self.project.validate()

        # Apply estimated northing calibration to SCADA wind directions
        self.project.scada["WMET_HorWdDir"] = (self.project.scada["WMET_HorWdDir"] + 15.85) % 360.0

    def test_wake_losses_without_UQ(self):
        reset_prng()
        # ____________________________________________________________________
        # Test POR and long-term corrected wake losses at plant and turbine level, without UQ.
        # Limit wind direction assets to three reliable turbines and limit date range to exclude
        # change in wind direction reference. Otherwise, use default parameters.
        self.analysis = wake_losses.WakeLosses(
            plant=self.project,
            wind_direction_asset_ids=["R80711", "R80721", "R80736"],
            end_date="2015-11-25 00:00",
            UQ=False,
        )

        # Run Wake Loss analysis, using default parameters. Aside from no_wakes_ws_thresh_LT_corr,
        # use default parameters. Confirm the results are consistent.
        self.analysis.run(no_wakes_ws_thresh_LT_corr=15.0)
        self.check_simulation_results_wake_losses_without_UQ()

    def test_wake_losses_with_UQ(self):
        reset_prng()
        # ____________________________________________________________________
        # Test POR and long-term corrected wake losses at plant and turbine level, with UQ.
        # Limit wind direction assets to three reliable turbines and limit date range to exclude
        # change in wind direction reference. Otherwise, use default parameters.
        self.analysis = wake_losses.WakeLosses(
            plant=self.project,
            wind_direction_asset_ids=["R80711", "R80721", "R80736"],
            end_date="2015-11-25 00:00",
            UQ=True,
        )

        # Run Wake Loss analysis with 50 Monte Carlo iterations.
        # Aside from no_wakes_ws_thresh_LT_corr and num_sim, use default parameters.
        # Confirm the results are consistent.
        self.analysis.run(num_sim=50, no_wakes_ws_thresh_LT_corr=15.0)
        self.check_simulation_results_wake_losses_with_UQ()

    def test_wake_losses_with_UQ_new_parameters(self):
        reset_prng()
        # ____________________________________________________________________
        # Test POR and long-term corrected wake losses at plant and turbine level, with UQ.
        # Limit wind direction assets to three reliable turbines. Assign non-default start and
        # end dates and end date for reanalysis data for long-term correction.
        self.analysis = wake_losses.WakeLosses(
            plant=self.project,
            wind_direction_asset_ids=["R80711", "R80721", "R80736"],
            start_date="2014-03-01 00:00",
            end_date="2015-10-31 23:50",
            end_date_lt="2018-06-30 23:00",
            UQ=True,
        )

        # Run Wake Loss analysis with 50 Monte Carlo iterations.
        # Use non-default values for wind direction bin width for identifying freestream turbines,
        # freestream sector width, freestream power and wind speed averaging methods, and number of
        # years for long-term correction. Further, do not correct for derated turbines and do not
        # assume no wake losses above a certain wind speed for long-term correction.
        # Confirm the results are consistent.
        self.analysis.run(
            num_sim=50,
            wd_bin_width=10.0,
            freestream_sector_width=(60.0, 100.0),
            freestream_power_method="median",
            freestream_wind_speed_method="median",
            correct_for_derating=False,
            num_years_LT=(5, 15),
            assume_no_wakes_high_ws_LT_corr=False,
        )
        self.check_simulation_results_wake_losses_with_UQ_new_params()

    def check_simulation_results_wake_losses_without_UQ(self):
        # Make sure wake loss results are consistent to six decimal places
        # Confirm plant-level and turbine-level wake losses for POR and long-term corrected
        # wake loss estimates.
        expected_results_por = [0.341373, -11.731022, 10.896697, 4.066565, -1.901445]
        expected_results_lt = [0.366551, -9.720648, 10.275454, 2.925906, -2.043558]

        calculated_results_por = [100 * self.analysis.wake_losses_por]
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por))

        nptest.assert_array_almost_equal(expected_results_por, calculated_results_por)

        calculated_results_lt = [100 * self.analysis.wake_losses_lt]
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt))

        nptest.assert_array_almost_equal(expected_results_lt, calculated_results_lt)

    def check_simulation_results_wake_losses_with_UQ(self):
        # Make sure wake loss results are consistent to six decimal places
        # Confirm plant-level and turbine-level means and std. devs. from Monte Carlo simulation results
        # for POR and long-term corrected wake loss estimates.
        expected_results_por = [
            0.472843,
            1.525822,
            -11.556679,
            11.021310,
            4.174661,
            -1.781930,
            1.704648,
            1.367939,
            1.487695,
            1.557837,
        ]
        expected_results_lt = [
            0.646324,
            1.368698,
            -9.434742,
            10.603874,
            3.129519,
            -1.735322,
            1.536258,
            1.321525,
            1.355933,
            1.420263,
        ]

        calculated_results_por = [
            100 * self.analysis.wake_losses_por_mean,
            100 * self.analysis.wake_losses_por_std,
        ]
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por_mean))
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por_std))

        nptest.assert_array_almost_equal(expected_results_por, calculated_results_por)

        calculated_results_lt = [
            100 * self.analysis.wake_losses_lt_mean,
            100 * self.analysis.wake_losses_lt_std,
        ]
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt_mean))
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt_std))

        nptest.assert_array_almost_equal(expected_results_lt, calculated_results_lt)

    def check_simulation_results_wake_losses_with_UQ_new_params(self):
        # Make sure wake loss results are consistent to six decimal places
        # Confirm plant-level and turbine-level means and std. devs. from Monte Carlo simulation results
        # for POR and long-term corrected wake loss estimates.
        expected_results_por = [
            0.916847,
            2.543303,
            -10.936793,
            11.132650,
            5.244208,
            -1.772678,
            2.863249,
            2.275958,
            2.405864,
            2.637304,
        ]
        expected_results_lt = [
            1.140631,
            2.420886,
            -8.820857,
            10.996088,
            3.488778,
            -1.101484,
            2.496149,
            2.330931,
            2.502628,
            2.430939,
        ]

        calculated_results_por = [
            100 * self.analysis.wake_losses_por_mean,
            100 * self.analysis.wake_losses_por_std,
        ]
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por_mean))
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por_std))

        nptest.assert_array_almost_equal(expected_results_por, calculated_results_por)

        calculated_results_lt = [
            100 * self.analysis.wake_losses_lt_mean,
            100 * self.analysis.wake_losses_lt_std,
        ]
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt_mean))
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt_std))

        nptest.assert_array_almost_equal(expected_results_lt, calculated_results_lt)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
