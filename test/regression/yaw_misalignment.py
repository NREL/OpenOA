import random
import unittest

import numpy as np
import pandas as pd
import pytest
from numpy import testing as nptest

from openoa.analysis import yaw_misalignment


from test.conftest import project_ENGIE, example_data_path_str  # isort: skip


def reset_prng():
    np.random.seed(42)
    random.seed(42)


class TestStaticYawMisalignment(unittest.TestCase):
    def setUp(self):
        """
        Python Unittest setUp method.
        Load data from disk into PlantData objects and prepare the data for testing the
        StaticYawMisalignment method.
        """
        reset_prng()

        # Set up data to use for testing (ENGIE example plant)
        self.project = project_ENGIE.prepare(example_data_path_str)
        self.project.analysis_type.append("StaticYawMisalignment")
        self.project.validate()

    def test_yaw_misaliginment_without_UQ(self):
        reset_prng()
        # ____________________________________________________________________
        # Test estimated static yaw misalignment values for each turbine and wind speed bin,
        # without UQ. Aside from the expanded wind speed bin range, the min_vane_count value, and
        # use_power_coeff, use default parameters.
        self.analysis = yaw_misalignment.StaticYawMisalignment(
            plant=self.project,
            UQ=False,
        )

        # Run yaw misalignment analysis. Confirm the results are consistent.
        self.analysis.run(
            ws_bins=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            min_vane_bin_count=50,
            use_power_coeff=True,
        )
        self.check_simulation_results_yaw_misalignment_without_UQ()

    def test_yaw_misaliginment_with_UQ(self):
        reset_prng()
        # ____________________________________________________________________
        # Test estimated static yaw misalignment values for two turbines and 7 wind speed bins,
        # with UQ. Aside from using a subset of turbine ids, a reduced number of Monte Carlo
        # simulations, the expanded wind speed bin range, the min_vane_count value, and
        # use_power_coeff, use default parameters.
        self.analysis = yaw_misalignment.StaticYawMisalignment(
            plant=self.project,
            turbine_ids=["R80721", "R80790"],
            UQ=True,
        )

        # Run yaw misalignment analysis. Confirm the results are consistent.
        self.analysis.run(
            num_sim=20,
            ws_bins=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            min_vane_bin_count=50,
            use_power_coeff=True,
        )
        self.check_simulation_results_yaw_misalignment_with_UQ()

    def test_yaw_misaliginment_with_UQ_new_parameters(self):
        reset_prng()
        # ____________________________________________________________________
        # Test estimated static yaw misalignment values for the other two turbines and 4 wind speed
        # bins, with UQ. The parameter use_power_coeff will be set to its default value of False.
        # Assign non-default values for the turbine ids, number of Monte Carlo simulations, wind
        # speed bin range, wind speed bin width, wind vane bin width, and min_vane_count value.
        self.analysis = yaw_misalignment.StaticYawMisalignment(
            plant=self.project,
            turbine_ids=["R80711", "R80736"],
            UQ=True,
        )

        # Run yaw misalignment analysis. Confirm the results are consistent.
        self.analysis.run(
            num_sim=20,
            ws_bins=[4.5, 6.5, 8.5, 10.5],
            ws_bin_width=2.0,
            vane_bin_width=2.0,
            min_vane_bin_count=50,
            use_power_coeff=False,
        )
        self.check_simulation_results_yaw_misalignment_with_UQ_new_params()

    def check_simulation_results_yaw_misalignment_without_UQ(self):
        # Make sure yaw misalignment results are consistent to six decimal places without UQ.
        # Average yaw misaligment values for each turbine
        expected_yaw_mis_results_overall = [1.2216161, 3.1924941, 1.25333209, 2.87197031]

        # Yaw misaligment values for each turbine and wind speed
        expected_yaw_mis_results_ws = np.array(
            [
                [
                    0.3297503,
                    -0.24664158,
                    -0.56610915,
                    -0.73125923,
                    0.3361486,
                    2.41273684,
                    7.01668691,
                ],
                [0.41128732, 1.8577071, 0.79602936, 0.3530183, 4.45674696, 8.09740152, 6.37526813],
                [
                    0.68372144,
                    0.93489074,
                    0.14714568,
                    -1.15133491,
                    0.77473953,
                    2.82345142,
                    4.56071075,
                ],
                [0.77126641, 1.43002494, 0.38850337, 1.32806792, 4.03322139, 5.7117009, 6.44100723],
            ]
        )

        # Mean wind vane angles for each turbine and wind speed
        expected_mean_vane_results_ws = np.array(
            [
                [
                    0.0679621,
                    0.03167305,
                    -0.11787591,
                    -0.12698153,
                    -0.06495893,
                    -0.09881869,
                    -0.35582746,
                ],
                [
                    0.53450601,
                    -0.13305667,
                    -0.04768337,
                    -0.09320136,
                    -0.49048915,
                    -0.61387488,
                    -0.72109781,
                ],
                [
                    0.4199062,
                    0.13011631,
                    -0.03578331,
                    0.00710971,
                    -0.13209128,
                    -0.12596164,
                    -0.15211136,
                ],
                [
                    0.3712822,
                    0.02856771,
                    0.00510602,
                    -0.14140901,
                    -0.26497619,
                    -0.37104273,
                    -0.47997909,
                ],
            ]
        )

        calculated_yaw_mis_results_overall = self.analysis.yaw_misalignment

        nptest.assert_array_almost_equal(
            expected_yaw_mis_results_overall, calculated_yaw_mis_results_overall, decimal=5
        )

        calculated_yaw_mis_results_ws = self.analysis.yaw_misalignment_ws

        nptest.assert_array_almost_equal(
            expected_yaw_mis_results_ws, calculated_yaw_mis_results_ws, decimal=5
        )

        calculated_mean_vane_results_ws = self.analysis.mean_vane_angle_ws

        nptest.assert_array_almost_equal(
            expected_mean_vane_results_ws, calculated_mean_vane_results_ws, decimal=5
        )

    def check_simulation_results_yaw_misalignment_with_UQ(self):
        # Make sure yaw misalignment results are consistent to six decimal places with UQ.
        # Average, std. dev., and 95% confidence intervals of yaw misaligment values for each
        # turbine
        expected_yaw_mis_results_avg_overall = [3.39786547, 2.83936406]
        expected_yaw_mis_results_std_overall = [1.07213289, 0.20931013]
        # expected_yaw_mis_results_95ci_overall = np.array(
        #     [[2.34081532, 6.26432238], [2.54339698, 3.27505334]]
        # )

        # Average, std. dev., and 95% confidence intervals of yaw misaligment values for each
        # turbine and wind speed bin
        expected_yaw_mis_results_avg_ws = np.array(
            [
                [
                    0.29327747,
                    1.67630044,
                    0.75801711,
                    0.28591006,
                    4.32244541,
                    8.66811546,
                    7.78099234,
                ],
                [
                    0.68337628,
                    1.54011846,
                    0.40385385,
                    1.22481477,
                    4.43486872,
                    5.40353726,
                    6.18497911,
                ],
            ]
        )

        expected_yaw_mis_results_std_ws = np.array(
            [
                [0.27452239, 0.25452165, 0.14938781, 0.26991718, 1.26730681, 7.1395735, 3.85032651],
                [0.32210774, 0.2110342, 0.13357729, 0.39072353, 0.90206007, 0.67756395, 1.17456696],
            ]
        )

        # expected_yaw_mis_results_95ci_ws = np.array(
        #     [
        #         [
        #             [-0.18432679, 0.76832261],
        #             [1.30016145, 2.087357],
        #             [0.43874719, 1.00960424],
        #             [-0.07071617, 0.77464821],
        #             [2.44271128, 6.85382468],
        #             [3.30097555, 26.7256876],
        #             [4.65659024, 17.31832531],
        #         ],
        #         [
        #             [0.15930799, 1.19891363],
        #             [1.20126952, 1.92257919],
        #             [0.20607552, 0.66990626],
        #             [0.57507836, 1.92932954],
        #             [3.35247832, 6.65183919],
        #             [4.3391722, 6.80123652],
        #             [4.3402178, 8.74113964],
        #         ],
        #     ]
        # )

        calculated_yaw_mis_results_avg_overall = self.analysis.yaw_misalignment_avg

        nptest.assert_array_almost_equal(
            expected_yaw_mis_results_avg_overall, calculated_yaw_mis_results_avg_overall, decimal=5
        )

        calculated_yaw_mis_results_std_overall = self.analysis.yaw_misalignment_std

        nptest.assert_array_almost_equal(
            expected_yaw_mis_results_std_overall, calculated_yaw_mis_results_std_overall, decimal=5
        )

        # calculated_yaw_mis_results_95ci_overall = self.analysis.yaw_misalignment_95ci

        # nptest.assert_array_almost_equal(
        #     expected_yaw_mis_results_95ci_overall,
        #     calculated_yaw_mis_results_95ci_overall,
        #     decimal=5,
        # )

        calculated_yaw_mis_results_avg_ws = self.analysis.yaw_misalignment_avg_ws

        nptest.assert_array_almost_equal(
            expected_yaw_mis_results_avg_ws, calculated_yaw_mis_results_avg_ws, decimal=5
        )

        calculated_yaw_mis_results_std_ws = self.analysis.yaw_misalignment_std_ws

        nptest.assert_array_almost_equal(
            expected_yaw_mis_results_std_ws, calculated_yaw_mis_results_std_ws, decimal=5
        )

        # calculated_yaw_mis_results_95ci_ws = self.analysis.yaw_misalignment_95ci_ws

        # nptest.assert_array_almost_equal(
        #     expected_yaw_mis_results_95ci_ws, calculated_yaw_mis_results_95ci_ws, decimal=5
        # )

    def check_simulation_results_yaw_misalignment_with_UQ_new_params(self):
        # Make sure yaw misalignment results are consistent to six decimal places with UQ.
        # Average, std. dev., and 95% confidence intervals of yaw misaligment values for each
        # turbine
        expected_yaw_mis_results_avg_overall = [1.75706202, 1.47057408]
        expected_yaw_mis_results_std_overall = [0.45806813, 0.34581758]
        # expected_yaw_mis_results_95ci_overall = np.array(
        #     [[1.06914694, 2.70370656], [1.04684644, 2.35213362]]
        # )

        # Average, std. dev., and 95% confidence intervals of yaw misaligment values for each
        # turbine and wind speed bin
        expected_yaw_mis_results_avg_ws = np.array(
            [
                [-0.08193632, 0.05485578, 2.05100775, 5.00430343],
                [0.92771598, 0.38065238, 1.25760663, 3.31632077],
            ]
        )

        expected_yaw_mis_results_std_ws = np.array(
            [
                [0.38934707, 0.14762966, 1.16797822, 1.51815545],
                [0.41923111, 0.08455833, 0.60220585, 1.37422865],
            ]
        )

        # expected_yaw_mis_results_95ci_ws = np.array(
        #     [
        #         [
        #             [-0.74446219, 0.65962477],
        #             [-0.24180804, 0.27773465],
        #             [-0.27946949, 4.04273506],
        #             [3.2966667, 8.66164706],
        #         ],
        #         [
        #             [0.36079998, 1.80124256],
        #             [0.17200058, 0.50014899],
        #             [0.10686569, 2.2130124],
        #             [1.90708771, 6.94464261],
        #         ],
        #     ]
        # )

        calculated_yaw_mis_results_avg_overall = self.analysis.yaw_misalignment_avg

        nptest.assert_array_almost_equal(
            expected_yaw_mis_results_avg_overall, calculated_yaw_mis_results_avg_overall, decimal=5
        )

        calculated_yaw_mis_results_std_overall = self.analysis.yaw_misalignment_std

        nptest.assert_array_almost_equal(
            expected_yaw_mis_results_std_overall, calculated_yaw_mis_results_std_overall, decimal=5
        )

        # calculated_yaw_mis_results_95ci_overall = self.analysis.yaw_misalignment_95ci

        # nptest.assert_array_almost_equal(
        #     expected_yaw_mis_results_95ci_overall,
        #     calculated_yaw_mis_results_95ci_overall,
        #     decimal=5,
        # )

        calculated_yaw_mis_results_avg_ws = self.analysis.yaw_misalignment_avg_ws

        nptest.assert_array_almost_equal(
            expected_yaw_mis_results_avg_ws, calculated_yaw_mis_results_avg_ws, decimal=5
        )

        calculated_yaw_mis_results_std_ws = self.analysis.yaw_misalignment_std_ws

        nptest.assert_array_almost_equal(
            expected_yaw_mis_results_std_ws, calculated_yaw_mis_results_std_ws, decimal=5
        )

        # calculated_yaw_mis_results_95ci_ws = self.analysis.yaw_misalignment_95ci_ws

        # nptest.assert_array_almost_equal(
        #     expected_yaw_mis_results_95ci_ws, calculated_yaw_mis_results_95ci_ws, decimal=5
        # )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
