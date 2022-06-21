import unittest

import numpy as np
import pandas as pd
from numpy import testing as nptest

from operational_analysis.toolkits import met_data_processing as mt


class SimpleMetProcessing(unittest.TestCase):
    def setUp(self):
        pass

    def test_compute_wind_direction(self):
        u = [0, -1, -1, -1, 0, 1, 1, 1]  # u-vector of wind
        v = [-1, -1, 0, 1, 1, 1, 0, -1]  # v-vector of wind
        wd_ans = [0, 45, 90, 135, 180, 225, 270, 315]  # Expected result

        y = mt.compute_wind_direction(u, v)  # Test result
        nptest.assert_array_equal(y, wd_ans)

    def test_compute_u_v_components(self):
        wind_speed = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # Wind speed
        wind_direction = np.array([0, 45, 90, 135, 180, 225, 270, 315])  # Wind direction
        u, v = mt.compute_u_v_components(wind_speed, wind_direction)  # Test result

        sqrt_2 = 1 / np.sqrt(2)  # Handy constant

        u_ans = np.array([0, -sqrt_2, -1, -sqrt_2, 0, sqrt_2, 1, sqrt_2])  # Expected result for 'u'
        v_ans = np.array([-1, -sqrt_2, 0, sqrt_2, 1, sqrt_2, 0, -sqrt_2])  # Expected result for 'v'

        nptest.assert_array_almost_equal(u, u_ans, decimal=5)
        nptest.assert_array_almost_equal(v, v_ans, decimal=5)

    def test_compute_air_density(self):
        # Test data frame with pressure and temperature data

        temp = np.arange(280, 300, 5)
        pres = np.arange(90000, 110000, 5000)

        rho = mt.compute_air_density(temp, pres)  # Test result
        rho_ans = np.array([1.11744, 1.1581, 1.19706, 1.23427])  # Expected result

        nptest.assert_array_almost_equal(rho, rho_ans, decimal=5)

    def test_pressure_vertical_extrapolation(self):
        # Define test data
        p_samp = np.array([1e6, 9.5e5])  # pressure at lower level
        z0_samp = np.array([0, 30])  # lower level height
        z1_samp = np.array([100, 100])  # extrapolation level height
        t_samp = np.array([290, 300])  # average temperature in layer between z0 and z1

        p1 = mt.pressure_vertical_extrapolation(p_samp, t_samp, z0_samp, z1_samp)  # Test result
        p1_ans = np.array([988288.905, 942457.391])  # Expected result

        nptest.assert_array_almost_equal(p1, p1_ans, decimal=2)

    def test_air_density_adjusted_wind_speed(self):
        # Test dataframe with wind speed and density data
        wind_speed = np.arange(0, 10, 2)
        dens = np.arange(1.10, 1.20, 0.02)

        adjusted_ws = mt.air_density_adjusted_wind_speed(wind_speed, dens)  # Test answer
        adjusted_ws_ans = np.array([0.0, 1.988235, 4.0, 6.034885, 8.092494])  # Expected answer

        nptest.assert_array_almost_equal(adjusted_ws, adjusted_ws_ans, decimal=5)

    def test_compute_turbulence_intensity(self):

        mean = np.linspace(2.0, 25.0, 10)
        std = np.linspace(0.1, 2.0, 10)
        computed_TI = mt.compute_turbulence_intensity(mean, std)
        expected_TI = np.array(
            [
                0.05,
                0.06829268,
                0.0734375,
                0.07586207,
                0.07727273,
                0.07819549,
                0.07884615,
                0.07932961,
                0.07970297,
                0.08,
            ]
        )
        nptest.assert_allclose(
            computed_TI, expected_TI, err_msg="Turbulence intensity not properly computed."
        )

    def test_compute_shear(self):
        expected_alpha = np.array([-0.1, 0.1, 0.2, 0.4])
        height_low = 30.0
        height_mid = 60.0
        # height_high = 80.0

        df = pd.DataFrame(
            data={
                "wind_low": np.array(
                    [4.2870938501451725, 7.464263932294459, 5.223303379776745, 3.031433133020796]
                ),
                "wind_mid": np.array([4.0, 8.0, 6.0, 4.0]),
                "wind_high": np.array(
                    [3.886566631452294, 8.233488071718085, 6.355343046292873, 4.487820581784798]
                ),
            }
        )
        # Two sensor test
        windspeed_heights = {"wind_low": height_low, "wind_mid": height_mid}
        computed_alpha = mt.compute_shear(df, windspeed_heights, "")
        nptest.assert_allclose(
            computed_alpha, expected_alpha, err_msg="Shear two-sensor computation failing."
        )

        # Multiple sensor test
        windspeed_heights = {"wind_low": 30.0, "wind_mid": 60.0, "wind_high": 80.0}
        computed_alpha = mt.compute_shear(df, windspeed_heights, "wind_mid")
        nptest.assert_allclose(
            computed_alpha, expected_alpha, err_msg="Shear multi-sensor optimization failing."
        )

    def test_compute_veer(self):
        wind_low = np.linspace(2.0, 10.0, 10)
        wind_high = np.linspace(8.0, 25.0, 10)
        height_low = 30.0
        height_high = 80.0

        computed_veer = mt.compute_veer(wind_low, height_low, wind_high, height_high)
        expected_veer = np.array([0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3])
        nptest.assert_allclose(computed_veer, expected_veer, err_msg="Veer computed incorrectly.")

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
