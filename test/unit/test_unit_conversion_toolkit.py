import unittest

import numpy as np
import pandas as pd
from numpy import testing as nptest

from openoa.utils import unit_conversion


class SimpleUnitConversionTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_convert_power_to_energy(self):
        np.random.seed(42)
        power = np.random.random(100) * 100
        time_interval = {"10min": 10, "30min": 30.0, "h": 60}  # Minutes
        df = pd.DataFrame(data={"power_kw": power})

        for key, item in time_interval.items():
            energy = power * item / 60
            energy_test = unit_conversion.convert_power_to_energy(df["power_kw"], key)
            nptest.assert_almost_equal(
                energy, energy_test, err_msg="Convert power to energy is broken"
            )

    def test_compute_gross_energy(self):
        # Set some test values
        net = pd.DataFrame([[1, 0, 1], [1, 1, 1], [1, 1, 1]], columns=["test1", "test2", "test3"])
        avail = pd.DataFrame(
            [[0.05, 0.05, 0.05], [0.08, 0.9, 0.9], [0.2, 0.2, 0]],
            columns=["test1", "test2", "test3"],
        )
        avail = pd.DataFrame(
            [[0.05, 0.05, 0.05], [0.08, 0.9, 0.9], [0.2, 0.2, 0]],
            columns=["test1", "test2", "test3"],
        )
        curtail = pd.DataFrame(
            [[0.05, 0.05, 0.05], [0.05, 0.2, -0.1], [0.05, 0.05, 0.05]],
            columns=["test1", "test2", "test3"],
        )

        # Make sure different combinations of 'frac' and 'energy' based measurements work out
        nptest.assert_almost_equal(
            unit_conversion.compute_gross_energy(
                net.test1, avail.test1, curtail.test1, "frac", "frac"
            ),
            [1.1111, 1.1494, 1.3333],
            decimal=4,
        )
        nptest.assert_almost_equal(
            unit_conversion.compute_gross_energy(
                net.test1, avail.test1, curtail.test1, "energy", "frac"
            ),
            [1.1026, 1.1326, 1.2526],
            decimal=4,
        )
        nptest.assert_almost_equal(
            unit_conversion.compute_gross_energy(
                net.test1, avail.test1, curtail.test1, "frac", "energy"
            ),
            [1.1026, 1.1370, 1.3000],
            decimal=4,
        )
        nptest.assert_almost_equal(
            unit_conversion.compute_gross_energy(
                net.test1, avail.test1, curtail.test1, "energy", "energy"
            ),
            [1.1000, 1.1300, 1.2500],
            decimal=4,
        )

        # Make sure exceptions are thrown when bad input data is identified
        def func():  # Function to return exception
            unit_conversion.compute_gross_energy(
                net.test2, avail.test2, curtail.test2, "frac", "frac"
            )

        with self.assertRaises(ValueError):
            func()

        def func():  # Function to return exception
            unit_conversion.compute_gross_energy(
                net.test3, avail.test3, curtail.test3, "frac", "frac"
            )

        with self.assertRaises(ValueError):
            func()

    def test_convert_feet_to_meter(self):
        random_ft = np.random.random(100) * 10
        random_m = random_ft * 0.3048

        df = pd.DataFrame(data={"test_data_ft": random_ft})

        ft_to_m_test = unit_conversion.convert_feet_to_meter(df["test_data_ft"])

        nptest.assert_almost_equal(
            random_m, ft_to_m_test, err_msg="Convert feet to meter is broken"
        )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
