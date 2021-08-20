import unittest

import numpy as np
import pandas as pd
from numpy import testing as nptest

from operational_analysis.toolkits import filters


class SimpleFilters(unittest.TestCase):
    def setUp(self):
        pass

    def test_range_flag(self):
        x = pd.Series(np.array([-1, 0, 1]))
        y = filters.range_flag(x, -0.5, 0.5)
        self.assertTrue(y.equals(pd.Series([True, False, True])))

    def test_unresponsive_flag(self):
        x = pd.Series(np.array([-1, -1, -1, 2, 2, 2, 3, 4, 5, 1, 1, 1, 1, 3, 3]))
        y = pd.Series(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
            ]
        )
        y_test = filters.unresponsive_flag(x, threshold=3)
        self.assertTrue(y.equals(y_test))

    def test_window_range_flag(self):
        x = pd.Series(np.array([-1, -1, -1, 1, 1, 1, -1]))
        window = pd.Series(np.array([1, 2, 3, 4, 5, 6, 7]))
        y = pd.Series([False, False, True, False, False, False, True])
        y_test = filters.window_range_flag(window, 3, 8, x, -0.5, 1.5)
        self.assertTrue(y.equals(y_test))

    def test_std_range_flag(self):
        x = pd.Series(np.array([-1, -1, -1, 1, -1, -1, -1]))
        flag = filters.std_range_flag(x, 2)
        expected = pd.Series([False, False, False, True, False, False, False])
        nptest.assert_array_equal(flag, expected)

    # TODO: Test more code paths in bin_filter
    def test_bin_filter(self):
        x_val = pd.Series(np.array([-1, -1, -1, -1, -1, 10, -1]))
        x_bin = pd.Series(np.array([1, 1.5, 2, 2.5, 3, 3.5, 4]))
        flag = filters.bin_filter(x_bin, x_val, 3)
        expected = pd.Series([False, False, False, False, False, True, False])
        nptest.assert_array_equal(flag, expected)

    def test_cluster_mahalanobis_2d(self):
        col1 = pd.Series(np.array([1.0, 1.01, 1.001, 2.0, 2.01, 2.001, 2.0001]))
        col2 = pd.Series(np.array([3.0, 3.02, 3.001, 4.0, 4.01, 4.001, 5.0001]))
        flag = filters.cluster_mahalanobis_2d(col1, col2, 2, 1.5)
        expected = pd.Series(np.array([False, False, False, False, False, False, True]))
        nptest.assert_array_equal(flag, expected)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
