import unittest

import numpy as np
import pandas as pd
from numpy import testing as nptest

from operational_analysis.toolkits import imputing


class SimpleFilters(unittest.TestCase):
    def setUp(self):
        # Test dataframe #1: two assets, one NaN data in first asset that requires imputation
        self.test_df = pd.DataFrame(
            data={
                "time": ["01", "02", "03", "04", "05", "01", "02", "03", "04", "05"],
                "data": [0, np.nan, 4, 5, 8, 13, 18, 20, 20, 30],
                "id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
            },
            index=np.arange(10),
        )

        # Test dataframe #2: two assets, first asset as 2 NaN entries, 2nd asset as one NaN entry overlapping with
        # first asset
        self.test2_df = pd.DataFrame(
            data={
                "time": ["01", "02", "03", "04", "05", "01", "02", "03", "04", "05"],
                "data": [0, np.nan, np.nan, 5, 8, 13, np.nan, 20, 20, 30],
                "id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
            },
            index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        )

        # Test dataframe #3: single asset, no NaN data
        self.test3_df = pd.DataFrame(
            index=["a", "b", "c", "d", "e"],
            data={"data1": [1, 2, 3, 4, 5], "align": ["a", "b", "c", "d", "e"]},
        )

        # Test dataframe #4: single asset, one less entry than dataframe #3
        self.test4_df = pd.DataFrame(
            index=np.arange(4), data={"data2": [1, 2, 4, 5], "align": ["a", "c", "d", "e"]}
        )

        # Test dataframe #5, single asset, 3 NaN data
        self.test5_df = pd.DataFrame(
            index=["a", "b", "c", "d", "e"],
            data={"data1": [1, np.nan, np.nan, np.nan, 5], "align": ["a", "b", "c", "d", "e"]},
        )

        # Test dataframe #6, same as #5 but with values reversed
        self.test6_df = pd.DataFrame(
            index=["a", "b", "c", "d", "e"],
            data={"data1": [5, np.nan, np.nan, np.nan, 1], "align": ["a", "b", "c", "d", "e"]},
        )

        # Test dataframe #7, same as #5 but all values NaN
        self.test7_df = pd.DataFrame(
            index=["a", "b", "c", "d", "e"],
            data={
                "data1": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "align": ["a", "b", "c", "d", "e"],
            },
        )

        # Data frame of two assets with no overlapping occurrences of valid data
        self.test9_df = pd.DataFrame(
            data={
                "time": ["01", "02", "03", "04", "05", "01", "02", "03", "04", "05"],
                "data": [0, np.nan, np.nan, 5, 8, np.nan, 20, 20, np.nan, np.nan],
                "id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
            },
            index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        )

        # Data frame of 3 assets with overlapping NaN occurrences and highly correlated data
        # Asset 'b' in particular shouldn't be able to get imputed
        self.test10_df = pd.DataFrame(
            data={
                "time": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                ],
                "data": [
                    0,
                    np.nan,
                    np.nan,
                    5,
                    8,
                    np.nan,
                    23,
                    33,
                    np.nan,
                    np.nan,
                    20.5,
                    41,
                    np.nan,
                    85,
                    np.nan,
                ],
                "id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "c", "c", "c", "c", "c"],
            },
            index=np.arange(15),
        )

        # Data farme of 3 assets with overlapping NaN occurrences and highly correlated data
        # All data should be imputed
        self.test11_df = pd.DataFrame(
            data={
                "time": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                ],
                "data": [
                    0,
                    np.nan,
                    np.nan,
                    5,
                    8,
                    11,
                    14,
                    np.nan,
                    23,
                    33,
                    np.nan,
                    48,
                    60,
                    68,
                    20.5,
                    41,
                    np.nan,
                    85,
                    np.nan,
                    120,
                    145,
                ],
                "id": [
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                ],
            },
            index=np.arange(21),
        )

        # Data frame of two assets that are poorly correlated
        # No data should be imputed
        self.test12_df = pd.DataFrame(
            data={
                "time": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                ],
                "data": [0, np.nan, np.nan, 5, 8, 11, 14, 40, 40, np.nan, 20, np.nan, 80, 10],
                "id": ["a", "a", "a", "a", "a", "a", "a", "c", "c", "c", "c", "c", "c", "c"],
            },
            index=np.arange(14),
        )

    def test_correlation_matrix_by_id_column(self):
        # Test 1, make sure a simple correlation of two assets works
        y = imputing.correlation_matrix_by_id_column(self.test_df, "time", "id", "data")
        nptest.assert_array_almost_equal(
            y, np.array([[np.nan, 0.970166], [0.970166, np.nan]]), decimal=4
        )

        # Test 2, if no overlapping data are present, make sure correlation matrix is all NaN
        y2 = imputing.correlation_matrix_by_id_column(self.test9_df, "time", "id", "data")
        nptest.assert_array_equal(y2, np.array([[np.nan, np.nan], [np.nan, np.nan]]))

    def test_impute_data(self):
        # Test 1, make sure single NaN is imputed properly and that the index is correct
        y = imputing.impute_data(
            self.test_df.loc[self.test_df.id == "a"],
            "data",
            self.test_df.loc[self.test_df.id == "b"],
            "data",
            "time",
        )
        nptest.assert_almost_equal(np.float64(y.values[1]), np.float64(2.989779), decimal=4)

        # Test 2, make sure only the first NaN entry is imputed
        y2 = imputing.impute_data(
            self.test2_df.loc[self.test2_df.id == "a"],
            "data",
            self.test2_df.loc[self.test2_df.id == "b"],
            "data",
            "time",
        )
        nptest.assert_almost_equal(np.float64(y2.loc["c"]), np.float64(3.874429), decimal=4)
        nptest.assert_equal(y2.loc["b"], np.nan)

        # Test 3, make sure no data is imputed when no NaN are present
        y3 = imputing.impute_data(self.test3_df, "data1", self.test4_df, "data2", "align")
        nptest.assert_array_almost_equal(y3.to_numpy(), self.test3_df["data1"].to_numpy())

        # Test 4, make sure two NaNs are imputed when there 3 total but reference data only matches 2
        y4 = imputing.impute_data(self.test5_df, "data1", self.test4_df, "data2", "align")
        nptest.assert_array_almost_equal(y4.loc[["c", "d"]], np.array([2.0, 4.0]), decimal=4)

        # Test 5, make sure exception is thrown if no valid data are available
        def func():  # Function to return exception
            imputing.imputing.impute_data(self.test5_df, "data1", self.test7_df, "data2", "align")

        with self.assertRaises(Exception):
            func()

    def test_impute_all_assets_by_correlation(self):
        # Test 1, pass data frame with three highly correlated assets, ensure all NaN data are imputed in
        # final output
        y = imputing.impute_all_assets_by_correlation(
            self.test11_df, "data", "data", "time", "id", 0.7
        ).to_frame()
        ans = pd.Series([0.440789, 3.401316, 14.3677, 42.8312, 62.887218, 96.734818])
        nptest.assert_array_almost_equal(
            (y.loc[self.test11_df["data"].isnull(), "imputed_data"]).values, ans.values, decimal=4
        )

        # Test 2, 3 highly correlated assets with less data, such that asset 'b' has no data imputed
        y2 = imputing.impute_all_assets_by_correlation(
            self.test10_df, "data", "data", "time", "id", 0.7
        ).to_frame()
        nan_ind = self.test10_df.loc[self.test10_df["data"].isnull()].index
        ans = pd.Series([1.589147, np.nan, np.nan, np.nan, np.nan, np.nan, 123.7000])
        nptest.assert_array_almost_equal(y2.loc[nan_ind, "imputed_data"], ans, decimal=4)

        # Test 3, 2 poorly correlated data sets, no data should be imputed
        y3 = imputing.impute_all_assets_by_correlation(
            self.test12_df, "data", "data", "time", "id", 0.7
        ).to_frame()
        nptest.assert_array_almost_equal(y3["imputed_data"], self.test12_df["data"], decimal=4)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
