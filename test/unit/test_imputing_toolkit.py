import unittest
from multiprocessing.sharedctypes import Value

import numpy as np
import pandas as pd
from numpy import testing as nptest

from openoa.utils import imputing


class SimpleFilters(unittest.TestCase):
    def setUp(self):
        # Test dataframe #1: two assets, one NaN data in first asset that requires imputation
        self.test_df = pd.DataFrame(
            data={
                "time": ["01", "02", "03", "04", "05", "01", "02", "03", "04", "05"],
                "data": [0, np.nan, 4, 5, 8, 13, 18, 20, 20, 30],
                "asset_id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
            },
            index=np.arange(10),
        ).set_index(["time", "asset_id"])

        # Test dataframe #2: two assets, first asset as 2 NaN entries, 2nd asset as one NaN entry overlapping with
        # first asset
        self.test2_df = pd.DataFrame(
            data={
                "time": ["01", "02", "03", "04", "05", "01", "02", "03", "04", "05"],
                "data": [0, np.nan, np.nan, 5, 8, 13, np.nan, 20, 20, 30],
                "asset_id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
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
                "asset_id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
            },
            index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        ).set_index(["time", "asset_id"])

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
                "asset_id": [
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
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                ],
            },
            index=np.arange(15),
        ).set_index(["time", "asset_id"])

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
                "asset_id": [
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
        ).set_index(["time", "asset_id"])

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
                "asset_id": [
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                ],
            },
            index=np.arange(14),
        ).set_index(["time", "asset_id"])

    def test_asset_correlation_matrix(self):
        # Test 1, make sure a simple correlation of two assets works
        y = imputing.asset_correlation_matrix(self.test_df, "data")
        nptest.assert_array_almost_equal(
            y, np.array([[np.nan, 0.970166], [0.970166, np.nan]]), decimal=4
        )

        # Test 2, if no overlapping data are present, make sure correlation matrix is all NaN
        y2 = imputing.asset_correlation_matrix(self.test9_df, "data")
        nptest.assert_array_equal(y2, np.array([[np.nan, np.nan], [np.nan, np.nan]]))

    def test_impute_data(self):
        # Test 1a, make sure single NaN is imputed using old style of inputs
        y = np.float64(2.989779)
        y_test = imputing.impute_data(
            target_data=self.test_df.xs("a", level=1),
            reference_data=self.test_df.xs("b", level=1),
            target_col="data",
            reference_col="data",
            align_col="time",
            method="linear",
        )
        nptest.assert_almost_equal(y_test.values[1], y, decimal=4)

        # Test 1b, make sure single NaN is imputed properly and that the index is correct
        data = self.test_df.unstack().droplevel(0, axis=1)
        y_test = imputing.impute_data(
            data=data, target_col="a", reference_col="b", method="polynomial", degree=1
        )
        nptest.assert_almost_equal(y_test.values[1], y, decimal=4)

        # Test 2, make sure only the first NaN entry is imputed
        y_test = imputing.impute_data(
            "data",
            "data",
            self.test2_df.loc[self.test2_df.asset_id == "a"],
            self.test2_df.loc[self.test2_df.asset_id == "b"],
            "time",
        )
        nptest.assert_almost_equal(np.float64(y_test.loc["c"]), np.float64(3.874429), decimal=4)
        nptest.assert_equal(y_test.loc["b"], np.nan)

        # Test 3, make sure no data is imputed when no NaN are present
        # NOTE: This test shouldn't have ever worked, but for some reason does, but open to
        # other amendments that don't just get rid of a test case
        y_test = imputing.impute_data(
            target_col="data1",
            reference_col="data2",
            target_data=self.test3_df,
            reference_data=self.test4_df,
            align_col="align",
        )
        nptest.assert_array_almost_equal(y_test.to_numpy(), self.test3_df["data1"].to_numpy())

        # Test 4, make sure two NaNs are imputed when there 3 total but reference data only matches 2
        y_test = imputing.impute_data(
            reference_col="data2",
            target_col="data1",
            reference_data=self.test4_df,
            target_data=self.test5_df,
            align_col="align",
        )
        nptest.assert_array_almost_equal(y_test.loc[["c", "d"]], np.array([2.0, 4.0]), decimal=4)

        # Test 5a: Invalid reference data column
        with self.assertRaises(ValueError):
            imputing.impute_data(
                target_data=self.test5_df,
                target_col="data1",
                reference_data=self.test7_df,
                reference_col="data2",
                align_col="align",
            )

        # Test 5b: Invalid target data column
        with self.assertRaises(ValueError):
            imputing.impute_data(
                target_data=self.test5_df,
                target_col="data2",
                reference_data=self.test7_df,
                reference_col="data1",
                align_col="align",
            )

        # Test 5c: Bad data inputs
        with self.assertRaises(TypeError):
            imputing.impute_data(
                target_data=self.test5_df.to_numpy(),
                target_col="data1",
                reference_data=self.test7_df.to_numpy(),
                reference_col="data2",
                align_col="align",
            )

        # Test 5c: Invalid alignment column
        with self.assertRaises(ValueError):
            imputing.impute_data(
                target_data=self.test5_df,
                target_col="data1",
                reference_data=self.test7_df,
                reference_col="data2",
                align_col="index",
            )

    def test_impute_all_assets_by_correlation(self):
        # Test 1, pass data frame with three highly correlated assets, ensure all NaN data are imputed in
        # final output
        y_test = imputing.impute_all_assets_by_correlation(
            self.test11_df, "data", "data", r2_threshold=0.7
        ).to_frame()
        y = pd.Series([0.440789, 3.401316, 14.3677, 42.8312, 62.887218, 96.734818])
        nptest.assert_array_almost_equal(
            (y_test.loc[self.test11_df["data"].isnull(), "imputed_data"]).values,
            y.values,
            decimal=4,
        )

        # Test 2, 3 highly correlated assets with less data, such that asset 'b' has no data imputed
        y = pd.Series([1.589147, np.nan, np.nan, np.nan, np.nan, np.nan, 123.7000])
        y_test = imputing.impute_all_assets_by_correlation(
            self.test10_df, "data", "data", r2_threshold=0.7
        ).to_frame()
        nan_ind = self.test10_df.loc[self.test10_df["data"].isnull()].index
        nptest.assert_array_almost_equal(y_test.loc[nan_ind, "imputed_data"], y, decimal=4)

        # Test 3, 2 poorly correlated data sets, no data should be imputed
        y_test = imputing.impute_all_assets_by_correlation(
            self.test12_df, "data", "data", r2_threshold=0.7
        ).to_frame()
        nptest.assert_array_almost_equal(y_test["imputed_data"], self.test12_df["data"], decimal=4)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
