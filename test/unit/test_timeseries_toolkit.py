import unittest
from datetime import datetime

import numpy as np
import pandas as pd
from pytz import timezone
from numpy import testing as nptest

from operational_analysis.toolkits import timeseries


class SimpleTimeseriesTests(unittest.TestCase):
    def setUp(self):
        self.mountain_tz = timezone("US/Mountain")
        self.pacific_tz = timezone("US/Pacific")
        self.summer_midnight = datetime(2018, 0o7, 16, 0, 0, 0)
        self.winter_midnight = datetime(2018, 0o1, 11, 0, 0, 0)
        self.day_of_data = pd.Series(
            pd.date_range(start="1/1/2018 00:00:00", end="1/1/2018 23:59:59", freq="10min")
        )
        self.two_days_of_data = self.day_of_data.append(
            pd.Series(
                pd.date_range(start="2/1/2018 00:00:00", end="2/1/2018 23:59:59", freq="10min")
            )
        )

    def test_convert_local_to_utc(self):
        # Pass in a localized datetime with matching tz string and make sure it throws an exception
        self.assertRaises(
            Exception,
            self.mountain_tz.localize(self.summer_midnight),
            "T1: No exception raised for a datetime object with baked in TZInfo",
        )

        # Pass in a non-localized datetime with tz string
        mm_utc = timeseries.convert_local_to_utc(self.summer_midnight, "US/Pacific")
        hours_diff = self.summer_midnight.hour - mm_utc.hour
        # PDT is UTC -7
        self.assertTrue(hours_diff == -7, "T2: PDT is not UTC -7?")

        # Pass in a non-localized winter datetime with tz string
        mm_utc = timeseries.convert_local_to_utc(self.winter_midnight, "US/Mountain")
        hours_diff = self.summer_midnight.hour - mm_utc.hour
        # MST is UTC -7
        self.assertTrue(hours_diff == -7, "T3: MST is not UTC -7?")

    def test_find_time_gaps(self):
        # A full day worth of data has zero gaps
        day_of_data = self.day_of_data
        no_gaps = timeseries.find_time_gaps(day_of_data, "10min")
        self.assertEqual(no_gaps.size, 0, "T1: Something with no gaps was reported to have gaps")

        # Removing two gaps should result is a result size of two
        missing_two = day_of_data.drop([2, 3])
        two_gaps = timeseries.find_time_gaps(missing_two, "10min")
        self.assertEqual(two_gaps.size, 2, "T2: Did not properly detect two gaps in 10M timeseries")

        # Shuffling the above series should maintain the same number of gaps
        shuffled_missing_two = pd.Series(np.random.permutation(missing_two))
        two_gaps = timeseries.find_time_gaps(shuffled_missing_two, "10min")
        self.assertEqual(
            two_gaps.size, 2, "T3: Did not properly detect two gaps in shuffled 10M timeseries"
        )

        # An empty series has zero gaps
        empty_series = pd.Series(dtype=np.float64)
        no_gaps = timeseries.find_time_gaps(empty_series, "10min")
        self.assertEqual(no_gaps.size, 0, "T4: Empty series should have zero gaps")

    def test_find_duplicate_times(self):
        # Manually set one row to another and detect it
        day_of_data = self.day_of_data.copy()
        day_of_data[1] = day_of_data[2]
        dupes = timeseries.find_duplicate_times(day_of_data, "10min")
        self.assertEqual(dupes.size, 1, "T1: Detect one duplicated row")

        # Input series of length zero
        day_of_data = pd.Series(dtype=np.float64)
        dupes = timeseries.find_duplicate_times(day_of_data, "10min")
        self.assertEqual(dupes.size, 0, "T2: Empty series should have zero duplicates")

    def test_gap_fill_data_frame(self):
        # df with a gap
        day_of_data = self.day_of_data.copy()
        missing_two = day_of_data.drop([2, 3])
        missing_two_df = pd.DataFrame({"time": missing_two, "col1": missing_two})
        filled = timeseries.gap_fill_data_frame(missing_two_df, "time", "10min")
        self.assertEqual(
            day_of_data.size,
            filled["time"].size,
            "T1: Gap filling should increase size of this dataframe",
        )

        # df with no gaps
        day_of_data = self.day_of_data.copy()
        day_of_data_df = pd.DataFrame({"time": day_of_data, "col1": day_of_data})
        filled = timeseries.gap_fill_data_frame(day_of_data_df, "time", "10min")
        self.assertEqual(
            filled["time"].size, day_of_data.size, "T2: Full series should not have any new members"
        )

        # empty input df
        empty = pd.Series(dtype=np.float64)
        empty_df = pd.DataFrame({"time": empty, "col1": empty})
        filled = timeseries.gap_fill_data_frame(empty_df, "time", "10min")
        self.assertEqual(filled["time"].size, 0, "T3: Empty dataframe should still be empty")

    def test_num_days(self):
        # Test 1 day of data
        day_of_data = pd.DataFrame(index=self.day_of_data)
        num = timeseries.num_days(day_of_data)
        self.assertEqual(num, 1, "One day of data...")

        # Test 0 days of data
        empty_data = pd.DataFrame(index=pd.DatetimeIndex([]))
        num = timeseries.num_days(empty_data)
        self.assertEqual(num, 0, "Zero days of data...")

        # Test 2 days of data separated by a month gap
        two_days = pd.DataFrame(index=self.two_days_of_data)
        num = timeseries.num_days(two_days)
        self.assertEqual(num, 32, "Two days of data separated by a month...")

    def test_num_hours(self):
        # Test 1 day of data
        day_of_data = pd.DataFrame(index=self.day_of_data)
        num = timeseries.num_hours(day_of_data)
        self.assertEqual(num, 24, "One day of data...")

        # Test 0 days of data
        empty_data = pd.DataFrame(index=pd.DatetimeIndex([]))
        num = timeseries.num_hours(empty_data)
        self.assertEqual(num, 0, "Zero days of data...")

        # Test 2 days of data separated by a month gap
        two_days = pd.DataFrame(index=self.two_days_of_data)
        num = timeseries.num_hours(two_days)
        self.assertEqual(num, 32 * 24, "Two days of data separated by a month...")

    def test_percent_nan(self):
        test_dict = {}
        test_dict["a"] = pd.Series(["", 1, 2, 1e5, np.Inf])
        test_dict["b"] = pd.Series(["", np.nan, 2, 1e5, np.Inf])
        test_dict["c"] = pd.Series([np.nan, 1, 2, 1e5, np.nan])

        nan_values = {"a": 0.0, "b": 0.2, "c": 0.4}

        for a, b in test_dict.items():
            nptest.assert_almost_equal(
                nan_values[a],
                timeseries.percent_nan(test_dict[a]),
                err_msg="NaN percentage function is broken",
            )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
