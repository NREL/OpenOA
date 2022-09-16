from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy import testing as nptest
from pandas import testing as tm

from openoa.utils._converters import (
    _list_of_len,
    df_to_series,
    series_to_df,
    series_method,
    dataframe_method,
    convert_args_to_lists,
    multiple_df_to_single_df,
)


test_df1 = pd.DataFrame(
    np.arange(15).reshape(5, 3, order="F"), columns=["a", "b", "c"], dtype=float
)
test_series_a1 = pd.Series(range(5), name="a", dtype=float)
test_series_b1 = pd.Series(range(5, 10), name="b", dtype=float)
test_series_c1 = pd.Series(range(10, 15), name="c", dtype=float)
test_df_list1 = [test_series_a1.to_frame(), test_series_b1.to_frame(), test_series_c1.to_frame()]

test_df2 = test_df1.copy()
test_df2.loc[5] = [15.0, np.nan, np.nan]
test_series_a2 = test_series_a1.copy()
test_series_a2.loc[5] = 15.0
test_df_list2 = [test_series_a2.to_frame(), test_series_b1.to_frame(), test_series_c1.to_frame()]

test_df3 = test_df2.copy()
test_df3.index.name = "index"
test_align_col3 = "index"
test_df_list3 = [
    test_df3[["a"]].reset_index(drop=False),
    test_df3[["b"]].reset_index(drop=False),
    test_df3[["c"]].reset_index(drop=False),
]


@series_method(data_cols=["col1", "col2"])
def sample_series_handling_method(
    col1: pd.Series | str, x: float, y: float, col2: pd.Series | str, data: pd.DataFrame = None
) -> tuple[pd.Series, pd.Series, None]:
    """A method that returns the column and data objects to ensure correctness of the wrapper."""
    return col1, col2, data


@dataframe_method(data_cols=["col_a", "col_b"])
def sample_df_handling_method(
    col_a: pd.Series | str, x: float, col_b: pd.Series | str, y: float, data: pd.DataFrame = None
) -> tuple[str, str, pd.DataFrame]:
    """A method that returns the column and data objects to ensure correctness of the wrapper."""
    return col_a, col_b, data


def test_list_of_len():
    """Tests the `_list_of_len` method."""

    # Test for a 1 element list
    x = [1]
    length = 4
    y = [1, 1, 1, 1]
    y_test = _list_of_len(x, length)
    assert y == y_test

    # Test for a multi-element list with the length as a multiple of the length of the list
    x = ["a", "b", "C"]
    length = 6
    y = ["a", "b", "C", "a", "b", "C"]
    y_test = _list_of_len(x, length)
    assert y == y_test

    # Test for a multi-element list that should have a resulting unequal number of repeated elements
    x = [1, "4", 7]
    length = 4
    y = [1, "4", 7, 1]
    y_test = _list_of_len(x, length)
    assert y == y_test


def test_convert_args_to_lists():
    """Tests the `convert_args_to_lists` method."""

    # Test for a list that already contains lists
    x = [["a"], [5, 6]]
    length = 2
    y = [["a"], [5, 6]]
    y_test = convert_args_to_lists(length, *x)
    assert y == y_test

    # Test for a list of single element arguments
    x = ["a", 5, 6]
    length = 2
    y = [["a", "a"], [5, 5], [6, 6]]
    y_test = convert_args_to_lists(length, *x)
    assert y == y_test

    # Test for list of mixed length elements
    x = ["a", 5, [6]]
    length = 2
    y = [["a", "a"], [5, 5], [6]]
    y_test = convert_args_to_lists(length, *x)
    assert y == y_test


def test_df_to_series():
    """Tests the `df_to_series` method."""

    # Test that each column is returned correctly, in a few different order variations
    y = [test_series_a1, test_series_b1, test_series_c1]
    y_test = df_to_series(test_df1, "a", "b", "c")
    for el, el_test in zip(y, y_test):
        tm.assert_series_equal(el, el_test)

    y = [test_series_c1, test_series_a1]
    y_test = df_to_series(test_df1, "c", "a")
    for el, el_test in zip(y, y_test):
        tm.assert_series_equal(el, el_test)

    y = [test_series_b1]
    y_test = df_to_series(test_df1, "b")
    for el, el_test in zip(y, y_test):
        tm.assert_series_equal(el, el_test)

    # Test that None is returned for a passed None value
    y = [None, test_series_a1, None, test_series_b1, test_series_c1, None]
    y_test = df_to_series(test_df1, None, "a", None, "b", "c", None)
    for el, el_test in zip(y, y_test):
        if el is None:
            assert el_test is None
            continue
        tm.assert_series_equal(el, el_test)

    # Test for bad inputs `data`
    with pytest.raises(TypeError):
        df_to_series(test_series_a1, "a")

    with pytest.raises(TypeError):
        df_to_series(4, "a")

    # Test for a series proved to `args`
    with pytest.raises(TypeError):
        df_to_series(test_df1, "a", test_series_b1)

    # Test for missing columns
    with pytest.raises(ValueError):
        df_to_series(test_df1, "a", "d")


def test_multiple_df_to_single_df():
    """Tests the `multiple_df_to_single_df` method."""

    # Test a basic working case with single column DataFrames
    y = test_df1
    y_test = multiple_df_to_single_df(*test_df_list1)
    tm.assert_frame_equal(y, y_test)

    # Test a basic working case with different length inputs
    y = test_df2
    y_test = multiple_df_to_single_df(*test_df_list2)
    tm.assert_frame_equal(y, y_test)

    # Test a basic working case with an `align_col` argument
    y = test_df3
    y_test = multiple_df_to_single_df(*test_df_list3, align_col=test_align_col3)
    tm.assert_frame_equal(y, y_test)
    assert test_df2.index.name is None and y_test.index.name == "index"
    tm.assert_frame_equal(test_df2, y_test, check_names=False)

    # Ensure non DataFrame arguments fail
    with pytest.raises(TypeError):
        multiple_df_to_single_df(test_series_a1, *test_df_list1)

    # Check that a missing `align_col` fails
    with pytest.raises(ValueError):
        multiple_df_to_single_df(*test_df_list3, align_col="Index")


def test_series_to_df():
    """Tests the `series_to_df` method."""

    # Test simple use case
    y = test_df1
    y_test, (y_test_a1, y_test_b1, y_test_c1) = series_to_df(
        test_series_a1, test_series_b1, test_series_c1
    )
    tm.assert_frame_equal(y, y_test)
    assert y_test_a1 == test_series_a1.name
    assert y_test_b1 == test_series_b1.name
    assert y_test_c1 == test_series_c1.name

    # Ensure nans get inserted appropriately
    y = test_df2
    y_test, (y_test_a2, y_test_b2, y_test_c2) = series_to_df(
        test_series_a2, test_series_b1, test_series_c1
    )
    tm.assert_frame_equal(y, y_test)
    assert y_test_a2 == test_series_a2.name
    assert y_test_b2 == test_series_b1.name
    assert y_test_c2 == test_series_c1.name

    # Ensure all invalid inputs will fail
    with pytest.raises(TypeError):
        series_to_df(test_series_a1, test_series_b1, "c")

    # Ensure one input series maps correctly
    for x, y in zip([test_series_a1, test_series_b1, test_series_c1], test_df_list1):
        y_test, [name] = series_to_df(x)
        tm.assert_frame_equal(y, y_test)
        assert name == x.name


def test_series_method():
    """Tests the `series_method` wrapper via `sample_series_handling_method()`."""

    # Ensure that the wrapper converts the string column names to series objects and the data kwarg to None
    y_test_a, y_test_c, y_test_df = sample_series_handling_method("a", 1.0, 2.0, "c", data=test_df1)
    tm.assert_series_equal(test_series_a1, y_test_a)
    tm.assert_series_equal(test_series_c1, y_test_c)
    assert y_test_df is None

    # Ensure a None column argument gets handled correctly
    y_test_a, y_test_none, y_test_df = sample_series_handling_method(
        "a", 1.0, 2.0, None, data=test_df1
    )
    tm.assert_series_equal(test_series_a1, y_test_a)
    assert y_test_none is None
    assert y_test_df is None

    # Ensure invalid arguments are handle correctly
    with pytest.raises(TypeError):
        sample_series_handling_method(test_series_a1, 1.0, 2.0, "c", data=test_df1)

    with pytest.raises(ValueError):
        sample_series_handling_method("a", 1.0, 2.0, "c")


def test_dataframe_method():
    """Tests the `series_method` wrapper via `sample_df_handling_method()`."""

    # Ensure that the wrapper converts the Series to column names and the data to a DataFrame
    y = test_df1[["c", "a"]]
    y_test_c, y_test_a, y_test_df = sample_df_handling_method(
        test_series_c1,
        1.0,
        test_series_a1,
        2.0,
    )
    tm.assert_frame_equal(y, y_test_df)
    assert y_test_a == "a"
    assert y_test_c == "c"

    # Ensure that the wrapper converts the series to series.name when a DataFrame is passed
    y = test_df1[["c", "a"]]
    y_test_c, y_test_a, y_test_df = sample_df_handling_method(
        test_series_c1, 1.0, test_series_a1, 2.0, data=test_df1
    )
    tm.assert_frame_equal(test_df1, y_test_df)
    assert y_test_a == "a"
    assert y_test_c == "c"

    # Ensure nothing changes when names and dataframes are passed
    y_test_c, y_test_a, y_test_df = sample_df_handling_method(
        "c", 1.0, test_series_a1, 2.0, data=test_df1
    )
    tm.assert_frame_equal(test_df1, y_test_df)
    assert y_test_a == "a"
    assert y_test_c == "c"

    # Ensure that the wrapper converts the series to series.name when a DataFrame is passed
    y_test_c, y_test_a, y_test_df = sample_df_handling_method("c", 1.0, "a", 2.0, data=test_df1)
    tm.assert_frame_equal(test_df1, y_test_df)
    assert y_test_a == "a"
    assert y_test_c == "c"

    # Ensure that the wrapper converts the series to series.name when a DataFrame is passed
    y = test_df1[["c", "a"]]
    y_test_c, y_test_a, y_test_df = sample_df_handling_method(
        test_series_c1, 1.0, test_series_a1, 2.0, data=test_df1
    )
    tm.assert_frame_equal(test_df1, y_test_df)
    assert y_test_a == "a"
    assert y_test_c == "c"

    # Check for failure on inconsistent data passing when no dataframe is provided
    with pytest.raises(TypeError):
        sample_df_handling_method("c", 1.0, test_series_c1, 2.0)

    # Check for failure when an invalid column name is passed
    with pytest.raises(ValueError):
        sample_df_handling_method("d", 1.0, "a", 2.0, data=test_df1)
