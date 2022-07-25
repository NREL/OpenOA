import attr
import numpy as np
import pandas as pd
import pytest

from openoa import plant


# Test all the standalone utility/helper methods


def test_analysis_type_validator() -> None:
    """Tests the `PlantData.analysis_type` validator method: `analysis_type_validator`.
    The expected input for this method is `list[str | None]`, and all values that feed
    into are converted into this format, so no tests will exist for checking the input
    format.
    """

    instance = plant.PlantData
    attribute = attr.fields(plant.PlantData).analysis_type

    # Test the acceptance of the valid inputs, except None
    valid = [*plant.ANALYSIS_REQUIREMENTS] + ["all"]
    assert plant.analysis_type_validator(instance, attribute, valid) is None

    # Test that None is valid, but raises a warning
    with pytest.warns(UserWarning):
        plant.analysis_type_validator(instance, attribute, [None])

    # Test that all valid cases work
    valid.append(None)
    assert plant.analysis_type_validator(instance, attribute, valid) is None

    # Test a couple of edge cases to show that only valid inputs are allowed
    # Add a mispelling
    with pytest.raises(ValueError):
        plant.analysis_type_validator(instance, attribute, valid + ["Montecarloaep"])

    # Add a completely wrong value
    with pytest.raises(ValueError):
        plant.analysis_type_validator(instance, attribute, valid + ["this is wrong"])


def test_frequency_validator() -> None:
    """Tests the `frequency_validator` method. All inputs are formatted to the desired input,
    so testing of the input types is not required.
    """

    # Test None as desired frequency returns True always
    assert plant.frequency_validator("anything", None, exact=True)
    assert plant.frequency_validator("anything", None, exact=False)
    assert plant.frequency_validator(None, None, exact=True)
    assert plant.frequency_validator(None, None, exact=False)

    # Test None as actual frequency returns False as long as desired isn't also True (checked above)
    assert not plant.frequency_validator(None, "anything", exact=True)
    assert not plant.frequency_validator(None, "whatever", exact=False)

    # Test for exact matches
    actual = "10T"
    desired_valid_1 = "10T"  # single input case
    desired_valid_2 = ("10T", "H", "N")  # set of options case
    desired_invalid = plant._at_least_hourly  # set of non exact matches

    assert plant.frequency_validator(actual, desired_valid_1, True)
    assert plant.frequency_validator(actual, desired_valid_2, True)
    assert not plant.frequency_validator(actual, desired_invalid, True)

    # Test for non-exact matches
    actual_1 = "10T"
    actual_2 = "1min"
    actual_3 = "20S"
    desired_valid = plant._at_least_hourly  # set of generic hourly or higher resolution frequencies
    desired_invalid = (
        "M",
        "MS",
        "W",
        "D",
        "H",
    )  # set of greater than or equal to hourly frequency resolutions

    assert plant.frequency_validator(actual_1, desired_valid, False)
    assert plant.frequency_validator(actual_2, desired_valid, False)
    assert plant.frequency_validator(actual_3, desired_valid, False)
    assert not plant.frequency_validator(actual_1, desired_invalid, False)
    assert not plant.frequency_validator(actual_2, desired_invalid, False)
    assert not plant.frequency_validator(actual_3, desired_invalid, False)


def test_convert_to_list():
    """Tests the converter function for turning single inputs into a list of input,
    or applying a manipulation across a list of inputs
    """

    # Test that a list of the value is returned
    assert plant.convert_to_list(1) == [1]
    assert plant.convert_to_list(None) == [None]
    assert plant.convert_to_list("input") == ["input"]
    assert plant.convert_to_list(42.8) == [42.8]

    # Test that the same list is returned
    assert plant.convert_to_list(range(3)) == [0, 1, 2]
    assert plant.convert_to_list([44, "six", 1.2]) == [44, "six", 1.2]
    assert plant.convert_to_list((44, "six", 1.2)) == [44, "six", 1.2]

    # Test that an invalid type is passed to the manipulation argument
    with pytest.raises(ValueError):
        assert plant.convert_to_list(range(3), 1)

    # Test that lists of mixed inputs error out with a type-specific converter function
    with pytest.raises(ValueError):
        assert plant.convert_to_list(range(3), str.upper)

    with pytest.raises(ValueError):
        assert plant.convert_to_list(["?", "one", "string", 2], float)

    # Test that valid manipulations work
    assert plant.convert_to_list(range(3), float) == [0.0, 1.0, 2.0]
    assert plant.convert_to_list([1.1, 2.2, 3.9], int) == [1, 2, 3]
    assert plant.convert_to_list(["loud", "noises"], str.upper) == ["LOUD", "NOISES"]
    assert plant.convert_to_list(["quiet", "VOICes"], str.lower) == ["quiet", "voices"]


def column_validator():
    """Tests the `plant.column_validator` method to ensure dataframes contain all of the
    required columns.
    """
    df = pd.DataFrame([range(4)], columns=[f"col{i}" for i in range(1, 5)])

    # Test for a complete match
    col_map = {f"new_col_{i}": f"col{i}" for i in range(1, 5)}
    assert plant.column_validator(df, column_names=col_map) == []

    # Test for a partial match with extra columns causing no issue at all
    col_map = {f"new_col_{i}": f"col{i}" for i in range(1, 3)}
    assert plant.column_validator(df, column_names=col_map) == []

    # Test for an incomplete match
    col_map = {f"new_col_{i}": f"col{i}" for i in range(1, 6)}
    assert plant.column_validator(df, column_names=col_map) == ["col5"]


def test_dtype_converter():
    """Tests the `plant.dtype_converter` method to ensure that columns get converted
    correctly or return a list of error columns. This assumes that datetime columns
    have already been converted to pandas datetime objects in the reading methods.
    """
    df = pd.DataFrame([], columns=["time", "float_col", "string_col", "problem_col"])
    df.time = pd.date_range(start="2022-July-25 00:00:00", end="2022-July-25 1:00:00", freq="10T")
    df.float_col = np.random.random(7).astype(str)
    df.string_col = np.arange(7)
    df.problem_col = ["one", "two", "string", "invalid", 5, 6.0, 7]

    column_types_invalid_1 = dict(
        time=pd.DatetimeIndex, float_col=float, string_col=str, problem_col=float
    )
    column_types_invalid_2 = dict(
        time=np.datetime64, float_col=float, string_col=str, problem_col=int
    )
    column_types_valid = dict(time=np.datetime64, float_col=float, string_col=str, problem_col=str)

    assert plant.dtype_converter(df, column_types_invalid_1) == ["problem_col"]
    assert plant.dtype_converter(df, column_types_invalid_2) == ["problem_col"]
    assert plant.dtype_converter(df, column_types_valid) == []


def test_analysis_filter():
    # Save for later
    pass


def test_compose_error_message():
    # Potentially goes with test_analysis_filter, but passing on them both for now
    pass


def test_load_to_pandas():
    # Save for later
    pass


def test_load_to_pandas_dict():
    # Save for later
    pass


def test_rename_columns():
    """Tests the `plant.rename_columns` method for renaming dataframes."""
    df = pd.DataFrame([range(4)], columns=[f"col{i}" for i in range(1, 5)])

    # Test for a standard mapping
    col_map = {f"col{i}": f"new_col_{i}" for i in range(1, 5)}
    new_df = plant.rename_columns(df, col_map, reverse=False)
    assert new_df.columns.to_list() == list(col_map.values())

    # Test for the reverse case
    col_map = {f"new_col_{i}": f"col{i}" for i in range(1, 5)}
    new_df = plant.rename_columns(df, col_map, reverse=True)
    assert new_df.columns.to_list() == list(col_map.keys())


def test_FromDictMixin():
    pass
