"""Provides a number of methods and bases classes to keep the openoa/plant.py manageable."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence
from pathlib import Path

import attr
import attrs
import numpy as np
import pandas as pd
import pyspark as spark
from attrs import define


@define(auto_attribs=True)
class FromDictMixin:
    """A Mixin class to allow for kwargs overloading when a data class doesn't
    have a specific parameter definied. This allows passing of larger dictionaries
    to a data class without throwing an error.

    Raises
    ------
    AttributeError
        Raised if the required class inputs are not provided.
    """

    @classmethod
    def from_dict(cls, data: dict):
        """Maps a data dictionary to an `attrs`-defined class.
        TODO: Add an error to ensure that either none or all the parameters are passed in
        Args:
            data : dict
                The data dictionary to be mapped.
        Returns:
            cls
                The `attrs`-defined class.
        """
        # Get all parameters from the input dictionary that map to the class initialization
        kwargs = {
            a.name: data[a.name]
            for a in cls.__attrs_attrs__  # type: ignore
            if a.name in data and a.init
        }

        # Map the inputs must be provided: 1) must be initialized, 2) no default value defined
        required_inputs = [
            a.name
            for a in cls.__attrs_attrs__  # type: ignore
            if a.init and isinstance(a.default, attr._make._Nothing)  # type: ignore
        ]
        undefined = sorted(set(required_inputs) - set(kwargs))
        if undefined:
            raise AttributeError(
                f"The class defintion for {cls.__name__} is missing the following inputs: {undefined}"
            )
        return cls(**kwargs)  # type: ignore


def frequency_validator(
    actual_freq: str, desired_freq: Optional[str | None | set[str]], exact: bool
) -> bool:
    """Helper function to check if the actual datetime stamp frequency is valid compared
    to what is required.

    Args:
        actual_freq (str): The frequency of the datetime stamp, or `df.index.freq`.
        desired_freq (Optional[str  |  None  |  set[str]]): Either the exact frequency,
            required or a set of options that are also valid, in which case any numeric
            information encoded in `actual_freq` will be dropped.
        exact (bool): If the provided frequency codes should be exact matches (`True`),
            or, if `False`, the check should be for a combination of matches.

    Returns:
        bool: If the actual datetime frequency is sufficient, per the match requirements.
    """
    if desired_freq is None:
        return True

    if actual_freq is None:
        return False

    if isinstance(desired_freq, str):
        desired_freq = set([desired_freq])

    if exact:
        return actual_freq in desired_freq

    actual_freq = "".join(filter(str.isalpha, actual_freq))
    return actual_freq in desired_freq


def convert_to_list(
    value: Sequence | str | int | float | None,
    manipulation: Callable | None = None,
) -> list:
    """Converts an unknown element that could be a list or single, non-sequence element
    to a list of elements.

    Parameters
    ----------
    value : Sequence | str | int | float
        The unknown element to be converted to a list of element(s).
    manipulation: Callable | None
        A function to be performed upon the individual elements, by default None.

    Returns
    -------
    list
        The new list of elements.
    """

    if isinstance(value, (str, int, float)) or value is None:
        value = [value]
    if manipulation is not None:
        return [manipulation(el) for el in value]
    return list(value)


def column_validator(df: pd.DataFrame, column_names={}) -> None | list[str]:
    """Validates that the column names exist as provided for each expected column.

    Args:
        df (pd.DataFrame): The DataFrame for column naming validation
        column_names (dict, optional): Dictionary of column type (key) to real column
            value (value) pairs. Defaults to {}.

    Returns:
        None | list[str]: A list of error messages that can be raised at a later step
            in the validation process.
    """
    try:
        missing = set(column_names.values()).difference(df.columns)
    except AttributeError:
        # Catches 'NoneType' object has no attribute 'columns' for no data
        missing = column_names.values()
    if missing:
        return list(missing)
    return []


def dtype_converter(df: pd.DataFrame, column_types={}) -> list[str]:
    """Converts the columns provided in `column_types` of `df` to the appropriate data
    type.

    Args:
        df (pd.DataFrame): The DataFrame for type validation/conversion
        column_types (dict, optional): Dictionary of column name (key) and data type
            (value) pairs. Defaults to {}.

    Returns:
        None | list[str]: List of error messages that were encountered in the conversion
            process that will be raised at another step of the data validation.
    """
    errors = []
    for column, new_type in column_types.items():
        if new_type in (np.datetime64, pd.DatetimeIndex):
            try:
                df[column] = pd.DatetimeIndex(df[column])
            except Exception as e:  # noqa: disable=E722
                errors.append(column)
            continue
        try:
            df[column] = df[column].astype(new_type)
        except:  # noqa: disable=E722
            errors.append(column)

    return errors


def load_to_pandas(data: str | Path | pd.DataFrame | spark.sql.DataFrame) -> pd.DataFrame | None:
    """Loads the input data or filepath to apandas DataFrame.

    Args:
        data (str | Path | pd.DataFrame | spark.DataFrame): The input data.

    Raises:
        ValueError: Raised if an invalid data type was passed.

    Returns:
        pd.DataFrame | None: The passed `None` or the converted pandas DataFrame object.
    """
    if data is None:
        return data
    elif isinstance(data, (str, Path)):
        return pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, spark.sql.DataFrame):
        return data.toPandas()
    else:
        raise ValueError("Input data could not be converted to pandas")


def load_to_pandas_dict(
    data: dict[str | Path | pd.DataFrame | spark.sql.DataFrame],
) -> dict[str, pd.DataFrame] | None:
    """Converts a dictionary of data or data locations to a dictionary of `pd.DataFrame`s
    by iterating over the dictionary and passing each value to `load_to_pandas`.

    Args:
        data (dict[str  |  Path  |  pd.DataFrame  |  spark.sql.DataFrame]): The input data.

    Returns:
        dict[str, pd.DataFrame] | None: The passed `None` or the converted `pd.DataFrame`
            object.
    """
    if data is None:
        return data
    for key, val in data.items():
        data[key] = load_to_pandas(val)
    return data


def rename_columns(df: pd.DataFrame, col_map: dict, reverse: bool = True) -> pd.DataFrame:
    """Renames the pandas DataFrame columns using col_map. Intended to be used in
    conjunction with the a data objects meta data column mapping (reverse=True).

        Args:
            df (pd.DataFrame): The DataFrame to have its columns remapped.
            col_map (dict): Dictionary of existing column names and new column names.
            reverse (bool, optional): True, if the new column names are the keys (using the
                xxMetaData.col_map as input), or False, if the current column names are the
                values (original column names). Defaults to True.

        Returns:
            pd.DataFrame: Input DataFrame with remapped column names.
    """
    if not reverse:
        col_map = {v: k for k, v in col_map.items()}
    return df.rename(columns=col_map)
