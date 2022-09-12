"""
This is module for common data conversion and checking methods and decorators that are used
throughout the utils subpackage.
"""

from __future__ import annotations

from math import ceil

import pandas as pd


def _list_of_len(x: list, length: int) -> list:
    """Converts a list of dynamic length to one of length `length` by repeating the elements of `x`
    until the desired length is reached.

    Args:
        x (`list`): The list to be expanded.
        length (`int`): The desired length of `x`

    Returns:
        list: A list of length `length` with repeating elements of `x`.
    """
    if (actual := len(x)) == length:
        return x
    return (x * ceil(length / actual))[:length]


def convert_args_to_lists(length: int, *args) -> list[list]:
    """Convert method arguments to a list of length `length` for each argument passed.

    Args:
        length (int): The length of the argument list.
        args: A series of arguments to be converted series of individual lists of length `length` for
            each argument passed.

    Returns:
        list[list]: A list of lists of length `length` for each argument passed.
    """
    return [a if isinstance(a, list) else [a] * length for a in args]


def df_to_series(data: pd.DataFrame, *args: str) -> tuple[pd.Series, ...]:
    """Converts a `DataFrame` and dynamic number of column names to a an equal number of pandas `Series`
    corresponding to the column names.

    Args:
        data (obj:`pandas.DataFrame`): The `DataFrame` object containg the column names in `args`.
        args(obj:`str`): A dynamic number of strings that make up the column names that need to be
            returned back as pandas `Series` objects.

    Raises:
        ValueError: Raised if any of the args passed is not contained in `data`.

    Returns:
        tuple[pandas.Series, ...]: A pandas `Series` for each of the column names passed in `args`
    """
    if len(invalid := set(args).difference(data.columns)) > 0:
        raise ValueError(f"The following args are not columns of `data`: {invalid}")
    return tuple(data.loc[:, col].copy() for col in args)


def multiple_df_to_single_df(*args: pd.DataFrame, align_col: str | None = None) -> pd.DataFrame:
    """Convert multiple `DataFrames` to a single `DataFrame` either along the index, when `align_col`
    is None, or along the `align_col` when a value is provided.

    Args:
        args(:obj:`pandas.DataFrame`): A dynamic number of DataFrame inputs that need to be joined.
        align_col(:obj:`str` | `None`, optional): The common column shared among `args`, or the index,
            if `None`. Defaults to None.

    Raises:
        TypeError: Raised if any of the passed arguments isn't a pandas `DataFrame`
        ValueError: Raised if a value is provided to args does not contain the `align_col`.

    Returns:
        pd.DataFrame: _description_
    """
    if not all(isinstance(el, pd.DataFrame) for el in args):
        raise TypeError("At least one of the provided values was not a pandas Series")
    if align_col is not None:
        if not all(align_col in df for df in args):
            raise ValueError(
                f"At least of of the dataframes provided to *args does not contain the `align_col`: {align_col}"
            )
        args = [df.set_index(align_col) for df in args]

    return pd.concat(args, join="outer", axis=1)


def series_to_df(*args: pd.Series) -> pd.DataFrame:
    """Convert a dynamic number of pandas `Series` to a single pandas `DataFrame` by concatenating
    with an outer join, so the any missing values being filled with a NaN value, and each argument
    becomes a column of the resulting `DataFrame`.

    Args:
        args(:obj:`pandas.Series`): A series of of pandas `Series` objects that share a common axis.

    Returns:
        pd.DataFrame: A single data structure combining all the passed arguments.
    """
    if not all(isinstance(el, pd.Series) for el in args):
        raise TypeError("At least one of the provided values was not a pandas Series")
    args = [el.to_frame() for el in args]
    if len(args) > 1:
        return multiple_df_to_single_df(*args)
    return args[0]
