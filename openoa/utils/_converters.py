"""
This is module for common data conversion and checking methods and decorators that are used
throughout the utils subpackage.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Callable
from inspect import getfullargspec
from tkinter import N
from operator import setitem
from functools import wraps
from itertools import filterfalse
from multiprocessing.sharedctypes import Value

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
    corresponding to the column names. If `None` is passed as an argument to args, then `None` will
    be returned in place of a Series to maintain consistent ordering of inputs and outputs.

    Args:
        data (obj:`pandas.DataFrame`): The `DataFrame` object containg the column names in `args`.
        args(obj:`str`): A dynamic number of strings that make up the column names that need to be
            returned back as pandas `Series` objects.

    Raises:
        ValueError: Raised if `data` is not a pandas `DataFrame`.
        ValueError: Raised if any of the args passed is not contained in `data`.

    Returns:
        tuple[pandas.Series, ...]: A pandas `Series` for each of the column names passed in `args`
    """
    if len(args) == 0:
        raise ValueError("No column names provided to args for conversion to Series objects.")

    series_args = [isinstance(arg, pd.Series) for arg in args]
    if data is None:
        if all(el or isinstance(arg, type(None)) for el, arg in zip(series_args, args)):
            return args
        raise ValueError("No input provided to `data`; cannot convert args to Series.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The input to `data` must be a pandas `DataFrame`.")

    if any(series_args):
        raise TypeError(
            "When `data` is passed, all data column arguments must be the name of the column in `data`, and not a pandas Series."
        )

    # Check for valid column names in args, ignoring any None values
    if len(invalid := set(filterfalse(lambda x: x is None, args)).difference(data.columns)) > 0:
        raise ValueError(f"The following args are not columns of `data`: {invalid}")
    return tuple(data.loc[:, col].copy() if col is not None else col for col in args)


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
        raise TypeError("At least one of the provided values was not a pandas DataFrame")
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


def series_method(data_cols: list[str] = None):
    """Wrapper method for methods that operate on pandas `Series`, and not `DataFrame`s that allows
    the passing of column names that are potentially contained in a pandas `DataFrame` to be pulled
    out as separate pandas `Series` objects to be passed back to the method. This is a convenience
    wrapper that reduces the amount of boilerplate required to enable both `DataFrame` and `Series`
    arguments to be used interchangably in a single method's API.

    Args:
        data_cols (list[str], optional): The names of the method arguments that should be converted
            from `str` to pandas `Series` when `data` is provided as a pandas `DataFrame` to the
            focal method. Defaults to None.
    """

    def decorator(func: Callable):
        """Gathes the arg indices from `data_cols` to be used in `wrapper`."""
        argspec = getfullargspec(func)
        arg_ix_list = []
        if data_cols is not None:
            arg_ix_list = [argspec.args.index(name) for name in data_cols]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            """Returns the results of `func` after converting the arguments as needed."""
            if no_df := (df := kwargs.get("data", None)) is None:
                if arg_ix_list == []:
                    # Let the original method handle the provided arguments if wrapper not configured
                    return func(*args, *kwargs)
            args = list(args)
            arg_list = []
            for ix, name in zip(arg_ix_list, data_cols):
                try:
                    arg_list.append(args[ix])
                except IndexError:
                    # Check that the argument isn't in fact an optional keyword argument
                    try:
                        arg_list.append(kwargs[name])
                    except KeyError:
                        # Insert a None if no arg/kwarg is found, which is due to the use of unpassed
                        # kwarg argument that is defaulted to None
                        arg_list.append(None)
            new_args = df_to_series(df, *arg_list)
            for ix, name, new in zip(arg_ix_list, data_cols, new_args):
                try:
                    setitem(args, ix, new)
                except IndexError:
                    try:
                        kwargs[name] = new
                    except KeyError:
                        # No need to pass through a non-existent input that is defaulted to None
                        pass
            if not no_df:
                kwargs.pop("data")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def dataframe_method(data_cols: list[str] = None):
    """Wrapper method for methods that operate on a pandas `DataFrame`, and not `Series` that allows
    the passing of the `Series`, so that they can be combined in a `DataFrame`and passed back to the
    method. This is a convenience wrapper that reduces the amount of boilerplate required to enable
    both `DataFrame` and `Series` arguments to be used interchangably in a single method's API.

    Args:
        data_cols (list[str], optional): The names of the method arguments that should be converted
            from a pandas `Series` to `str`, along with the creation of the `data` keyword argument
            when the column data is passed as a `Series`. Defaults to None.
    """

    def decorator(func: Callable):
        """Gathes the arg indices from `data_cols` to be used in `wrapper`."""
        argspec = getfullargspec(func)
        arg_ix_list = []
        if data_cols is not None:
            arg_ix_list = [argspec.args.index(name) for name in data_cols]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            """Returns the results of `func` after converting the arguments as needed."""
            if kwargs["data"] is not None and arg_ix_list == []:
                return func(*args, *kwargs)
            args = list(args)
            df = series_to_df(*(args[ix] for ix in arg_ix_list))
            _ = [setitem(args, ix, args[ix].name) for ix in arg_ix_list]
            kwargs["data"] = df
            return func(*args, **kwargs)

        return wrapper

    return decorator
