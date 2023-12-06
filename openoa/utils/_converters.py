"""
This is module for common data conversion and checking methods and decorators that are used
throughout the utils subpackage.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Type, Callable
from inspect import getfullargspec
from functools import wraps
from itertools import filterfalse

import pandas as pd


def _list_of_len(x: list, length: int) -> list:
    """Converts a list of dynamic length to one of length `length` by repeating the elements of :py:attr:`x`
    until the desired length is reached.

    Args:
        x (`list`): The list to be expanded.
        length (`int`): The desired length of :py:attr:`x`

    Returns:
        list: A list of length :py:attr:`length` with repeating elements of :py:attr:`x`.
    """
    if (actual := len(x)) == length:
        return x
    return (x * ceil(length / actual))[:length]


def _check_cols_in_df(data, *args):
    """Ensures that the columm names provided to :py:attr:`args` are contained in the pandas ``DataFrame``, :py:attr:`data`.

    Args:
        data (obj:`pandas.DataFrame`): The DataFrame object containing a dynamic-length list of
            columns contained in :py:attr:`args`.
        args : Desired column names of :py:attr:`data`, or ``None``.

    Raises:
        ValueError: Raised if one of the values provided to :py:attr:`args` is not column or ``None``.
    """
    if any(isinstance(arg, pd.Series) for arg in args):
        raise TypeError(
            "Some of the column arguments are Series, which cannot be DataFrame column names."
        )
    if len(invalid := set(filterfalse(lambda x: x is None, args)).difference(data.columns)) > 0:
        raise ValueError(f"The following args are not columns of `data`: {invalid}")


def _get_arguments(args: list, kwargs: dict, arg_ix_list: list[int], data_cols: list[str]) -> list:
    """Gets the desired arguments from a combined list of indices their corresponding argument names
    from the :py:attr:`args` and :py:attr:`kwargs` passsed to a function.

    Args:
        args (:obj:`list`): The orginal function arguments
        kwargs (:obj:`dict`): The original function keyword arguments
        arg_ix_list (:obj:`list[int]`): The indicies within :py:attr:`args` of the values of :py:attr:`data_cols`.
        data_cols (:obj:`list[str]`): The names of the desired :py:attr:`args` or :py:attr:`kwargs`.

    Returns:
        ``list``: A list of the extracted values or None if it does not exist.
    """
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
    return arg_list


def _update_arguments(
    args: list, kwargs: dict, arg_ix_list: list, data_cols: list, arg_list: list
) -> tuple[list, dict]:
    """Update the :py:attr:`args` and :py:attr:`kwargs` for a set of updated arguments contained in :py:attr:`arg_list`.

    Args:
        args (`list`): The originally passed function arguments.
        kwargs (`dict`): The originally passed function keyword arguments.
        arg_ix_list (`list`): The indices of the arguments to update.
        data_cols (`list`): The names of the arguments to update.
        arg_list (`list`): The new argument values, corresponding to both :py:attr:`arg_ix_list` and :py:attr:`data_cols`.

    Returns:
        tuple[list, dict]: _description_
    """
    for ix, name, new in zip(arg_ix_list, data_cols, arg_list):
        try:
            args[ix] = new
        except IndexError:
            try:
                kwargs[name] = new
            except KeyError:
                # No need to pass through a non-existent input that is defaulted to None
                pass
    return args, kwargs


def convert_args_to_lists(length: int, *args) -> list[list]:
    """Convert method arguments to a list of length `length` for each argument passed.

    Args:
        length (int): The length of the argument list.
        args: A series of arguments to be converted series of individual lists of length :py:attr:`length` for
            each argument passed.

    Returns:
        list[list]: A list of lists of length :py:attr:`length` for each argument passed.
    """
    return [a if isinstance(a, list) else [a] * length for a in args]


def df_to_series(data: pd.DataFrame, *args: str) -> tuple[pd.Series, ...]:
    """Converts a ``DataFrame`` and dynamic number of column names to a an equal number of pandas ``Series``
    corresponding to the column names. If ``None`` is passed as an argument to args, then ``None`` will
    be returned in place of a Series to maintain consistent ordering of inputs and outputs.

    Args:
        data (obj:`pandas.DataFrame`): The ``DataFrame`` object containg the column names in :py:attr:`args`.
        args(obj:`str`): A dynamic number of strings that make up the column names that need to be
            returned back as pandas ``Series`` objects.

    Raises:
        ValueError: Raised if :py:attr:`data` is not a pandas ``DataFrame``.
        ValueError: Raised if any of the args passed is not contained in ``data``.

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
    _check_cols_in_df(data, *args)
    return tuple(data.loc[:, col].copy() if col is not None else col for col in args)


def multiple_df_to_single_df(*args: pd.DataFrame, align_col: str | None = None) -> pd.DataFrame:
    """Convert multiple ``DataFrames`` to a single ``DataFrame`` either along the index, when :py:attr:`align_col`
    is None, or along the :py:attr:`align_col` when a value is provided.

    Args:
        args(:obj:`pandas.DataFrame`): A dynamic number of ``DataFrame`` inputs that need to be joined.
        align_col(:obj:`str` | :obj:`None`, optional): The common column shared among :py:attr:`args`, or the index,
            if ``None``. Defaults to None.

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


def series_to_df(*args: pd.Series, names: list[str] = None) -> tuple[pd.DataFrame, list[str | int]]:
    """Convert a dynamic number of pandas ``Series`` to a single pandas ``DataFrame`` by concatenating
    with an outer join, so the any missing values being filled with a NaN value, and each argument
    becomes a column of the resulting ``DataFrame``.

    Args:
        args(:obj:`pandas.Series`): A series of of pandas ``Series`` objects that share a common axis.
        names(:obj:`list[str]`): A list of alternative names for the Series to be used as the column
            name in the returned ``DataFrame`` and list of new args in place of None.

    Returns:
        ``tuple[pandas.DataFrame, list[str | int, ...]]``: A single data structure combining all the
            passed arguments, and the `name` associated with each passed `Series`.
    """
    if not all(isinstance(el, pd.Series) for el in args):
        raise TypeError("At least one of the provided values was not a pandas Series")

    # Rename the series to the name of the method argument if it doesn't already have name
    if names is None:
        names = [None] * len(args)
    names = [name if el.name is None else el.name for el, name in zip(args, names)]
    args = [el.rename(name) if el.name is None else el for el, name in zip(args, names)]

    args = [el.to_frame() for el in args]
    if len(args) > 1:
        return multiple_df_to_single_df(*args), names
    return args[0], names


def series_method(data_cols: list[str] = None):
    """Wrapper method for methods that operate on pandas ``Series``, and not ``DataFrame``s that allows
    the passing of column names that are potentially contained in a pandas ``DataFrame`` to be pulled
    out as separate pandas ``Series`` objects to be passed back to the method. This is a convenience
    wrapper that reduces the amount of boilerplate required to enable both ``DataFrame`` and ``Series``
    arguments to be used interchangably in a single method's API.

    Args:
        data_cols (list[str], optional): The names of the method arguments that should be converted
            from ``str`` to pandas ``Series`` when ``data`` is provided as a pandas ``DataFrame`` to the
            focal method. Defaults to None.
    """

    def decorator(func: Callable):
        """Gathes the arg indices from :py:attr:`data_cols` to be used in ``wrapper``."""
        argspec = getfullargspec(func)
        arg_ix_list = []
        if data_cols is not None:
            arg_ix_list = [argspec.args.index(name) for name in data_cols]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            if no_df := (df := kwargs.get("data", None)) is None:
                if arg_ix_list == []:
                    # Let the original method handle the provided arguments if unconfigured
                    return func(*args, *kwargs)

            args = list(args)

            # Fetch the user inputs, convert them to the Series and None values, as appropriate, and
            # update the args and kwargs for the new values
            arg_list = _get_arguments(args, kwargs, arg_ix_list, data_cols)  # Fetch inputs
            new_args = df_to_series(df, *arg_list)  # Convert inputs
            args, kwargs = _update_arguments(args, kwargs, arg_ix_list, data_cols, new_args)
            if not no_df:
                kwargs.pop("data")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def dataframe_method(data_cols: list[str] = None):
    """Wrapper method for methods that operate on a pandas ``DataFrame``, and not ``Series`` that allows
    the passing of the ``Series``, so that they can be combined in a ``DataFrame`` and passed back to the
    method. This is a convenience wrapper that reduces the amount of boilerplate required to enable
    both ``DataFrame`` and ``Series`` arguments to be used interchangably in a single method's API.

    Args:
        data_cols (list[str], optional): The names of the method arguments that should be converted
            from a pandas ``Series`` to ``str``, along with the creation of the :py:attr:`data` keyword argument
            when the column data is passed as a ``Series``. Defaults to None.
    """

    def decorator(func: Callable):
        """Gathes the arg indices from :py:attr:`data_cols` to be used in ``wrapper``."""
        argspec = getfullargspec(func)
        arg_ix_list = []
        if data_cols is not None:
            arg_ix_list = [argspec.args.index(name) for name in data_cols]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            args = list(args)
            arg_list = _get_arguments(args, kwargs, arg_ix_list, data_cols)
            if (df := kwargs.get("data", None)) is not None:
                # If a DataFrame is provided and the wrapper is unconfigured, then pass straight to the function
                if arg_ix_list == []:
                    return func(*args, *kwargs)

                # If data is passed, then convert Series arguments to Series.name, and check for the
                # existence of the column in the data
                arg_list = [arg.name if isinstance(arg, pd.Series) else arg for arg in arg_list]
                _check_cols_in_df(df, *arg_list)

                # Update the args and kwargs as need and call the function
                args, kwargs = _update_arguments(args, kwargs, arg_ix_list, data_cols, arg_list)
                return func(*args, **kwargs)

            # When no data is provided, then convert the Series arguments, update args and kwargs,
            # appropriately, then call the function
            df, arg_list = series_to_df(*arg_list, names=data_cols)
            args, kwargs = _update_arguments(args, kwargs, arg_ix_list, data_cols, arg_list)
            kwargs["data"] = df
            return func(*args, **kwargs)

        return wrapper

    return decorator
