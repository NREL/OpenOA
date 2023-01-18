from __future__ import annotations

import json
import itertools
from copy import deepcopy
from typing import Callable, Optional, Sequence
from pathlib import Path

import yaml
import attrs
import numpy as np
import pandas as pd
import pyspark as spark
from attrs import field, define
from pyproj import Transformer
from shapely.geometry import Point

import openoa.utils.timeseries as ts
import openoa.utils.met_data_processing as met
from openoa.utils.metadata_fetch import attach_eia_data
from openoa.utils.unit_conversion import convert_power_to_energy


# *************************************************************************
# Define the analysis requirements for ease of findability and modification
# *************************************************************************


# Datetime frequency checks
_at_least_monthly = ("M", "MS", "W", "D", "H", "T", "min", "S", "L", "ms", "U", "us", "N")
_at_least_daily = ("D", "H", "T", "min", "S", "L", "ms", "U", "us", "N")
_at_least_hourly = ("H", "T", "min", "S", "L", "ms", "U", "us", "N")

ANALYSIS_REQUIREMENTS = {
    "MonteCarloAEP": {
        "meter": {
            "columns": ["MMTR_SupWh"],
            "freq": _at_least_monthly,
        },
        "curtail": {
            "columns": ["IAVL_DnWh", "IAVL_ExtPwrDnWh"],
            "freq": _at_least_monthly,
        },
        "reanalysis": {
            "columns": ["WMETR_HorWdSpd", "WMETR_AirDen"],
            "conditional_columns": {
                "reg_temperature": ["WMETR_EnvTmp"],
                "reg_wind_direction": ["WMETR_HorWdSpdU", "WMETR_HorWdSpdV"],
            },
            "freq": _at_least_monthly,
        },
    },
    "TurbineLongTermGrossEnergy": {
        "scada": {
            "columns": ["id", "WMET_HorWdSpd", "WTUR_W"],
            "freq": _at_least_daily,
        },
        "reanalysis": {
            "columns": ["WMETR_HorWdSpd", "WMETR_HorWdDir", "WMETR_AirDen"],
            "freq": _at_least_daily,
        },
    },
    "ElectricalLosses": {
        "scada": {
            "columns": ["WTUR_SupWh"],
            "freq": _at_least_daily,
        },
        "meter": {
            "columns": ["MMTR_SupWh"],
            "freq": _at_least_monthly,
        },
    },
    "WakeLosses": {
        "scada": {
            "columns": ["id", "windspeed", "power"],
            "freq": _at_least_hourly,
        },
        "reanalysis": {
            "columns": ["windspeed", "wind_direction"],
            "freq": _at_least_hourly,
        },
    },
}


# ****************************************
# Validators, Loading, and General methods
# ****************************************


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
            if a.init and isinstance(a.default, type(attrs.NOTHING))  # type: ignore
        ]
        undefined = sorted(set(required_inputs) - set(kwargs))
        if undefined:
            raise AttributeError(
                f"The class defintion for {cls.__name__} is missing the following inputs: {undefined}"
            )
        return cls(**kwargs)  # type: ignore


def _analysis_filter(error_dict: dict, analysis_types: list[str] = ["all"]) -> dict:
    """Filters the errors found by the analysis requirements  provided by the `analysis_types`.

    Args:
        error_dict (`dict`): The dictionary of errors separated by the keys:
            "missing", "dtype", and "frequency".
        analysis_types (`list[str]`, optional): The list of analysis types to
            consider for validation. If "all" is contained in the list, then all errors
            are returned back, and if `None` is contained in the list, then no errors
            are returned, otherwise the union of analysis requirements is returned back.
            Defaults to ["all"].

    Returns:
        dict: The missing column, bad dtype, and incorrect timestamp frequency errors
            corresponding to the user's analysis types.
    """
    if "all" in analysis_types:
        return error_dict

    if None in analysis_types:
        return {}

    categories = ("scada", "meter", "tower", "curtail", "reanalysis", "asset")
    requirements = {key: ANALYSIS_REQUIREMENTS[key] for key in analysis_types}
    column_requirements = {
        cat: set(
            itertools.chain(*[r.get(cat, {}).get("columns", []) for r in requirements.values()])
        )
        for cat in categories
    }

    # Filter the missing columns, so only analysis-specific columns are provided
    error_dict["missing"] = {
        key: values.intersection(error_dict["missing"].get(key, []))
        for key, values in column_requirements.items()
    }

    # Filter the bad dtype columns, so only analysis-specific columns are provided
    error_dict["dtype"] = {
        key: values.intersection(error_dict["dtype"].get(key, []))
        for key, values in column_requirements.items()
    }

    return error_dict


def _compose_error_message(error_dict: dict, analysis_types: list[str] = ["all"]) -> str:
    """Takes a dictionary of error messages from the `PlantData` validation routines,
    filters out errors unrelated to the intended analysis types, and creates a
    human-readable error message.

    Args:
        error_dict (dict): See `PlantData._errors` for more details.
        analysis_types (list[str], optional): The user-input analysis types, which are
            used to filter out unlreated errors. Defaults to ["all"].

    Returns:
        str: The human-readable error message breakdown.
    """
    if "all" not in analysis_types:
        error_dict = _analysis_filter(error_dict, analysis_types)

    if None in analysis_types:
        return ""

    messages = [
        f"`{name}` data is missing the following columns: {cols}"
        for name, cols in error_dict["missing"].items()
        if len(cols) > 0
    ]
    messages.extend(
        [
            f"`{name}` data columns were of the wrong type: {cols}"
            for name, cols in error_dict["dtype"].items()
            if len(cols) > 0
        ]
    )
    messages.extend(
        [
            f"`{name}` data is of the wrong frequency: {freq}"
            for name, freq in error_dict["frequency"].items()
        ]
    )
    return "\n".join(messages)


def frequency_validator(
    actual_freq: str | int | float | None,
    desired_freq: Optional[str | None | set[str]],
    exact: bool,
) -> bool:
    """Helper function to check if the actual datetime stamp frequency is valid compared
    to what is required.

    Args:
        actual_freq(:obj:`str` | `int` | `float` | `None`): The frequency of the timestamp, either
            as an offset alias or manually determined in seconds between timestamps.
        desired_freq (Optional[str  |  None  |  set[str]]): Either the exact frequency,
            required or a set of options that are also valid, in which case any numeric
            information encoded in `actual_freq` will be dropped.
        exact(:obj:`bool`): If the provided frequency codes should be exact matches (`True`),
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

    # If an offset alias couldn't be found, then convert the desired frequency strings to seconds
    if not isinstance(actual_freq, str):
        desired_freq = set([ts.offset_to_seconds(el) for el in desired_freq])

    if exact:
        return actual_freq in desired_freq

    if isinstance(actual_freq, str):
        actual_freq = "".join(filter(str.isalpha, actual_freq))
        return actual_freq in desired_freq

    # For non-exact matches, just check that the actual is less than the maximum allowable frequency
    return actual_freq < max(desired_freq)


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


def convert_reanalysis(value: dict[str, dict]):
    return {k: ReanalysisMetaData.from_dict(v) for k, v in value.items()}


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
    if reverse:
        col_map = {v: k for k, v in col_map.items()}
    return df.rename(columns=col_map)


# ***************************************
# Define the meta data validation classes
# ***************************************


@define(auto_attribs=True)
class SCADAMetaData(FromDictMixin):  # noqa: F821
    """A metadata schematic to create the necessary column mappings and other validation
    components, or other data about the SCADA data, that will contribute to a larger
    plant metadata schema/routine.

    Args:
        time (str): The datetime stamp for the SCADA data, by default "time". This data should be of
            type: `np.datetime64[ns]`, or able to be converted to a pandas DatetimeIndex. Additional
            columns describing the datetime stamps are: `frequency`
        WTUR_TurNam (str): The turbine identifier column in the SCADA data, by default "id". This data should be of
            type: `str`.
        WTUR_W (str): The power produced, in kW, column in the SCADA data, by default "WTUR_W".
            This data should be of type: `float`.
        WMET_HorWdSpd (str): The measured windspeed, in m/s, column in the SCADA data, by default "WMET_HorWdSpd".
            This data should be of type: `float`.
        WMET_HorWdDir (str): The measured wind direction, in degrees, column in the SCADA data, by default
            "WMET_HorWdDir". This data should be of type: `float`.
        WTUR_TurSt (str): The status code column in the SCADA data, by default "WTUR_TurSt". This data
            should be of type: `str`.
        WROT_BlPthAngVal (str): The pitch, in degrees, column in the SCADA data, by default "WROT_BlPthAngVal". This data
            should be of type: `float`.
        WMET_EnvTmp (str): The temperature column in the SCADA data, by default "WMET_EnvTmp". This
            data should be of type: `float`.
        frequency (str): The frequency of `time` in the SCADA data, by default "10T". The input
            should align with the `Pandas frequency offset aliases`_.


    .. _Pandas frequency offset aliases:
       https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    """

    # DataFrame columns
    time: str = field(default="time")
    WTUR_TurNam: str = field(default="WTUR_TurNam")
    WTUR_W: str = field(default="WTUR_W")
    WMET_HorWdSpd: str = field(default="WMET_HorWdSpd")
    WMET_HorWdDir: str = field(default="WMET_HorWdDir")
    WTUR_TurSt: str = field(default="WTUR_TurSt")
    WROT_BlPthAngVal: str = field(default="WROT_BlPthAngVal")
    WMET_EnvTmp: str = field(default="WMET_EnvTmp")

    # Data about the columns
    frequency: str = field(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = field(default="scada", init=False)
    WTUR_SupWh: str = field(default="WTUR_SupWh", init=False)  # calculated in PlantData
    col_map: dict = field(init=False)
    dtypes: dict = field(
        default=dict(
            time=np.datetime64,
            WTUR_TurNam=str,
            WTUR_W=float,
            WMET_HorWdSpd=float,
            WMET_HorWdDir=float,
            WTUR_TurSt=str,
            WROT_BlPthAngVal=float,
            WMET_EnvTmp=float,
            WTUR_SupWh=float,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = field(
        default=dict(
            time="datetim64[ns]",
            WTUR_TurNam=None,
            WTUR_W="kW",
            WMET_HorWdSpd="m/s",
            WMET_HorWdDir="deg",
            WTUR_TurSt=None,
            WROT_BlPthAngVal="deg",
            WMET_EnvTmp="C",
            WTUR_SupWh="kWh",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            WTUR_TurNam=self.WTUR_TurNam,
            WTUR_W=self.WTUR_W,
            WMET_HorWdSpd=self.WMET_HorWdSpd,
            WMET_HorWdDir=self.WMET_HorWdDir,
            WTUR_TurSt=self.WTUR_TurSt,
            WROT_BlPthAngVal=self.WROT_BlPthAngVal,
            WMET_EnvTmp=self.WMET_EnvTmp,
            WTUR_SupWh=self.WTUR_SupWh,
        )


@define(auto_attribs=True)
class MeterMetaData(FromDictMixin):  # noqa: F821
    """A metadata schematic to create the necessary column mappings and other validation
    components, or other data about energy meter data, that will contribute to a larger
    plant metadata schema/routine.

    Args:
        time (str): The datetime stamp for the meter data, by default "time". This data should
            be of type: `np.datetime64[ns]`, or able to be converted to a pandas DatetimeIndex.
            Additional columns describing the datetime stamps are: `frequency`
        MMTR_SupWh (str): The energy produced, in kWh, column in the meter data, by default
            "MMTR_SupWh". This data should be of type: `float`.
        frequency (str): The frequency of `time` in the meter data, by default "10T". The input
            should align with the `Pandas frequency offset aliases`_.


    .. _Pandas frequency offset aliases:
       https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    """

    # DataFrame columns
    time: str = field(default="time")
    MMTR_SupWh: str = field(default="MMTR_SupWh")

    # Data about the columns
    frequency: str = field(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = field(default="meter", init=False)
    col_map: dict = field(init=False)
    dtypes: dict = field(
        default=dict(
            time=np.datetime64,
            MMTR_SupWh=float,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = field(
        default=dict(
            time="datetim64[ns]",
            MMTR_SupWh="kWh",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            MMTR_SupWh=self.MMTR_SupWh,
        )


@define(auto_attribs=True)
class TowerMetaData(FromDictMixin):  # noqa: F821
    """A metadata schematic to create the necessary column mappings and other validation
    components, or other data about meteorological tower (met tower) data, that will contribute to a
    larger plant metadata schema/routine.

    Args:
        time (str): The datetime stamp for the met tower data, by default "time". This data should
            be of type: `np.datetime64[ns]`, or able to be converted to a pandas DatetimeIndex.
            Additional columns describing the datetime stamps are: `frequency`
        id (str): The met tower identifier column in the met tower data, by default "id". This data
            should be of type: `str`.
        frequency (str): The frequency of `time` in the met tower data, by default "10T". The input
            should align with the `Pandas frequency offset aliases`_.

    .. _Pandas frequency offset aliases:
       https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    """

    # DataFrame columns
    time: str = field(default="time")
    id: str = field(default="id")

    # Data about the columns
    frequency: str = field(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = field(default="tower", init=False)
    col_map: dict = field(init=False)
    dtypes: dict = field(
        default=dict(
            time=np.datetime64,
            id=str,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = field(
        default=dict(
            time="datetim64[ns]",
            id=None,
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            id=self.id,
        )


@define(auto_attribs=True)
class StatusMetaData(FromDictMixin):  # noqa: F821
    """A metadata schematic to create the necessary column mappings and other validation
    components, or other data about the turbine status log data, that will contribute to a
    larger plant metadata schema/routine.

    Args:
        time (str): The datetime stamp for the status data, by default "time". This data should
            be of type: `np.datetime64[ns]`, or able to be converted to a pandas DatetimeIndex.
            Additional columns describing the datetime stamps are: `frequency`
        id (str): The turbine identifier column in the status data, by default "id". This data
            should be of type: `str`.
        status_id (str): The status code identifier column in the status data, by default "id". This data
            should be of type: `str`.
        status_code (str): The status code column in the status data, by default "id". This data
            should be of type: `str`.
        status_text (str): The status text description column in the status data, by default "id".
            This data should be of type: `str`.
        frequency (str): The frequency of `time` in the met tower data, by default "10T". The input
            should align with the `Pandas frequency offset aliases`_.

    .. _Pandas frequency offset aliases:
       https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    """

    # DataFrame columns
    time: str = field(default="time")
    id: str = field(default="id")
    status_id: str = field(default="status_id")
    status_code: str = field(default="status_code")
    status_text: str = field(default="status_text")

    # Data about the columns
    frequency: str = field(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = field(default="status", init=False)
    col_map: dict = field(init=False)
    dtypes: dict = field(
        default=dict(
            time=np.datetime64,
            id=str,
            status_id=np.int64,
            status_code=np.int64,
            status_text=str,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = field(
        default=dict(
            time="datetim64[ns]",
            id=None,
            status_id=None,
            status_code=None,
            status_text=None,
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            id=self.id,
            status_id=self.status_id,
            status_code=self.status_code,
            status_text=self.status_text,
        )


@define(auto_attribs=True)
class CurtailMetaData(FromDictMixin):  # noqa: F821
    """A metadata schematic to create the necessary column mappings and other validation
    components, or other data about the plant curtailment data, that will contribute to a
    larger plant metadata schema/routine.

    Args:
        time (str): The datetime stamp for the curtailment data, by default "time". This data should
            be of type: `np.datetime64[ns]`, or able to be converted to a pandas DatetimeIndex.
            Additional columns describing the datetime stamps are: `frequency`
        IAVL_ExtPwrDnWh (str): The curtailment, in kWh, column in the curtailment data, by default
            "IAVL_ExtPwrDnWh". This data should be of type: `float`.
        IAVL_DnWh (str): The availability, in kWh, column in the curtailment data, by default
            "IAVL_DnWh". This data should be of type: `float`.
        frequency (str): The frequency of `time` in the met tower data, by default "10T". The input
            should align with the `Pandas frequency offset aliases`_.

    .. _Pandas frequency offset aliases:
       https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    """

    # DataFrame columns
    time: str = field(default="time")
    IAVL_ExtPwrDnWh: str = field(default="IAVL_ExtPwrDnWh")
    IAVL_DnWh: str = field(default="IAVL_DnWh")

    # Data about the columns
    frequency: str = field(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = field(default="curtail", init=False)
    col_map: dict = field(init=False)
    dtypes: dict = field(
        default=dict(
            time=np.datetime64,
            IAVL_ExtPwrDnWh=float,
            IAVL_DnWh=float,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = field(
        default=dict(
            time="datetim64[ns]",
            IAVL_ExtPwrDnWh="kWh",
            IAVL_DnWh="kWh",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            IAVL_ExtPwrDnWh=self.IAVL_ExtPwrDnWh,
            IAVL_DnWh=self.IAVL_DnWh,
        )


@define(auto_attribs=True)
class AssetMetaData(FromDictMixin):  # noqa: F821
    """A metadata schematic to create the necessary column mappings and other validation
    components, or other data about the site's asset metadata, that will contribute to a
    larger plant metadata schema/routine.

    Args:
        id (str): The asset identifier column in the asset metadata, by default "id". This data
            should be of type: `str`.
        latitude (str): The asset's latitudinal position, in WGS84, column in the asset metadata, by
            default "latitude". This data should be of type: `float`.
        longitude (str): The asset's longitudinal position, in WGS84, column in the asset metadata,
            by default "longitude". This data should be of type: `float`.
        rated_power (str): The asset's rated power, in kW, column in the asset metadata, by default
            "rated_power". This data should be of type: `float`.
        hub_height (str): The asset's hub height, in m, column in the asset metadata, by default
            "hub_height". This data should be of type: `float`.
        elevation (str): The asset's elevation above sea level, in m, column in the asset metadata,
            by default "elevation". This data should be of type: `float`.
        type (str): The type of asset column in the asset metadata, by default "type". This data
            should be of type: `str`.
    """

    # DataFrame columns
    id: str = field(default="id")
    latitude: str = field(default="latitude")
    longitude: str = field(default="longitude")
    rated_power: str = field(default="rated_power")
    hub_height: str = field(default="hub_height")
    rotor_diameter: str = field(default="rotor_diameter")
    elevation: str = field(default="elevation")
    type: str = field(default="type")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = field(default="asset", init=False)
    col_map: dict = field(init=False)
    dtypes: dict = field(
        default=dict(
            id=str,
            latitude=float,
            longitude=float,
            rated_power=float,
            hub_height=float,
            rotor_diameter=float,
            elevation=float,
            type=str,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = field(
        default=dict(
            id=None,
            latitude="WGS84",
            longitude="WGS84",
            rated_power="kW",
            hub_height="m",
            rotor_diameter="m",
            elevation="m",
            type=None,
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            id=self.id,
            latitude=self.latitude,
            longitude=self.longitude,
            rated_power=self.rated_power,
            hub_height=self.hub_height,
            rotor_diameter=self.rotor_diameter,
            elevation=self.elevation,
            type=self.type,
        )


@define(auto_attribs=True)
class ReanalysisMetaData(FromDictMixin):  # noqa: F821
    """A metadata schematic for each of the reanalsis products to be used for operationa analyses
    to create the necessary column mappings and other validation components, or other data about
    the site's asset metadata, that will contribute to a larger plant metadata schema/routine.

    Args:
        time (str): The datetime stamp for the curtailment data, by default "time". This data should
            be of type: `np.datetime64[ns]`, or able to be converted to a pandas DatetimeIndex.
            Additional columns describing the datetime stamps are: `frequency`
        WMETR_HorWdSpd (:obj:`str`): The reanalysis non-directional windspeed data column name, in
            m/s, by default "WMETR_HorWdSpd".
        WMETR_HorWdSpdU (:obj:`str`): The reanalysis u-direction windspeed data column name, in m/s,
            by default "WMETR_HorWdSpdU".
        WMETR_HorWdSpdV (:obj:`str`): The reanalysis v-directional windspeed data column name, in
            m/s, by default "WMETR_HorWdSpdV".
        WMETR_HorWdDir (:obj:`str`): The reanalysis windspeed horizontal direction data column name,
            in degrees, by default "WMETR_HorWdDir".
        WMETR_EnvTmp (:obj:`str`): The temperature data column name in the renalysis data, in
            degrees Kelvin, by default "WMETR_EnvTmp".
        WMETR_AirDen (:obj:`str`): The air density reanalysis data column name, in kg/m^3, by
            default "WMETR_AirDen".
        WMETR_EnvPres (:obj:`str`): The surface air pressure reanalysis data column name, in Pa, by
            default "WMETR_EnvPres".
        frequency (:obj:`str`): The frequency of the timestamps in the :py:attr:`time` column, by
            default "10T".
    """

    time: str = field(default="time")
    WMETR_HorWdSpd: str = field(default="WMETR_HorWdSpd")
    WMETR_HorWdSpdU: str = field(default="WMETR_HorWdSpdU")
    WMETR_HorWdSpdV: str = field(default="WMETR_HorWdSpdV")
    WMETR_HorWdDir: str = field(default="WMETR_HorWdDir")
    WMETR_EnvTmp: str = field(default="WMETR_EnvTmp")
    WMETR_AirDen: str = field(default="WMETR_AirDen")
    WMETR_EnvPres: str = field(default="surface_pressure")

    # Data about the columns
    frequency: str = field(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = field(default="reanalysis", init=False)
    col_map: dict = field(init=False)
    dtypes: dict = field(
        default=dict(
            time=np.datetime64,
            WMETR_HorWdSpd=float,
            WMETR_HorWdSpdU=float,
            WMETR_HorWdSpdV=float,
            WMETR_HorWdDir=float,
            WMETR_EnvTmp=float,
            WMETR_AirDen=float,
            WMETR_EnvPres=float,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = field(
        default=dict(
            time="datetim64[ns]",
            WMETR_HorWdSpd="m/s",
            WMETR_HorWdSpdU="m/s",
            WMETR_HorWdSpdV="m/s",
            WMETR_HorWdDir="deg",
            WMETR_EnvTmp="K",
            WMETR_AirDen="kg/m^3",
            WMETR_EnvPres="Pa",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            WMETR_HorWdSpd=self.WMETR_HorWdSpd,
            WMETR_HorWdSpdU=self.WMETR_HorWdSpdU,
            WMETR_HorWdSpdV=self.WMETR_HorWdSpdV,
            WMETR_HorWdDir=self.WMETR_HorWdDir,
            WMETR_EnvTmp=self.WMETR_EnvTmp,
            WMETR_AirDen=self.WMETR_AirDen,
            WMETR_EnvPres=self.WMETR_EnvPres,
        )


@define(auto_attribs=True)
class PlantMetaData(FromDictMixin):  # noqa: F821
    """Composese the metadata/validation requirements from each of the individual data
    types that can compose a `PlantData` object.

    Args:
        latitude (`float`): The wind power plant's center point latitude.
        longitude (`float`): The wind power plant's center point longitude.
        capacity (`float`): The capacity of the plant in MW
        scada (`SCADAMetaData`): A dictionary containing the `SCADAMetaData`
            column mapping and frequency parameters. See `SCADAMetaData` for more details.
        meter (`MeterMetaData`): A dictionary containing the `MeterMetaData`
            column mapping and frequency parameters. See `MeterMetaData` for more details.
        tower (`TowerMetaData`): A dictionary containing the `TowerMetaData`
            column mapping and frequency parameters. See `TowerMetaData` for more details.
        status (`StatusMetaData`): A dictionary containing the `StatusMetaData`
            column mapping parameters. See `StatusMetaData` for more details.
        curtail (`CurtailMetaData`): A dictionary containing the `CurtailMetaData`
            column mapping and frequency parameters. See `CurtailMetaData` for more details.
        asset (`AssetMetaData`): A dictionary containing the `AssetMetaData`
            column mapping parameters. See `AssetMetaData` for more details.
        reanalysis (`dict[str, ReanalysisMetaData]`): A dictionary containing the
            reanalysis type (as keys, such as "era5" or "merra2") and `ReanalysisMetaData`
            column mapping and frequency parameters for each type of reanalysis data
            provided. See `ReanalysisMetaData` for more details.
    """

    latitude: float = field(default=0, converter=float)
    longitude: float = field(default=0, converter=float)
    capacity: float = field(default=0, converter=float)
    scada: SCADAMetaData = field(default={}, converter=SCADAMetaData.from_dict)
    meter: MeterMetaData = field(default={}, converter=MeterMetaData.from_dict)
    tower: TowerMetaData = field(default={}, converter=TowerMetaData.from_dict)
    status: StatusMetaData = field(default={}, converter=StatusMetaData.from_dict)
    curtail: CurtailMetaData = field(default={}, converter=CurtailMetaData.from_dict)
    asset: AssetMetaData = field(default={}, converter=AssetMetaData.from_dict)
    reanalysis: dict[str, ReanalysisMetaData] = field(
        default={}, converter=convert_reanalysis  # noqa: F821
    )  # noqa: F821

    @property
    def column_map(self) -> dict[str, dict]:
        """Provides the column mapping for all of the available data types with
        the name of each data type as the key and the dictionary mapping as the values.
        """
        values = dict(
            scada=self.scada.col_map,
            meter=self.meter.col_map,
            tower=self.tower.col_map,
            status=self.status.col_map,
            asset=self.asset.col_map,
            curtail=self.curtail.col_map,
            reanalysis={},
        )
        if self.reanalysis != {}:
            values["reanalysis"] = {k: v.col_map for k, v in self.reanalysis.items()}
        return values

    @property
    def dtype_map(self) -> dict[str, dict]:
        """Provides the column dtype matching for all of the available data types with
        the name of each data type as the keys, and the column dtype mapping as values.
        """
        types = dict(
            scada=self.scada.dtypes,
            meter=self.meter.dtypes,
            tower=self.tower.dtypes,
            status=self.status.dtypes,
            asset=self.asset.dtypes,
            curtail=self.curtail.dtypes,
            reanalysis={},
        )
        if self.reanalysis != {}:
            types["reanalysis"] = {k: v.dtypes for k, v in self.reanalysis.items()}
        return types

    @property
    def coordinates(self) -> tuple[float, float]:
        """Returns the latitude, longitude pair for the wind power plant.

        Returns:
            tuple[float, float]: The (latitude, longitude) pair
        """
        return self.latitude, self.longitude

    @classmethod
    def from_json(cls, metadata_file: str | Path) -> PlantMetaData:
        """Loads the metadata from a JSON file.

        Args:
            metadata_file (`str | Path`): The full path and file name of the JSON file.

        Raises:
            FileExistsError: Raised if the file doesn't exist at the provided location.

        Returns:
            PlantMetaData
        """
        metadata_file = Path(metadata_file).resolve()
        if not metadata_file.is_file():
            raise FileExistsError(f"Input JSON file: {metadata_file} is an invalid input.")

        with open(metadata_file) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, metadata_file: str | Path) -> PlantMetaData:
        """Loads the metadata from a YAML file with a PyYAML encoding.

        Args:
            metadata_file (`str | Path`): The full path and file name of the YAML file.

        Raises:
            FileExistsError: Raised if the file doesn't exist at the provided location.

        Returns:
            PlantMetaData
        """
        metadata_file = Path(metadata_file).resolve()
        if not metadata_file.is_file():
            raise FileExistsError(f"Input YAML file: {metadata_file} is an invalid input.")

        with open(metadata_file) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def load(cls, data: str | Path | dict | PlantMetaData) -> PlantMetaData:
        """Loads the metadata from either a dictionary or file such as a JSON or YAML file.

        Args:
            metadata_file (`str | Path | dict`): Either a pre-loaded dictionary or
                the full path and file name of the JSON or YAML file.

        Raises:
            ValueError: Raised if the file name doesn't reflect a JSON or YAML encoding.
            ValueError: Raised if the data provided isn't of the correct data type.

        Returns:
            PlantMetaData
        """
        if isinstance(data, PlantMetaData):
            return data

        if isinstance(data, str):
            data = Path(data).resolve()

        if isinstance(data, Path):
            if data.suffix == ".json":
                return cls.from_json(data)
            elif data.suffix in (".yaml", ".yml"):
                return cls.from_yaml(data)
            else:
                raise ValueError("Bad input file extension, must be one of: .json, .yml, or .yaml")

        if isinstance(data, dict):
            return cls.from_dict(data)

        raise ValueError("PlantMetaData can only be loaded from str, Path, or dict objects.")

    def frequency_requirements(self, analysis_types: list[str | None]) -> dict[str, set[str]]:
        """Creates a frequency requirements dictionary for each data type with the name
        as the key and a set of valid frequency fields as the values.

        Args:
            analysis_types (list[str  |  None]): The analyses the data is intended to be
                used for, which will determine what data need to be checked.

        Returns:
            dict[str, set[str]]: The dictionary of data type name and valid frequencies
                for the datetime stamps.
        """
        if "all" in analysis_types:
            requirements = deepcopy(ANALYSIS_REQUIREMENTS)
        else:
            requirements = {
                key: ANALYSIS_REQUIREMENTS[key] for key in analysis_types if key is not None
            }
        frequency_requirements = {
            key: {name: value["freq"] for name, value in values.items()}
            for key, values in requirements.items()
        }
        frequency = {
            k: []
            for k in set(
                itertools.chain.from_iterable([[*val] for val in frequency_requirements.values()])
            )
        }
        for vals in frequency_requirements.values():
            for name, req in vals.items():
                reqs = frequency[name]
                if reqs == []:
                    frequency[name] = set(req)
                else:
                    frequency[name] = reqs.intersection(req)
        return frequency


############################
# Define the PlantData class
############################


@define(auto_attribs=True)
class PlantData:
    """Overarching data object used for storing, accessing, and acting on the primary
    operational analysis data types, including: SCADA, meter, tower, status, curtailment,
    asset, and reanalysis data. As of version 3.0, this class provides an automated
    validation scheme through the use of `analysis_type` as well as a secondary scheme
    that can be run after further manipulations are performed. Additionally, version 3.0
    incorporates a metadata scheme `PlantMetaData` to map between user column naming
    conventions and the internal column naming conventions for both usability and code
    consistency.

    Args:
        metadata (`PlantMetaData`): A nested dictionary of the schema definition
            for each of the data types that will be input, and some additional plant
            parameters. See `PlantMetaData`, `SCADAMetaData`, `MeterMetaData`,
            `TowerMetaData`, `StatusMetaData`, `CurtailMetaData`, `AssetMetaData`,
            and/or `ReanalysisMetaData` for more information.
        analysis_type (`list[str]`): A single, or list of, analysis type(s) that
            will be run, that are configured in `ANALYSIS_REQUIREMENTS`.

            - None: Don't raise any errors for errors found in the data. This is intended
              for loading in messy data, but :py:meth:`validate` should be run later
              if planning on running any analyses.
            - "all": This is to check that all columns specified in the metadata schema
              align with the data provided, as well as data types and frequencies (where
              applicable).
            - "MonteCarloAEP": Checks the data components that are relevant to a Monte
              Carlo AEP analysis. See `ANALYSIS_REQUIREMENTS` for requirements details.
            - "TurbineLongTermGrossEnergy": Checks the data components that are relevant
              to a turbine long term gross energy analysis. See `ANALYSIS_REQUIREMENTS`
              for requirements details.
            - "ElectricalLosses": Checks the data components that are relevant to an
              electrical losses analysis. See `ANALYSIS_REQUIREMENTS` for requirements
              details.
            - "WakeLosses": Checks the data components that are relevant to a
              wake losses analysis. See `ANALYSIS_REQUIREMENTS` for requirements
              details.

        scada (`pd.DataFrame`): Either the SCADA data that's been pre-loaded to a
            pandas `DataFrame`, or a path to the location of the data to be imported.
            See :py:class:`SCADAMetaData` for column data specifications.
        meter (`pd.DataFrame`): Either the meter data that's been pre-loaded to a
            pandas `DataFrame`, or a path to the location of the data to be imported.
            See :py:class:`MeterMetaData` for column data specifications.
        tower (`pd.DataFrame`): Either the met tower data that's been pre-loaded
            to a pandas `DataFrame`, or a path to the location of the data to be
            imported. See :py:class:`TowerMetaDsata` for column data specifications.
        status (`pd.DataFrame`): Either the status data that's been pre-loaded to
            a pandas `DataFrame`, or a path to the location of the data to be imported.
            See :py:class:`StatusMetaData` for column data specifications.
        curtail (`pd.DataFrame`): Either the curtailment data that's been
            pre-loaded to a pandas `DataFrame`, or a path to the location of the data to
            be imported. See :py:class:`CurtailMetaData` for column data specifications.
        asset (`pd.DataFrame`): Either the asset summary data that's been
            pre-loaded to a pandas `DataFrame`, or a path to the location of the data to
            be imported. See :py:class:`AssetMetaData` for column data specifications.
        reanalysis (`dict[str, pd.DataFrame]`): Either the reanalysis data that's
            been pre-loaded to a dictionary of pandas `DataFrame`s with keys indicating
            the data source, such as "era5" or "merra2", or a dictionary of paths to the
            location of the data to be imported following the same key naming convention.
            See :py:class:`ReanalysisMetaData` for column data specifications.

    Raises:
        ValueError: Raised if any analysis specific validation checks don't pass with an
            error message highlighting the appropriate issues.
    """

    metadata: PlantMetaData = field(
        default={}, converter=PlantMetaData.load, on_setattr=[attrs.converters, attrs.validators]
    )
    analysis_type: list[str] | None = field(
        default=None,
        converter=convert_to_list,  # noqa: F821
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(list),
            member_validator=attrs.validators.in_([*ANALYSIS_REQUIREMENTS] + ["all", None]),
        ),
        on_setattr=[attrs.setters.convert, attrs.setters.validate],
    )
    scada: pd.DataFrame | None = field(default=None, converter=load_to_pandas)  # noqa: F821
    meter: pd.DataFrame | None = field(default=None, converter=load_to_pandas)  # noqa: F821
    tower: pd.DataFrame | None = field(default=None, converter=load_to_pandas)  # noqa: F821
    status: pd.DataFrame | None = field(default=None, converter=load_to_pandas)  # noqa: F821
    curtail: pd.DataFrame | None = field(default=None, converter=load_to_pandas)  # noqa: F821
    asset: pd.DataFrame | None = field(default=None, converter=load_to_pandas)  # noqa: F821
    reanalysis: dict[str, pd.DataFrame] | None = field(
        default=None, converter=load_to_pandas_dict  # noqa: F821
    )
    preprocess: Callable | None = field(default=None)  # Not currently in use

    # No user initialization required for attributes defined below here
    # Error catching in validation
    _errors: dict[str, list[str]] = field(
        default={"missing": {}, "dtype": {}, "frequency": {}, "attributes": []}, init=False
    )
    eia: dict = field(default={}, init=False)
    asset_distance_matrix: pd.DataFrame = field(init=False)
    asset_direction_matrix: pd.DataFrame = field(init=False)

    def __attrs_post_init__(self):
        self._calculate_reanalysis_columns()
        self._set_index_columns()
        self._validate_frequency()

        # Check the errors againts the analysis requirements
        error_message = _compose_error_message(self._errors, analysis_types=self.analysis_type)
        if error_message != "":
            raise ValueError(error_message)

        # Post-validation data manipulations
        # TODO: Need to have a class level input for the user-preferred projection system
        # TODO: Why does the non-WGS84 projection matter?
        self.parse_asset_geometry()
        self.calculate_asset_distance_matrix()
        self.calculate_asset_direction_matrix()
        self._calculate_turbine_energy()

        # Change the column names to the -25 convention for easier use in the rest of the code base
        self.update_column_names()

        if self.preprocess is not None:
            self.preprocess(
                self
            )  # TODO: should be a user-defined method to run the data cleansing steps

    @scada.validator
    @meter.validator
    @tower.validator
    @status.validator
    @curtail.validator
    @asset.validator
    @reanalysis.validator
    def data_validator(
        self, instance: attrs.Attribute, value: pd.DataFrame | dict[str | pd.DataFrame] | None
    ) -> None:
        """Validator function for each of the data buckets in `PlantData` that checks
        that the appropriate columns exist for each dataframe, each column is of the
        right type, and that the timestamp frequencies are appropriate for the given
        `analysis_type`.

        Args:
            instance (attrs.Attribute): The `attr` attribute details
            value (pd.DataFrame | dict[pd.DataFrame] | None): The attribute's
                user-provided value. A dictionary of dataframes is expected for
                reanalysis data only.
        """
        if None in self.analysis_type:
            return
        name = instance.name
        if value is None:
            self._errors["missing"].update(
                {name: list(getattr(self.metadata, name).col_map.values())}
            )
            self._errors["dtype"].update({name: list(getattr(self.metadata, name).dtypes.keys())})

        else:
            self._errors["missing"].update(self._validate_column_names(category=name))
            self._errors["dtype"].update(self._validate_dtypes(category=name))

    def _set_index_columns(self) -> None:
        """Sets the index value for each of the `PlantData` objects that are not `None`."""
        if self.scada is not None:
            time_col = self.metadata.scada.col_map["time"]
            id_col = self.metadata.scada.col_map["WTUR_TurNam"]
            self.scada[time_col] = pd.DatetimeIndex(self.scada[time_col])
            self.scada = self.scada.set_index([time_col, id_col])
            self.scada.index.names = ["time", "WTUR_TurNam"]

        if self.meter is not None:
            time_col = self.metadata.meter.col_map["time"]
            self.meter[time_col] = pd.DatetimeIndex(self.meter[time_col])
            self.meter = self.meter.set_index([time_col])
            self.meter.index.name = "time"

        if self.status is not None:
            time_col = self.metadata.status.col_map["time"]
            id_col = self.metadata.status.col_map["id"]
            self.status[time_col] = pd.DatetimeIndex(self.status[time_col])
            self.status = self.status.set_index([time_col, id_col])
            self.status.index.names = ["time", "id"]

        if self.tower is not None:
            time_col = self.metadata.tower.col_map["time"]
            id_col = self.metadata.tower.col_map["id"]
            self.tower[time_col] = pd.DatetimeIndex(self.tower[time_col])
            self.tower = self.tower.set_index([time_col, id_col])
            self.tower.index.names = ["time", "id"]

        if self.curtail is not None:
            time_col = self.metadata.curtail.col_map["time"]
            self.curtail[time_col] = pd.DatetimeIndex(self.curtail[time_col])
            self.curtail = self.curtail.set_index([time_col])
            self.curtail.index.name = "time"

        if self.asset is not None:
            id_col = self.metadata.asset.col_map["id"]
            self.asset = self.asset.set_index([id_col])
            self.asset.index.name = "id"

        if self.reanalysis is not None:
            for name in self.reanalysis:
                time_col = self.metadata.reanalysis[name].col_map["time"]
                self.reanalysis[name][time_col] = pd.DatetimeIndex(self.reanalysis[name][time_col])
                self.reanalysis[name] = self.reanalysis[name].set_index([time_col])
                self.reanalysis[name].index.name = "time"

    def _unset_index_columns(self) -> None:
        """Resets the index for each of the data types. This is intended solely for the use with
        the :py:meth:`validate` to ensure the validation methods are able to find the index columns
        in the column space
        """
        if self.scada is not None:
            self.scada.reset_index(drop=False, inplace=True)
        if self.meter is not None:
            self.meter.reset_index(drop=False, inplace=True)
        if self.status is not None:
            self.status.reset_index(drop=False, inplace=True)
        if self.tower is not None:
            self.tower.reset_index(drop=False, inplace=True)
        if self.curtail is not None:
            self.curtail.reset_index(drop=False, inplace=True)
        if self.asset is not None:
            self.asset.reset_index(drop=False, inplace=True)
        if self.reanalysis is not None:
            for name in self.reanalysis:
                self.reanalysis[name].reset_index(drop=False, inplace=True)

    @property
    def data_dict(self) -> dict[str, pd.DataFrame]:
        """Property that returns a dictionary of the data contained in the `PlantData` object.

        Returns:
            dict[str, pd.DataFrame]: A mapping of the data type's name and the `DataFrame`.
        """
        values = dict(
            scada=self.scada,
            meter=self.meter,
            tower=self.tower,
            asset=self.asset,
            status=self.status,
            curtail=self.curtail,
            reanalysis=self.reanalysis,
        )
        return values

    def to_csv(
        self,
        save_path: str | Path,
        with_openoa_col_names: bool = True,
        metadata: str = "metadata",
        scada: str = "scada",
        meter: str = "meter",
        tower: str = "tower",
        asset: str = "asset",
        status: str = "status",
        curtail: str = "curtail",
        reanalysis: str = "reanalysis",
    ) -> None:
        """Saves all of the dataframe objects to a CSV file in the provided `save_path` directory.

        Args:
            save_path (str | Path): The folder where all the data should be saved.
            with_openoa_col_names (bool, optional): Use the PlantData column names (True), or
                convert the column names back to the originally provided values. Defaults to True.
            metadata (str, optional): File name (without extension) to be used for the metadata. Defaults to "metadata".
            scada (str, optional): File name (without extension) to be used for the SCADA data. Defaults to "scada".
            meter (str, optional): File name (without extension) to be used for the meter data. Defaults to "meter".
            tower (str, optional): File name (without extension) to be used for the tower data. Defaults to "tower".
            asset (str, optional): File name (without extension) to be used for the asset data. Defaults to "scada".
            status (str, optional): File name (without extension) to be used for the status data. Defaults to "status".
            curtail (str, optional): File name (without extension) to be used for the curtailment data. Defaults to "curtail".
            reanalysis (str, optional): Base file name (without extension) to be used for the reanalysis data, where
                each dataset will use the name provided to form the following file name: {save_path}/{reanalysis}_{name}.
                Defaults to "reanalysis".
        """
        save_path = Path(save_path).resolve()
        if not save_path.exists():
            save_path.mkdir()

        meta = self.metadata.column_map

        if not with_openoa_col_names:
            self.update_column_names(to_original=True)
        else:
            for name, col_map in meta.items():
                if name == "reanalysis":
                    for re_name, re_col_map in col_map.items():
                        re_col_map = {k: k for k in re_col_map}
                        re_col_map["frequency"] = self.metadata.reanalysis[re_name].frequency
                        meta[name][re_name] = re_col_map
                    continue
                col_map = {k: k for k in col_map}
                meta_obj = getattr(self.metadata, name)
                if hasattr(meta_obj, "frequency"):
                    col_map["frequency"] = meta_obj.frequency
                meta[name] = col_map

        with open((save_path / metadata).with_suffix(".yml"), "w") as f:
            yaml.safe_dump(meta, f, default_flow_style=False, sort_keys=False)
        if self.scada is not None:
            self.scada.reset_index(drop=False).to_csv(
                (save_path / scada).with_suffix(".csv"), index=False
            )
        if self.status is not None:
            self.status.reset_index(drop=False).to_csv(
                (save_path / status).with_suffix(".csv"), index=False
            )
        if self.tower is not None:
            self.tower.reset_index(drop=False).to_csv(
                (save_path / tower).with_suffix(".csv"), index=False
            )
        if self.meter is not None:
            self.meter.reset_index(drop=False).to_csv(
                (save_path / meter).with_suffix(".csv"), index=False
            )
        if self.curtail is not None:
            self.curtail.reset_index(drop=False).to_csv(
                (save_path / curtail).with_suffix(".csv"), index=False
            )
        if self.asset is not None:
            self.asset.reset_index(drop=False).to_csv(
                (save_path / asset).with_suffix(".csv"), index=False
            )
        if self.reanalysis is not None:
            for name, df in self.reanalysis.items():
                df.reset_index(drop=False).to_csv(
                    (save_path / f"{reanalysis}_{name}").with_suffix(".csv"), index=False
                )

    def _validate_column_names(self, category: str = "all") -> dict[str, list[str]]:
        """Validates that the column names in each of the data types matches the mapping
        provided in the `metadata` object.

        Args:
            category (str, optional): _description_. Defaults to "all".

        Returns:
            dict[str, list[str]]: _description_
        """
        column_map = self.metadata.column_map

        missing_cols = {}
        for name, df in self.data_dict.items():
            if category != "all" and category != "name":
                # Skip any irrelevant columns if not processing all data types
                continue

            if name == "reanalysis":
                for sub_name, df in df.items():
                    missing_cols[f"{name}-{sub_name}"] = column_validator(
                        df, column_names=column_map[name][sub_name]
                    )
                continue

            missing_cols[name] = column_validator(df, column_names=column_map[name])
        return missing_cols

    def _validate_dtypes(self, category: str = "all") -> dict[str, list[str]]:
        """Validates the dtype for each column for the specified `category`.

        Args:
            category (`str`, optional): The name of the data that should be
                checked, or "all" to validate all of the data types. Defaults to "all".

        Returns:
            (`dict[str, list[str]]`): A dictionary of each data type and any
                columns that  don't match the required dtype and can't be converted to
                it successfully.
        """
        # Create a new mapping of the data's column names to the expected dtype
        # TODO: Consider if this should be a encoded in the metadata/plantdata object elsewhere
        column_name_map = self.metadata.column_map
        column_dtype_map = self.metadata.dtype_map
        column_map = {}
        for name in column_name_map:
            if name == "reanalysis":
                column_map["reanalysis"] = {}
                for name in column_name_map["reanalysis"]:
                    column_map["reanalysis"][name] = dict(
                        zip(
                            column_name_map["reanalysis"][name].values(),
                            column_dtype_map["reanalysis"][name].values(),
                        )
                    )
            else:
                column_map[name] = dict(
                    zip(column_name_map[name].values(), column_dtype_map[name].values())
                )

        error_cols = {}
        for name, df in self.data_dict.items():
            if category != "all" and category != name:
                # Skip irrelevant data types if not checking all data types
                continue

            if name == "reanalysis":
                for sub_name, df in df.items():
                    error_cols[f"{name}-{sub_name}"] = dtype_converter(
                        df, column_types=column_map[name][sub_name]
                    )
                continue

            error_cols[name] = dtype_converter(df, column_types=column_map[name])
        return error_cols

    def _validate_frequency(self, category: str = "all") -> list[str]:
        """Internal method to check the actual datetime frequencies against the required
        frequencies for the specified analysis types, and produces a list of data types
        that do not meet the frequency criteria.

        Args:
            category (`str`, optional): The data type category. Defaults to "all".

        Returns:
            list[str]: The list of data types that don't meet the required datetime frequency.
        """
        frequency_requirements = self.metadata.frequency_requirements(self.analysis_type)

        # Collect all the frequencies for each of the data types
        data_dict = self.data_dict
        actual_frequencies = {}

        for name, df in data_dict.items():
            if df is None:
                continue

            if name in ("scada", "status", "tower"):
                actual_frequencies[name] = ts.determine_frequency(df, "time")
            elif name in ("meter", "curtail"):
                actual_frequencies[name] = ts.determine_frequency(df)
            elif name == "reanalysis":
                actual_frequencies["reanalysis"] = {}
                for sub_name, df in data_dict[name].items():
                    actual_frequencies["reanalysis"][sub_name] = ts.determine_frequency(df)

        invalid_freq = {}
        for name, freq in actual_frequencies.items():
            if category != "all" and category != name:
                # If only checking one data type, then skip all others
                continue
            if name == "reanalysis":
                for sub_name, freq in freq.items():
                    is_valid = frequency_validator(freq, frequency_requirements.get(name), True)
                    is_valid |= frequency_validator(freq, frequency_requirements.get(name), False)
                    if not is_valid:
                        invalid_freq.update({f"{name}-{sub_name}": freq})
                continue
            is_valid = frequency_validator(freq, frequency_requirements.get(name), True)
            is_valid |= frequency_validator(freq, frequency_requirements.get(name), False)
            if not is_valid:
                invalid_freq.update({name: freq})

        return invalid_freq

    def validate(self, metadata: Optional[dict | str | Path | PlantMetaData] = None) -> None:
        """Secondary method to validate the plant data objects after loading or changing
        data with option to provide an updated `metadata` object/file as well

        Args:
            metadata (Optional[dict]): Updated metadata object, dictionary, or file to
                create the updated metadata for data validation, which should align with
                the mapped column names during initialization.

        Raises:
            ValueError: Raised at the end if errors are caught in the validation steps.
        """
        # Put the index columns back into the column space to ensure success of re-validation
        self._unset_index_columns()

        # Initialization will have converted the column naming convention, but an updated
        # metadata should account for the renaming of the columns
        if metadata is None:
            self.update_column_names(to_original=True)
        else:
            self.metadata = metadata

        # Reset the index columns to be part of the columns space so the validations still work

        self._errors = {
            "missing": self._validate_column_names(),
            "dtype": self._validate_dtypes(),
        }

        self._set_index_columns()
        self._errors["frequency"] = self._validate_frequency()

        # TODO: Check for extra columns?
        # TODO: Define other checks?

        error_message = _compose_error_message(self._errors, self.analysis_type)
        if error_message:
            raise ValueError(error_message)
        self.update_column_names()

    def _calculate_reanalysis_columns(self) -> None:
        """Calculates extra variables such as wind direction from the provided
        reanalysis data if they don't already exist.
        """
        if self.reanalysis is None:
            return
        reanalysis = {}
        for name, df in self.reanalysis.items():
            col_map = self.metadata.reanalysis[name].col_map
            u = col_map["WMETR_HorWdSpdU"]
            v = col_map["WMETR_HorWdSpdV"]
            has_u_v = (u in df) & (v in df)

            ws = col_map["WMETR_HorWdSpd"]
            if ws not in df and has_u_v:
                df[ws] = np.sqrt(df[u].values ** 2 + df[v].values ** 2)

            wd = col_map["WMETR_HorWdDir"]
            if wd not in df and has_u_v:
                # TODO: added .values to fix an issue where df[u] and df[v] with ANY NaN values would cause df[wd]
                # to be all NaN. Is there a better to fix this?
                df[wd] = met.compute_wind_direction(df[u], df[v]).values

            dens = col_map["WMETR_AirDen"]
            sp = col_map["WMETR_EnvPres"]
            temp = col_map["WMETR_EnvTmp"]
            has_sp_temp = (sp in df) & (temp in df)
            if dens not in df and has_sp_temp:
                df[dens] = met.compute_air_density(df[temp], df[sp])

            reanalysis[name] = df
        self.reanalysis = reanalysis

    def parse_asset_geometry(
        self,
        reference_system: str = "epsg:4326",
        utm_zone: int = None,
        reference_longitude: Optional[float] = None,
    ) -> None:
        """Calculate UTM coordinates from latitude/longitude.

        The UTM system divides the Earth into 60 zones, each 6deg of longitude in width. Zone 1
        covers longitude 180deg to 174deg W; zone numbering increases eastward to zone 60, which
        covers longitude 174deg E to 180deg. The polar regions south of 80deg S and north of 84deg N
        are excluded.

        Ref: http://geopandas.org/projections.html

        Args:
            reference_system (:obj:`str`, optional): Used to define the coordinate reference system (CRS).
                Defaults to the European Petroleum Survey Group (EPSG) code 4326 to be used with
                the World Geodetic System reference system, WGS 84.
            utm_zone (:obj:`int`, optional): UTM zone. If set to None (default), then calculated from
                the longitude.
            reference_longitude (:obj:`float`, optional): Reference longitude for calculating the UTM zone. If
                None (default), then taken as the average longitude of all assets.

        Returns: None
            Sets the asset "geometry" column.
        """
        if utm_zone is None:
            # calculate zone
            if reference_longitude is None:
                longitude = self.asset[self.metadata.asset.longitude].mean()
            utm_zone = int(np.floor((180 + longitude) / 6.0)) + 1

        to_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        transformer = Transformer.from_crs(reference_system.upper(), to_crs)
        lats, lons = transformer.transform(
            self.asset[self.metadata.asset.latitude].values,
            self.asset[self.metadata.asset.longitude].values,
        )

        # TODO: Should this get a new name that's in line with the -25 convention?
        self.asset["geometry"] = [Point(lat, lon) for lat, lon in zip(lats, lons)]

    def update_column_names(self, to_original: bool = False) -> None:
        """Renames the columns of each dataframe to the be the keys from the
        `metadata.xx.col_map` that was passed during initialization.

        Args:
            to_original (bool, optional): An indicator to map the column names back to
                the originally passed values. Defaults to False.
        """
        meta = self.metadata
        reverse = not to_original  # flip the boolean to correctly map between the col_map entries

        if self.scada is not None:
            self.scada = rename_columns(self.scada, meta.scada.col_map, reverse=reverse)
        if self.meter is not None:
            self.meter = rename_columns(self.meter, meta.meter.col_map, reverse=reverse)
        if self.tower is not None:
            self.tower = rename_columns(self.tower, meta.tower.col_map, reverse=reverse)
        if self.status is not None:
            self.status = rename_columns(self.status, meta.status.col_map, reverse=reverse)
        if self.curtail is not None:
            self.curtail = rename_columns(self.curtail, meta.curtail.col_map, reverse=reverse)
        if self.asset is not None:
            self.asset = rename_columns(self.asset, meta.asset.col_map, reverse=reverse)
        if self.reanalysis is not None:
            reanalysis = {}
            for name, df in self.reanalysis.items():
                reanalysis[name] = rename_columns(
                    df, meta.reanalysis[name].col_map, reverse=reverse
                )
            self.reanalysis = reanalysis

    def _calculate_turbine_energy(self) -> None:
        energy_col = self.metadata.scada.WTUR_SupWh
        power_col = self.metadata.scada.WTUR_W
        frequency = self.metadata.scada.frequency
        self.scada[energy_col] = convert_power_to_energy(self.scada[power_col], frequency)

    @property
    def turbine_ids(self) -> np.ndarray:
        """The 1D array of turbine IDs. This is created from the `asset` data, or unique IDs from the
        SCADA data, if `asset` is undefined.
        """
        if self.asset is None:
            return self.scada.index.get_level_values("id").unique()
        return self.asset.loc[self.asset["type"] == "turbine"].index.values

    @property
    def n_turbines(self) -> int:
        """The number of turbines contained in the data."""
        return self.turbine_ids.size

    def turbine_df(self, turbine_id: str) -> pd.DataFrame:
        """Filters `scada` on a single `turbine_id` and returns the filtered data frame.

        Args:
            turbine_id (str): The ID of the turbine to retrieve its data.

        Returns:
            pd.DataFrame: The turbine-specific SCADA data frame.
        """
        if self.scada is None:
            raise AttributeError("This method can't be used unless `scada` data is provided.")
        return self.scada.xs(turbine_id, level=1)

    @property
    def tower_ids(self) -> np.ndarray:
        """The 1D array of met tower IDs. This is created from the `asset` data, or unique IDs from the
        tower data, if `asset` is undefined.
        """
        if self.asset is None:
            return self.tower.index.get_level_values("id").unique()
        return self.asset.loc[self.asset["type"] == "tower"].index.values

    @property
    def n_towers(self) -> int:
        """The number of met towers contained in the data."""
        return self.tower_ids.size

    def tower_df(self, tower_id: str) -> pd.DataFrame:
        """Filters `tower` on a single `tower_id` and returns the filtered data frame.

        Args:
            tower_id (str): The ID of the met tower to retrieve its data.

        Returns:
            pd.DataFrame: The met tower-specific data frame.
        """
        if self.tower is None:
            raise AttributeError("This method can't be used unless `tower` data is provided.")
        return self.tower.xs(tower_id, level=1)

    @property
    def asset_ids(self) -> np.ndarray:
        """The ID array of turbine and met tower IDs. This is created from the `asset` data, or unique
        IDs from both the SCADA data and tower data, if `asset` is undefined.
        """
        if self.asset is None:
            return np.concatenate([self.turbine_ids, self.tower_ids])
        return self.asset.index.values

    # NOTE: v2 AssetData methods

    def calculate_asset_distance_matrix(self) -> pd.DataFrame:
        """Calculates the distance between all assets on the site with `np.inf` for the distance
        between an asset and itself.

        Returns:
            pd.DataFrame: Dataframe containing distances between each pair of assets
        """
        ix = self.asset.index.values
        distance = (
            pd.DataFrame(
                [i, j, self.asset.loc[i, "geometry"].distance(self.asset.loc[j, "geometry"])]
                for i, j in itertools.combinations(ix, 2)
            )
            .pivot(index=0, columns=1, values=2)
            .rename_axis(index={0: None}, columns={1: None})
            .fillna(0)
            .loc[ix[:-1], ix[1:]]
        )

        # Insert the first column and last row because the self-self combinations are not produced in the above
        distance.insert(0, ix[0], 0.0)
        distance.loc[ix[-1]] = 0

        # Maintain v2 compatibility of np.inf for the diagonal
        distance = distance + distance.values.T - np.diag(np.diag(distance.values))
        np.fill_diagonal(distance.values, np.inf)
        self.asset_distance_matrix = distance

    def turbine_distance_matrix(self, turbine_id: str = None) -> pd.DataFrame:
        """Returns the distances between all turbines in the plant with `np.inf` for the distance
        between a turbine and itself.

        Args:
            turbine_id (str, optional): Specific turbine ID for which the distances to other turbines
                are returned. If None, a matrix containing the distances between all pairs of turbines
                is returned. Defaults to None.
        Returns:
            pd.DataFrame: Dataframe containing distances between each pair of turbines
        """
        if self.asset_distance_matrix.size == 0:
            self.calculate_asset_distance_matrix()

        row_ix = self.turbine_ids if turbine_id is None else turbine_id
        return self.asset_distance_matrix.loc[row_ix, self.turbine_ids]

    def tower_distance_matrix(self, tower_id: str = None) -> pd.DataFrame:
        """Returns the distances between all towers in the plant with `np.inf` for the distance
        between a tower and itself.

        Args:
            tower_id (str, optional): Specific tower ID for which the distances to other towers
                are returned. If None, a matrix containing the distances between all pairs of towers
                is returned. Defaults to None.
        Returns:
            pd.DataFrame: Dataframe containing distances between each pair of towers
        """
        if self.asset_distance_matrix.size == 0:
            self.calculate_asset_distance_matrix()

        row_ix = self.tower_ids if tower_id is None else tower_id
        return self.asset_distance_matrix.loc[row_ix, self.tower_ids]

    def calculate_asset_direction_matrix(self) -> pd.DataFrame:
        """Calculates the direction between all assets on the site with `np.inf` for the direction
        between an asset and itself, for all assets.

        Returns:
            pd.DataFrame: Dataframe containing directions between each pair of assets (defined as the direction
                from the asset given by the row index to the asset given by the column index, relative to north)
        """
        ix = self.asset.index.values
        direction = (
            pd.DataFrame(
                [
                    i,
                    j,
                    np.degrees(
                        np.arctan2(
                            self.asset.loc[j, "geometry"].x - self.asset.loc[i, "geometry"].x,
                            self.asset.loc[j, "geometry"].y - self.asset.loc[i, "geometry"].y,
                        )
                    )
                    % 360.0,
                ]
                for i, j in itertools.combinations(ix, 2)
            )
            .pivot(index=0, columns=1, values=2)
            .rename_axis(index={0: None}, columns={1: None})
            .fillna(0)
            .loc[ix[:-1], ix[1:]]
        )

        # Insert the first column and last row because the self-self combinations are not produced in the above
        direction.insert(0, ix[0], 0.0)
        direction.loc[ix[-1]] = 0

        # Maintain v2 compatibility of np.inf for the diagonal
        direction = (
            direction
            + np.triu((direction.values - 180.0) % 360.0, 1).T
            - np.diag(np.diag(direction.values))
        )
        np.fill_diagonal(direction.values, np.inf)
        self.asset_direction_matrix = direction

    def turbine_direction_matrix(self, turbine_id: str = None) -> pd.DataFrame:
        """Returns the directions between all turbines in the plant with `np.inf` for the direction
        between a turbine and itself.

        Args:
            turbine_id (str, optional): Specific turbine ID for which the directions to other turbines
                are returned. If None, a matrix containing the directions between all pairs of turbines
                is returned. Defaults to None.
        Returns:
            pd.DataFrame: Dataframe containing directions between each pair of turbines (defined as the
                direction from the turbine given by the row index to the turbine given by the column
                index, relative to north)
        """
        if self.asset_direction_matrix.size == 0:
            self.calculate_asset_direction_matrix()

        row_ix = self.turbine_ids if turbine_id is None else turbine_id
        return self.asset_direction_matrix.loc[row_ix, self.turbine_ids]

    def tower_direction_matrix(self, tower_id: str = None) -> pd.DataFrame:
        """Returns the directions between all towers in the plant with `np.inf` for the direction
        between a tower and itself.

        Args:
            tower_id (str, optional): Specific tower ID for which the directions to other towers
                are returned. If None, a matrix containing the directions between all pairs of towers
                is returned. Defaults to None.
        Returns:
            pd.DataFrame: Dataframe containing directions between each pair of towers (defined as the
                direction from the tower given by the row index to the tower given by the column
                index, relative to north)
        """
        if self.asset_direction_matrix.size == 0:
            self.calculate_asset_direction_matrix()

        row_ix = self.tower_ids if tower_id is None else tower_id
        return self.asset_direction_matrix.loc[row_ix, self.tower_ids]

    def get_freestream_turbines(
        self, wd: float, freestream_method: str = "sector", sector_width: float = 90.0
    ):
        """
        Returns a list of freestream (unwaked) turbines for a given wind direction. Freestream turbines can be
        identified using different methods ("sector" or "IEC" methods). For the sector method, if there are any
        turbines upstream of a turbine within a fixed wind direction sector centered on the wind direction of interest,
        defined by the sector_width argument, the turbine is considered waked. The IEC method uses the freestream
        definition provided in Annex A of IEC 61400-12-1 (2005).

        Args:
            wd (float): Wind direction to identify freestream turbines for (degrees)
            freestream_method (str, optional): Method used to identify freestream turbines
                ("sector" or "IEC"). Defaults to "sector".
            sector_width (float, optional): Width of wind direction sector centered on the wind direction of
                interest used to determine whether a turbine is waked for the "sector" method (degrees). For a given
                turbine, if any other upstream turbines are located within the sector, then the turbine is considered
                waked. Defaults to 90 degrees.
        Returns:
            list: List of freestream turbine asset IDs
        """
        turbine_direction_matrix = self.turbine_direction_matrix()

        if freestream_method == "sector":
            # find turbines for which no other upstream turbines are within half of the sector width of the specified
            # wind direction
            freestream_indices = np.all(
                (np.abs(met.wrap_180(wd - turbine_direction_matrix.values)) > 0.5 * sector_width)
                | np.diag(np.ones(len(turbine_direction_matrix), dtype=bool)),
                axis=1,
            )
        elif freestream_method == "IEC":
            # find freestream turbines according to the definition in Annex A of IEC 61400-12-1 (2005)
            turbine_distance_matrix = self.turbine_distance_matrix()

            # normalize distances by rotor diameters of upstream turbines
            rotor_diameters_vector = self.asset.loc[
                turbine_direction_matrix.index, "rotor_diameter"
            ].values
            rotor_diameters = np.ones((len(turbine_direction_matrix), 1)) * rotor_diameters_vector
            turbine_distance_matrix /= rotor_diameters

            freestream_indices = np.all(
                (
                    (turbine_distance_matrix.values > 2)
                    & (
                        np.abs(met.wrap_180(wd - turbine_direction_matrix.values))
                        > 0.5
                        * (
                            1.3 * np.degrees(np.arctan(2.5 / turbine_distance_matrix.values + 0.15))
                            + 10
                        )
                    )
                )
                | (turbine_distance_matrix.values > 20)
                | (turbine_distance_matrix.values < 0),
                axis=1,
            )
        else:
            raise ValueError(
                'Invalid freestream method. Currently, "sector" and "IEC" are supported.'
            )

        return list(self.asset.index[freestream_indices])

    def calculate_nearest_neighbor(
        self, turbine_ids: list | np.ndarray = None, tower_ids: list | np.ndarray = None
    ) -> None:
        """Finds nearest turbine and met tower neighbors all of the available turbines and towers
        in `asset` or as defined in `turbine_ids` and `tower_ids`.

        Args:
            turbine_ids (list | np.ndarray, optional): A list of turbine IDs, if not using all
                turbines in the data. Defaults to None.
            tower_ids (list | np.ndarray, optional): A list of met tower IDs, if not using all
                met towers in the data. Defaults to None.

        Returns: None
            Creates the "nearest_turbine_id" and "nearest_tower_id" column in `asset`.
        """

        # Get the valid IDs for both the turbines and towers
        ix_turb = self.turbine_ids if turbine_ids is None else np.array(turbine_ids)
        ix_tower = self.tower_ids if tower_ids is None else np.array(tower_ids)
        ix = np.concatenate([ix_turb, ix_tower])

        distance = self.asset_distance_matrix.loc[ix, ix]

        nearest_turbine = distance[ix_turb].values.argsort(axis=1)
        nearest_turbine = pd.DataFrame(
            distance.columns.values[nearest_turbine], index=distance.index
        ).loc[ix, 0]

        nearest_tower = distance[ix_tower].values.argsort(axis=1)
        nearest_tower = pd.DataFrame(
            distance.columns.values[nearest_tower], index=distance.index
        ).loc[ix, 0]

        self.asset.loc[ix, "nearest_turbine_id"] = nearest_turbine.values
        self.asset.loc[ix, "nearest_tower_id"] = nearest_tower.values

    def nearest_turbine(self, id: str) -> str:
        """Finds the nearest turbine to the provided `id`.

        Args:
            id (str): A valid `asset` `id`.

        Returns:
            str: The turbine `id` closest to the provided `id`.
        """
        if "nearest_turbine_id" not in self.asset.columns:
            self.calculate_nearest_neighbor()
        return self.asset.loc[id, "nearest_turbine_id"].values[0]

    def nearest_tower(self, id: str) -> str:
        """Finds the nearest tower to the provided `id`.

        Args:
            id (str): A valid `asset` `id`.

        Returns:
            str: The tower `id` closest to the provided `id`.
        """
        if "nearest_tower_id" not in self.asset.columns:
            self.calculate_nearest_neighbor()
        return self.asset.loc[id, "nearest_tower_id"].values[0]


# **********************************************************
# Define additional class methods for custom loading methods
# **********************************************************


def from_entr(
    thrift_server_host: str = "localhost",
    thrift_server_port: int = 10000,
    database: str = "entr_warehouse",
    wind_plant: str = "",
    aggregation: str = "",
    date_range: list = None,
):
    """
    from_entr

    Load a PlantData object from data in an entr_warehouse.

    Args:
        thrift_server_url(str): URL of the Apache Thrift server
        database(str): Name of the Hive database
        wind_plant(str): Name of the wind plant you'd like to load
        aggregation: Not yet implemented
        date_range: Not yet implemented

    Returns:
        plant(PlantData): An OpenOA PlantData object.
    """
    from pyhive import hive

    conn = hive.Connection(host=thrift_server_host, port=thrift_server_port)

    scada_query = """SELECT Wind_turbine_name as Wind_turbine_name,
            Date_time as Date_time,
            cast(P_avg as float) as P_avg,
            cast(Power_W as float) as Power_W,
            cast(Ws_avg as float) as Ws_avg,
            Wa_avg as Wa_avg,
            Va_avg as Va_avg,
            Ya_avg as Ya_avg,
            Ot_avg as Ot_avg,
            Ba_avg as Ba_avg

    FROM entr_warehouse.la_haute_borne_scada_for_openoa
    """

    plant = PlantData()

    plant.scada.df = pd.read_sql(scada_query, conn)

    conn.close()

    return plant


setattr(PlantData, "from_entr", classmethod(from_entr))
setattr(PlantData, "attach_eia_data", attach_eia_data)
