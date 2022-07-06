from __future__ import annotations

import json
import itertools
from typing import Callable, Optional, Sequence
from pathlib import Path

import attr
import yaml
import numpy as np
import pandas as pd
import pyspark as spark
from attr import define

import openoa.utils.met_data_processing as met
from openoa.utils.reanalysis_downloading import download_reanalysis_data_planetos


# PlantData V2 with Attrs Dataclass

# Datetime frequency checks
_at_least_monthly = ("M", "MS", "W", "D", "H", "T", "min", "S", "L", "ms", "U", "us", "N")
_at_least_daily = ("D", "H", "T", "min", "S", "L", "ms", "U", "us", "N")
_at_least_hourly = ("H", "T", "min", "S", "L", "ms", "U", "us", "N")

ANALYSIS_REQUIREMENTS = {
    "MonteCarloAEP": {
        "meter": {
            "columns": ["energy"],
            "freq": _at_least_monthly,
        },
        "curtail": {
            "columns": ["availability", "curtailment"],
            "freq": _at_least_monthly,
        },
        "reanalysis": {
            "columns": ["windspeed", "density"],
            "conditional_columns": {
                "reg_temperature": ["temperature"],
                "reg_winddirection": ["windspeed_u", "windspeed_v"],
            },
            "freq": _at_least_monthly,
        },
    },
    "TurbineLongTermGrossEnergy": {
        "scada": {
            "columns": ["id", "windspeed", "power"],  # TODO: wtur_W_avg vs energy_kwh ?
            "freq": _at_least_daily,
        },
        "reanalysis": {
            "columns": ["windspeed", "wind_direction", "density"],
            "freq": _at_least_daily,
        },
    },
    "ElectricalLosses": {
        "scada": {
            "columns": ["energy"],
            "freq": _at_least_daily,
        },
        "meter": {
            "columns": ["energy"],
            "freq": _at_least_monthly,
        },
    },
}


def analysis_type_validator(
    instance: PlantData, attribute: attr.Attribute, value: list[str]
) -> None:
    """Validates the input from `PlantData` against the analysis requirements in
    `ANALYSIS_REQUIREMENTS`. If there is an error, then it gets added to the
    `PlantData._errors` dictionary to be raised in the post initialization hook.

    Args:
        instance (PlantData): The PlantData object.
        attribute (attr.Attribute): The converted `analysis_type` attribute object.
        value (list[str]): The input value from `analysis_type`.
    """
    if None in value:
        UserWarning("`None` was provided to `analysis_type`, so no validation will occur.")

    valid_types = [*ANALYSIS_REQUIREMENTS] + ["all", None]
    incorrect_types = set(value).difference(set(valid_types))
    if incorrect_types:
        raise ValueError(
            f"{attribute.name} input: {incorrect_types} is invalid, must be one of 'all' or a combination of: {[*ANALYSIS_REQUIREMENTS]}"
        )


def frequency_validator(
    actual_freq: str, desired_freq: Optional[str | set[str]], exact: bool
) -> bool:
    """Helper function to check if the actual datetime stamp frequency is valid compared
    to what is required.

    Args:
        actual_freq (str): The frequency of the datetime stamp, or `df.index.freq`.
        desired_freq (Optional[str  |  set[str]]): Either the exact frequency required,
            or a set of options that are also valid, in which case any numeric
            information encoded in `actual_freq` will be dropped.
        exact (bool): If the provided frequency codes should be exact matches (`True`),
            or, if `False`, the check should be for a combination of matches.

    Returns:
        bool: If the actual datetime frequency is sufficient, per the match requirements.
    """
    if exact:
        return actual_freq in desired_freq

    if desired_freq is None:
        return True

    if actual_freq is None:
        return False

    actual_freq = "".join(filter(str.isalpha, actual_freq))
    return actual_freq in desired_freq


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


#########################################
# Define the meta data validation classes
#########################################
@define(auto_attribs=True)
class SCADAMetaData(FromDictMixin):
    """A metadata schematic to create the necessary column mappings and other validation
    components, or other data about the SCADA data, that will contribute to a larger
    plant metadata schema/routine.

    Args:
        time (str): The datetime stamp for the SCADA data, by default "time". This data should be of
            type: `np.datetime64[ns]`. Additional columns describing the datetime stamps
            are: `frequency`
    """

    # DataFrame columns
    time: str = attr.ib(default="time")
    id: str = attr.ib(default="id")
    power: str = attr.ib(default="power")
    windspeed: str = attr.ib(default="windspeed")
    wind_direction: str = attr.ib(default="wind_direction")
    status: str = attr.ib(default="status")
    pitch: str = attr.ib(default="pitch")
    temperature: str = attr.ib(default="temperature")

    # Data about the columns
    frequency: str = attr.ib(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = attr.ib(default="scada", init=False)
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
        default=dict(
            time=np.datetime64,
            id=str,
            power=float,
            windspeed=float,
            wind_direction=float,
            status=str,
            pitch=float,
            temperature=float,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = attr.ib(
        default=dict(
            time="datetim64[ns]",
            id=None,
            power="kW",
            windspeed="m/s",
            wind_direction="deg",
            status=None,
            pitch="deg",
            temperature="C",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            id=self.id,
            power=self.power,
            windspeed=self.windspeed,
            wind_direction=self.wind_direction,
            status=self.status,
            pitch=self.pitch,
            temperature=self.temperature,
        )


@define(auto_attribs=True)
class MeterMetaData(FromDictMixin):

    # DataFrame columns
    time: str = attr.ib(default="time")
    power: str = attr.ib(default="power")
    energy: str = attr.ib(default="energy")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = attr.ib(default="meter", init=False)
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
        default=dict(
            time=np.datetime64,
            power=float,
            energy=float,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = attr.ib(
        default=dict(
            time="datetim64[ns]",
            power="kW",
            energy="kW",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            power=self.power,
            energy=self.energy,
        )


@define(auto_attribs=True)
class TowerMetaData(FromDictMixin):
    # DataFrame columns
    time: str = attr.ib(default="time")
    id: str = attr.ib(default="id")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = attr.ib(default="tower", init=False)
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
        default=dict(
            time=np.datetime64,
            id=str,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = attr.ib(
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
class StatusMetaData(FromDictMixin):
    # DataFrame columns
    time: str = attr.ib(default="time")
    id: str = attr.ib(default="id")
    status_id: str = attr.ib(default="status_id")
    status_code: str = attr.ib(default="status_code")
    status_text: str = attr.ib(default="status_text")

    # Data about the columns
    frequency: str = attr.ib(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = attr.ib(default="status", init=False)
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
        default=dict(
            time=np.datetime64,
            id=str,
            status_id=np.int64,
            status_code=np.int64,
            status_text=str,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = attr.ib(
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
class CurtailMetaData(FromDictMixin):
    # DataFrame columns
    time: str = attr.ib(default="time")
    curtailment: str = attr.ib(default="curtailment")
    availability: str = attr.ib(default="availability")
    net_energy: str = attr.ib(default="net_energy")

    # Data about the columns
    frequency: str = attr.ib(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = attr.ib(default="curtail", init=False)
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
        default=dict(
            time=np.datetime64,
            curtailment=float,
            availability=float,
            net_energy=float,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = attr.ib(
        default=dict(
            time="datetim64[ns]",
            curtailment=float,
            availability=float,
            net_energy="kW",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            curtailment=self.curtailment,
            availability=self.availability,
            net_energy=self.net_energy,
        )


@define(auto_attribs=True)
class AssetMetaData(FromDictMixin):
    # DataFrame columns
    id: str = attr.ib(default="id")
    latitude: str = attr.ib(default="latitude")
    longitude: str = attr.ib(default="longitude")
    rated_power: str = attr.ib(default="rated_power")
    hub_height: str = attr.ib(default="hub_height")
    rotor_diameter: str = attr.ib(default="rotor_diameter")
    elevation: str = attr.ib(default="elevation")
    type: str = attr.ib(default="type")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = attr.ib(default="asset", init=False)
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
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
    units: dict = attr.ib(
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
            hub_height=self.rated_power,
            rotor_diameter=self.rated_power,
            elevation=self.rated_power,
            type=self.type,
        )


@define(auto_attribs=True)
class ReanalysisMetaData(FromDictMixin):
    # DataFrame columns
    time: str = attr.ib(default="time")
    windspeed: str = attr.ib(default="windspeed")
    windspeed_u: str = attr.ib(default="windspeed_u")
    windspeed_v: str = attr.ib(default="windspeed_v")
    wind_direction: str = attr.ib(default="wind_direction")
    temperature: str = attr.ib(default="temperature")
    density: str = attr.ib(default="density")
    surface_pressure: str = attr.ib(default="surface_pressure")

    # Data about the columns
    frequency: str = attr.ib(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    name: str = attr.ib(default="reanalysis", init=False)
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
        default=dict(
            time=np.datetime64,
            windspeed=float,
            windspeed_u=float,
            windspeed_v=float,
            wind_direction=float,
            temperature=float,
            density=float,
            surface_pressure=float,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = attr.ib(
        default=dict(
            time="datetim64[ns]",
            windspeed="m/s",
            windspeed_u="m/s",
            windspeed_v="m/s",
            wind_direction="deg",
            temperature="K",
            density="kg/m^3",
            surface_pressure="Pa",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            windspeed=self.windspeed,
            windspeed_u=self.windspeed_u,
            windspeed_v=self.windspeed_v,
            wind_direction=self.wind_direction,
            temperature=self.temperature,
            density=self.density,
            surface_pressure=self.surface_pressure,
        )


def convert_reanalysis(value: dict[str, dict]):
    return {k: ReanalysisMetaData.from_dict(v) for k, v in value.items()}


@define(auto_attribs=True)
class PlantMetaData(FromDictMixin):
    """Composese the individual metadata/validation requirements from each of the
    individual data "types" that can compose a `PlantData` object.
    """

    latitude: float = attr.ib(default=0, converter=float)
    longitude: float = attr.ib(default=0, converter=float)
    scada: SCADAMetaData = attr.ib(default={}, converter=SCADAMetaData.from_dict)
    meter: MeterMetaData = attr.ib(default={}, converter=MeterMetaData.from_dict)
    tower: TowerMetaData = attr.ib(default={}, converter=TowerMetaData.from_dict)
    status: StatusMetaData = attr.ib(default={}, converter=StatusMetaData.from_dict)
    curtail: CurtailMetaData = attr.ib(default={}, converter=CurtailMetaData.from_dict)
    asset: AssetMetaData = attr.ib(default={}, converter=AssetMetaData.from_dict)
    reanalysis: dict[str, ReanalysisMetaData] = attr.ib(default={}, converter=convert_reanalysis)

    @property
    def column_map(self):
        values = dict(
            scada=self.scada.col_map,
            meter=self.meter.col_map,
            tower=self.tower.col_map,
            status=self.status.col_map,
            asset=self.asset.col_map,
            curtail=self.curtail.col_map,
            reanalysis={k: v.col_map for k, v in self.reanalysis.items()},
        )
        return values

    @property
    def type_map(self):
        types = dict(
            scada=self.scada.dtypes,
            meter=self.meter.dtypes,
            tower=self.tower.dtypes,
            status=self.status.dtypes,
            asset=self.asset.dtypes,
            curtail=self.curtail.dtypes,
            reanalysis={k: v.dtypes for k, v in self.reanalysis.items()},
        )
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
        metadata_file = Path(metadata_file).resolve()
        if not metadata_file.is_file():
            raise FileExistsError(f"Input JSON file: {metadata_file} is an invalid input.")

        with open(metadata_file) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, metadata_file: str | Path) -> PlantMetaData:
        metadata_file = Path(metadata_file).resolve()
        if not metadata_file.is_file():
            raise FileExistsError(f"Input YAML file: {metadata_file} is an invalid input.")

        with open(metadata_file) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def load(cls, data: str | Path | dict | PlantMetaData) -> PlantMetaData:
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


####################################################
# Define the data validator and conversion functions
####################################################


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
                df[column] = pd.DatetimeIndex(pd.to_datetime(df[column], utc=True))
            except Exception as e:  # noqa: disable=E722
                errors.append(column)
            continue
        try:
            df[column] = df[column].astype(new_type)
        except:  # noqa: disable=E722
            errors.append(column)

    return errors


def analysis_filter(error_dict: dict, analysis_types: list[str] = ["all"]) -> dict:
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

    # Filter the incorrect frequencies, so only analysis-specific categories are provided
    # TODO
    # error_dict["frequency"] = {
    #     key: value["freq"] for key, value in requirements if value["freq"] not in frequency
    # }

    return error_dict


def compose_error_message(error_dict: dict, analysis_types: list[str] = ["all"]) -> str:
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
        error_dict = analysis_filter(error_dict, analysis_types)

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
    messages.extend([f"`{name}` data is of the wrong frequecy" for name in error_dict["frequency"]])
    return "\n".join(messages)


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
                values. Defaults to True.

        Returns:
            pd.DataFrame: Input DataFrame with remapped column names.
    """
    if reverse:
        col_map = {v: k for k, v in col_map.items()}
    return df.rename(columns=col_map)


############################
# Define the PlantData class
############################


@define(auto_attribs=True)
class PlantData:
    """Data object for operational wind plant data, which can serialize all of these
    structures and reload them them from the cache as needed.

    This class holds references to all tables associated with a wind plant. The tables
    are grouped by type:
        - `scada`
        - `meter`
        - `tower`
        - `status`
        - `curtail`
        - `asset`
        - `reanalysis`

    Parameters
    ----------
    metadata : PlantMetaData
        A nested dictionary of the schema definition for each of the data types that
        will be input. See `SCADAMetaData`, etc. for more information.  <-- TODO
    scada : pd.DataFrame
        The SCADA data to be used for analyis. See `SCADAMetaData` for more details
        on the required columns, and other conventions
    TODO: FINISH THE DOCSTRING

    Raises:
        ValueError: Raised if any column names are missing in the input data, as
            specified in the appropriate schema
    """

    metadata: PlantMetaData = attr.ib(
        default={}, converter=PlantMetaData.load, on_setattr=[attr.converters, attr.validators]
    )
    analysis_type: list[str] | None = attr.ib(
        default=None,
        converter=convert_to_list,
        validator=analysis_type_validator,
        on_setattr=[attr.setters.convert, attr.setters.validate],
    )
    scada: pd.DataFrame | None = attr.ib(default=None, converter=load_to_pandas)
    meter: pd.DataFrame | None = attr.ib(default=None, converter=load_to_pandas)
    tower: pd.DataFrame | None = attr.ib(default=None, converter=load_to_pandas)
    status: pd.DataFrame | None = attr.ib(default=None, converter=load_to_pandas)
    curtail: pd.DataFrame | None = attr.ib(default=None, converter=load_to_pandas)
    asset: pd.DataFrame | None = attr.ib(default=None, converter=load_to_pandas)
    reanalysis: dict[str, pd.DataFrame] | None = attr.ib(
        default=None, converter=load_to_pandas_dict
    )
    preprocess: Callable | None = attr.ib(default=None)

    # Error catching in validation
    _errors: dict[str, list[str]] = attr.ib(
        default={"missing": {}, "dtype": {}, "frequency": []}, init=False
    )  # No user initialization required

    def __attrs_post_init__(self):
        self.reanalysis_validation()
        self._set_index_columns()
        self._validate_frequency()
        # Check the errors againts the analysis requirements
        error_message = compose_error_message(self._errors, analysis_types=self.analysis_type)
        if error_message != "":
            # raise ValueError("\n".join(itertools.chain(*self._errors.values())))
            raise ValueError(error_message)
        self.update_column_names()

        if self.preprocess is not None:
            self.preprocess(
                self
            )  # TODO: should be a user-defined method to run the data cleansing steps

    @scada.validator
    @meter.validator
    # @tower.validator
    @status.validator
    @curtail.validator
    @asset.validator
    def data_validator(self, instance: attr.Attribute, value: pd.DataFrame | None) -> None:
        """Validator function for each of the data buckets in `PlantData`.

        Args:
            instance (attr.Attribute): The `attr` attribute details
            value (pd.DataFrame | None): The attributes user-provided value.
        """
        if None in self.analysis_type:
            return
        name = instance.name
        if value is None:
            self._errors["missing"].update(
                {name: list(getattr(self.metadata, instance.name).col_map.values())}
            )
            self._errors["dtype"].update(
                {name: list(getattr(self.metadata, instance.name).dtypes.keys())}
            )

        else:
            self._errors["missing"].update(self._validate_column_names(category=name))
            self._errors["dtype"].update(self._validate_dtypes(category=name))

    def _set_index_columns(self) -> None:
        """Sets the index value for each of the `PlantData` objects that are not `None`."""
        if self.scada is not None:
            time_col = self.metadata.scada.col_map["time"]
            id_col = self.metadata.scada.col_map["id"]
            self.scada[time_col] = pd.DatetimeIndex(self.scada[time_col])
            self.scada = self.scada.set_index([time_col, id_col], drop=False)
            self.scada.index.names = ["time", "id"]

        if self.meter is not None:
            time_col = self.metadata.meter.col_map["time"]
            self.meter[time_col] = pd.DatetimeIndex(self.meter[time_col])
            self.meter = self.meter.set_index([time_col], drop=False)
            self.meter.index.name = "time"

        if self.status is not None:
            time_col = self.metadata.status.col_map["time"]
            id_col = self.metadata.status.col_map["id"]
            self.status[time_col] = pd.DatetimeIndex(self.status[time_col])
            self.status = self.status.set_index([time_col, id_col], drop=False)
            self.status.index.names = ["time", "id"]

        if self.curtail is not None:
            time_col = self.metadata.curtail.col_map["time"]
            self.curtail[time_col] = pd.DatetimeIndex(self.curtail[time_col])
            self.curtail = self.curtail.set_index([time_col], drop=False)
            self.curtail.index.name = "time"

        if self.asset is not None:
            id_col = self.metadata.asset.col_map["id"]
            self.asset = self.asset.set_index([id_col])
            self.asset.index.name = "id"

        if self.reanalysis is not None:
            for name in self.reanalysis:
                time_col = self.metadata.reanalysis["name"].col_map["time"]
                self.reanalysis["name"][time_col] = pd.DatetimeIndex(
                    self.reanalysis["name"][time_col]
                )
                self.reanalysis["name"] = self.reanalysis["name"].set_index([time_col], drop=False)
                self.reanalysis["name"].index.name = "time"

    def reanalysis_validation(self) -> None:
        """Provides the reanalysis data initialization and validation routine.

        Control Flow:
         - If `None` is provided, then run the `data_validator` method to collect
           missing columns and bad data types
         - If the dictionary values are a dictionary, then the reanalysis data will
           be downloaded using the dictionary as kwargs passed to the PlanetOS API
           in `openoa.toolkits.reanslysis_downloading`, with the product name and site
           coordinates being provided automatically. NOTE: This also calculates the
           derived variables such as wind direction upon downloading.
        - If a non-dictionary input is provided for a reanalysis product type, then the
          `load_to_pandas` method will be called on the input data.

        Raises:
            ValueError: Raised if reanalysis input is not a dictionary.
        """
        if None in self.analysis_type:
            return
        if self.reanalysis is None:
            self.data_validator(PlantData.reanalysis, self.reanalysis)
            return

        if not isinstance(self.reanalysis, dict):
            raise ValueError(
                "Reanalysis data should be provided as a dictionary of product name (keys) and api kwargs or data"
            )

        reanalysis = {}
        for name, value in self.reanalysis.items():
            if isinstance(value, dict):
                value.update(
                    dict(
                        dataset=name,
                        lat=self.metadata.latitude,
                        lon=self.metadata.longitude,
                        calc_derived_vars=True,
                    )
                )
                reanalysis[name] = download_reanalysis_data_planetos(**value)
            else:
                reanalysis[name] = load_to_pandas(value)

        self.reanalysis = reanalysis
        self._calculate_reanalysis_columns()

        # Capture the errors, but note that the frequency validation handles reanalysis
        # and doesn't need to be run separately
        self._errors["missing"].update(self._validate_column_names(category="reanalysis"))
        self._errors["dtype"].update(self._validate_dtypes(category="reanalysis"))

    @property
    def analysis_values(self):
        # if self.analysis_type == "x":
        #     return self.scada, self, self.meter, self.asset
        values = dict(
            scada=self.scada,
            meter=self.meter,
            # tower=self.tower,  # NOT IN USE CURRENTLY
            asset=self.asset,
            status=self.status,
            curtail=self.curtail,
            reanalysis=self.reanalysis,
        )
        return values

    def _validate_column_names(self, category: str = "all") -> dict[str, list[str]]:
        column_map = self.metadata.column_map

        if category == "reanalysis":
            missing_cols = {
                f"{category}-{name}": column_validator(df, column_names=column_map[category][name])
                for name, df in self.analysis_values[category].items()
            }
            return missing_cols if isinstance(missing_cols, dict) else {}

        if category != "all":
            df = self.analysis_values[category]
            missing_cols = {category: column_validator(df, column_names=column_map[category])}
            return missing_cols if isinstance(missing_cols, dict) else {}

        missing_cols = {
            name: column_validator(df, column_names=column_map[name])
            for name, df in self.analysis_values.items()
            if name != "reanalysis"
        }
        missing_cols.update(
            {
                f"reanalysis-{name}": column_validator(
                    df, column_names=column_map["reanalysis"][name]
                )
                for name, df, in self.analysis_values["reanalysis"].items()
            }
        )
        return missing_cols if isinstance(missing_cols, dict) else {}

    def _validate_dtypes(self, category: str = "all") -> dict[str, list[str]]:
        """Validates the dtype for each column for the specified `category`.

        Args:
            category (str, optional): The name of the data that should be checked, or "all" to
                validate all of the data types. Defaults to "all".

        Returns:
            dict[str, list[str]]: A dictionary of each data type and any columns that  don't
                match the required dtype and can't be converted to it successfully.
        """
        # Create a new mapping of the data's column names to the expected dtype
        # TODO: Consider if this should be a encoded in the metadata/plantdata object elsewhere
        column_name_map = self.metadata.column_map
        column_type_map = self.metadata.type_map
        column_map = {}
        for name in column_name_map:
            if name == "reanalysis":
                column_map["reanalysis"] = {}
                for name in column_name_map["reanalysis"]:
                    column_map["reanalysis"][name] = dict(
                        zip(
                            column_name_map["reanalysis"][name].values(),
                            column_type_map["reanalysis"][name].values(),
                        )
                    )
            else:
                column_map[name] = dict(
                    zip(column_name_map[name].values(), column_type_map[name].values())
                )

        error_cols = {}
        for name, df in self.analysis_values.items():
            if category != "all" and category != name:
                # Skip irrelevant data types if not checking all data types
                continue

            if name == "reanalysis":
                for sub_name, df in df.items():
                    error_cols[f"{category}-{name}"] = dtype_converter(
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
            category (str, optional): The data type category. Defaults to "all".

        Returns:
            list[str]: The list of data types that don't meet the required datetime frequency.
        """
        frequency_requirements = self.metadata.frequency_requirements(self.analysis_type)

        # Collect all the frequencies for each of the data types
        data_dict = self.analysis_values
        actual_frequencies = {}
        for name, df in data_dict.items():
            if df is None:
                continue

            if name in ("scada", "status"):
                freq = df.index.get_level_values("time").freq
                if freq is None:
                    freq = pd.infer_freq(df.index.get_level_values("time"))
                actual_frequencies[name] = freq
            elif name in ("meter", "curtail"):
                freq = df.index.freq
                if freq is None:
                    freq = pd.infer_freq(df.index)
                actual_frequencies[name] = freq
            elif name == "reanalysis":
                actual_frequencies["reanalysis"] = {}
                for sub_name, df in data_dict[name].items():
                    freq = df.index.freq
                    if freq is None:
                        freq = pd.infer_freq(df.index)
                    actual_frequencies["reanalysis"][sub_name] = freq

        invalid_freq = {}
        for name, freq in actual_frequencies.items():
            if category != "all" and category != name:
                # If only checking one data type, then skip all others
                continue
            if name == "reanalysis":
                for sub_name, freq in freq.items():
                    is_valid = frequency_validator(freq, frequency_requirements[name], True)
                    is_valid |= frequency_validator(freq, frequency_requirements[name], False)
                    if not is_valid:
                        invalid_freq.update({f"{name}-{sub_name}": freq})
                continue
            is_valid = frequency_validator(freq, frequency_requirements[name], True)
            is_valid |= frequency_validator(freq, frequency_requirements[name], False)
            if not is_valid:
                invalid_freq.update({name: freq})

        return invalid_freq

    def validate(self, metadata: Optional[dict | str | Path | PlantMetaData] = None) -> None:
        """Secondary method to validate the plant data objects after loading or changing
        data with option to provide an updated `metadata` object/file as well

        Args:
            metadata (Optional[dict]): Updated metadata object, dictionary, or file to
            create the updated metadata for data validation.

        Raises:
            ValueError: Raised at the end if errors are caught in the validation steps.
        """
        if metadata is not None:
            self.metadata = metadata

        self._errors = {
            "missing": self._validate_column_names(),
            "dtype": self._validate_dtypes(),
            "frequency": self._validate_frequency(),
        }
        self.reanalysis_validation()

        # TODO: Check for extra columns?
        # TODO: Define other checks?

        error_message = compose_error_message(self._errors, self.analysis_type)
        if error_message:
            raise ValueError(error_message)

        self.update_column_names()

    def _calculate_reanalysis_columns(self) -> None:
        """Calculates extra variables such as wind_direction from the provided
        reanalysis data if they don't already exist.
        """
        if self.reanalysis is None:
            return
        reanalysis = {}
        for name, df in self.reanalysis.items():
            col_map = self.metadata.reanalysis[name].col_map
            u = col_map["windspeed_u"]
            v = col_map["windspeed_v"]
            has_u_v = (u in df) & (v in df)

            ws = col_map["windspeed"]
            if ws not in df:
                if has_u_v:
                    df[ws] = np.sqrt(df[u].values ** 2 + df[v].values ** 2)

            wd = col_map["wind_direction"]
            if wd not in df:
                if has_u_v:
                    df[wd] = met.compute_wind_direction(df[u], df[v])

            dens = col_map["density"]
            sp = col_map["surface_pressure"]
            temp = col_map["temperature"]
            if dens not in df:
                if (sp in df) & (temp in df):
                    df[dens] = met.compute_air_density(df[temp], df[sp])

            reanalysis[name] = df
        self.reanalysis = reanalysis

    def update_column_names(self, to_original: bool = False) -> None:
        meta = self.metadata
        reverse = not to_original
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
            self.asset = rename_columns(self.asset, meta.asset.col_map)
        if self.reanalysis is not None:
            reanalysis = {}
            for name, df in self.reanalysis.items():
                reanalysis[name] = rename_columns(
                    df, meta.reanalysis[name].col_map, reverse=reverse
                )
            self.reanalysis = reanalysis

    # Not necessary, but could provide an additional way in
    @classmethod
    def from_entr(
        cls: PlantData,
        thrift_server_host: str = "localhost",
        thrift_server_port: int = 10000,
        database: str = "entr_warehouse",
        wind_plant: str = "",
        aggregation: str = "",
        date_range: list = None,
    ):
        """Load a PlantData object from data in an entr_warehouse.

        Args:
            thrift_server_url(str): URL of the Apache Thrift server
            database(str): Name of the Hive database
            wind_plant(str): Name of the wind plant you'd like to load
            aggregation: Not yet implemented
            date_range: Not yet implemented

        Returns:
            plant(PlantData): An OpenOA PlantData object.
        """
        return from_entr(
            thrift_server_host, thrift_server_port, database, wind_plant, aggregation, date_range
        )

    def turbine_ids(self) -> list[str]:
        """Convenience method for getting the unique turbine IDs from the scada data.

        Returns:
            list[str]: List of unique turbine identifiers.
        """
        return self.scada[self.metadata.scada.id].unique()


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
