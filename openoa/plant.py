from __future__ import annotations

import sys
import logging
import itertools
from typing import Callable, Optional, Sequence
from pathlib import Path

import yaml
import attrs
import numpy as np
import pandas as pd
from attrs import field, define
from pyproj import Transformer
from tabulate import tabulate
from IPython.display import Markdown, display
from shapely.geometry import Point

import openoa.utils.timeseries as ts
import openoa.utils.met_data_processing as met
from openoa.logging import set_log_level, setup_logging, logged_method_call
from openoa.schema.metadata import ANALYSIS_REQUIREMENTS, PlantMetaData
from openoa.utils.metadata_fetch import attach_eia_data
from openoa.utils.unit_conversion import convert_power_to_energy


setup_logging(level="WARNING")
logger = logging.getLogger(__name__)


# ****************************************
# Validators, Loading, and General methods
# ****************************************


@logged_method_call
def _analysis_filter(
    error_dict: dict, metadata: PlantMetaData, analysis_types: list[str] = ["all"]
) -> dict:
    """Filters the errors found by the analysis requirements  provided by the
    :py:attr:`analysis_types`.

    Args:
        error_dict (:obj:`dict`): The dictionary of errors separated by the keys:
            "missing", "dtype", and "frequency".
        metadata (:obj:`PlantMetaData`): The ``PlantMetaData`` object containing the column
            mappings for each data type.
        analysis_types (:obj:`list[str]`, optional): The list of analysis types to
            consider for validation. If "all" is contained in the list, then all errors
            are returned back, and if ``None`` is contained in the list, then no errors
            are returned, otherwise the union of analysis requirements is returned back.
            Defaults to ["all"].

    Returns:
        dict: The missing column, bad dtype, and incorrect timestamp frequency errors
            corresponding to the user's analysis types.
    """
    if "all" in analysis_types:
        return error_dict

    if analysis_types == [None]:
        return {}

    if None in analysis_types:
        _ = analysis_types.pop(analysis_types.index(None))

    categories = ("scada", "meter", "tower", "curtail", "reanalysis", "asset")
    requirements = {key: ANALYSIS_REQUIREMENTS[key] for key in analysis_types}
    column_requirements = {
        cat: set(
            itertools.chain(*[r.get(cat, {}).get("columns", []) for r in requirements.values()])
        )
        for cat in categories
    }
    for key, value in column_requirements.items():
        if key == "reanalysis":
            reanalysis_keys = [k for k in error_dict["missing"] if k.startswith(key)]
            _add = {}
            for k in reanalysis_keys:
                name = k.split("-")[1]
                col_map = getattr(metadata, key)[name].col_map
                _add[k] = {col_map[v] for v in value}
        else:
            col_map = getattr(metadata, key).col_map
            column_requirements.update({key: {col_map[v] for v in value}})
    column_requirements.update(_add)

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


@logged_method_call
def _compose_error_message(
    error_dict: dict, metadata: PlantMetaData, analysis_types: list[str] = ["all"]
) -> str:
    """Takes a dictionary of error messages from the ``PlantData`` validation routines,
    filters out errors unrelated to the intended analysis types, and creates a
    human-readable error message.

    Args:
        error_dict (dict): See ``PlantData._errors`` for more details.
        metadata (PlantMetaData): The ``PlantMetaData`` object containing the column
            mappings for each data type.
        analysis_types (list[str], optional): The user-input analysis types, which are
            used to filter out unlreated errors. Defaults to ["all"].

    Returns:
        str: The human-readable error message breakdown.
    """
    if analysis_types == [None]:
        return ""

    if "all" not in analysis_types:
        error_dict = _analysis_filter(error_dict, metadata, analysis_types)

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


@logged_method_call
def frequency_validator(
    actual_freq: str | int | float | None,
    desired_freq: str | set[str] | None,
    exact: bool,
) -> bool:
    """Helper function to check if the actual datetime stamp frequency is valid compared
    to what is required.

    Args:
        actual_freq(:obj:`str` | :obj:`int` | :obj:`float` | :obj:`None`): The frequency of the
            timestamp, either as an offset alias or manually determined in seconds between timestamps.
        desired_freq (Optional[str  |  None  |  set[str]]): Either the exact frequency,
            required or a set of options that are also valid, in which case any numeric
            information encoded in ``actual_freq`` will be dropped.
        exact(:obj:`bool`): If the provided frequency codes should be exact matches (``True``),
            or, if ``False``, the check should be for a combination of matches.

    Returns:
        (:obj:`bool`): If the actual datetime frequency is sufficient, per the match requirements.
    """
    if desired_freq is None:
        return True

    if actual_freq is None:
        return False

    if isinstance(desired_freq, str):
        desired_freq = {desired_freq}

    # If an offset alias couldn't be found, then convert the desired frequency strings to seconds
    # unless the frequency string is a monthly time encoding, which is deprecated.
    if not isinstance(actual_freq, str):
        desired_freq = {ts.offset_to_seconds(el) for el in desired_freq if el not in ("MS", "ME")}

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

    Args:
        value(:obj:`Sequence` | :obj:`str` | :obj:`int` | :obj:`float`): The unknown element to be
            converted to a list of element(s).
        manipulation(:obj:`Callable` | :obj:`None`) A function to be performed upon the individual elements, by default None.

    Returns:
        (:ojb:`list`): The new list of elements.
    """

    if isinstance(value, (str, int, float)) or value is None:
        value = [value]
    if manipulation is not None:
        return [manipulation(el) for el in value]
    return list(value)


@logged_method_call
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


@logged_method_call
def dtype_converter(df: pd.DataFrame, column_types={}) -> list[str]:
    """Converts the columns provided in :py:attr:`column_types` of :py:attr:`df` to the appropriate
    data type.

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


@logged_method_call
def load_to_pandas(data: str | Path | pd.DataFrame) -> pd.DataFrame | None:
    """Loads the input data or filepath to apandas DataFrame.

    Args:
        data (str | Path | pd.DataFrame): The input data.

    Raises:
        ValueError: Raised if an invalid data type was passed.

    Returns:
        pd.DataFrame | None: The passed ``None`` or the converted pandas DataFrame object.
    """
    if data is None:
        return data
    elif isinstance(data, (str, Path)):
        logger.info(f"Loading {data} to a pandas DataFrame")
        return pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise ValueError("Input data could not be converted to pandas")


def load_to_pandas_dict(
    data: dict[str | Path | pd.DataFrame],
) -> dict[str, pd.DataFrame] | None:
    """Converts a dictionary of data or data locations to a dictionary of ``pd.DataFrame``s
    by iterating over the dictionary and passing each value to ``load_to_pandas``.

    Args:
        data (dict[str  |  Path  |  pd.DataFrame]): The input data.

    Returns:
        dict[str, pd.DataFrame] | None: The passed ``None`` or the converted ``pd.DataFrame``
            object.
    """
    if data is None:
        return data
    for key, val in data.items():
        data[key] = load_to_pandas(val)
    return data


@logged_method_call
def rename_columns(df: pd.DataFrame, col_map: dict, reverse: bool = True) -> pd.DataFrame:
    """Renames the pandas DataFrame columns using col_map. Intended to be used in
    conjunction with the a data objects meta data column mapping (``reverse=True``).

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
            parameters. See ``PlantMetaData``, ``SCADAMetaData``, ``MeterMetaData``,
            ``TowerMetaData``, ``StatusMetaData``, ``CurtailMetaData``, ``AssetMetaData``,
            and/or ``ReanalysisMetaData`` for more information.
        analysis_type (`list[str]`): A single, or list of, analysis type(s) that
            will be run, that are configured in ``ANALYSIS_REQUIREMENTS``. See
            :py:attr:`openoa.schema.metadata.ANALYSIS_REQUIREMENTS` for requirements details.

            - None: Don't raise any errors for errors found in the data. This is intended
              for loading in messy data, but :py:meth:`validate` should be run later
              if planning on running any analyses.
            - "all": This is to check that all columns specified in the metadata schema
              align with the data provided, as well as data types and frequencies (where
              applicable).
            - "MonteCarloAEP": Checks the data components that are relevant to a Monte
              Carlo AEP analysis.
            - "MonteCarloAEP-temp": Checks the data components that are relevant to a
              Monte Carlo AEP analysis with ambient temperature data.
            - "MonteCarloAEP-wd": Checks the data components that are relevant to a
              Monte Carlo AEP analysis using an additional wind direction data point.
            - "MonteCarloAEP-temp-wd": Checks the data components that are relevant to a
              Monte Carlo AEP analysis with ambient temperature and wind direction data.
            - "TurbineLongTermGrossEnergy": Checks the data components that are relevant
              to a turbine long term gross energy analysis.
            - "ElectricalLosses": Checks the data components that are relevant to an
              electrical losses analysis.
            - "WakeLosses-scada": Checks the data components that are relevant to a
              wake losses analysis that uses the SCADA-based wind speed and direction
              data.
            - "WakeLosses-tower": Checks the data components that are relevant to a
              wake losses analysis that uses the met tower-based wind speed and
              direction data.

        scada (``pd.DataFrame``): Either the SCADA data that's been pre-loaded to a
            pandas `DataFrame`, or a path to the location of the data to be imported.
            See :py:class:`SCADAMetaData` for column data specifications.
        meter (``pd.DataFrame``): Either the meter data that's been pre-loaded to a
            pandas `DataFrame`, or a path to the location of the data to be imported.
            See :py:class:`MeterMetaData` for column data specifications.
        tower (``pd.DataFrame``): Either the met tower data that's been pre-loaded
            to a pandas `DataFrame`, or a path to the location of the data to be
            imported. See :py:class:`TowerMetaData` for column data specifications.
        status (``pd.DataFrame``): Either the status data that's been pre-loaded to
            a pandas `DataFrame`, or a path to the location of the data to be imported.
            See :py:class:`StatusMetaData` for column data specifications.
        curtail (``pd.DataFrame``): Either the curtailment data that's been
            pre-loaded to a pandas ``DataFrame``, or a path to the location of the data to
            be imported. See :py:class:`CurtailMetaData` for column data specifications.
        asset (``pd.DataFrame``): Either the asset summary data that's been
            pre-loaded to a pandas `DataFrame`, or a path to the location of the data to
            be imported. See :py:class:`AssetMetaData` for column data specifications.
        reanalysis (``dict[str, pd.DataFrame]``): Either the reanalysis data that's
            been pre-loaded to a dictionary of pandas ``DataFrame`` with keys indicating
            the data source, such as "era5" or "merra2", or a dictionary of paths to the
            location of the data to be imported following the same key naming convention.
            See :py:class:`ReanalysisMetaData` for column data specifications.

    Raises:
        ValueError: Raised if any analysis specific validation checks don't pass with an
            error message highlighting the appropriate issues.
    """

    log_level: str = field(default="WARNING", converter=set_log_level)
    metadata: PlantMetaData = field(
        default={},
        converter=PlantMetaData.load,
        on_setattr=[attrs.converters, attrs.validators],
        repr=False,
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

    # No user initialization required for attributes defined below here
    # Error catching in validation
    _errors: dict[str, list[str]] = field(
        default={"missing": {}, "dtype": {}, "frequency": {}, "attributes": []}, init=False
    )
    eia: dict = field(default={}, init=False)
    asset_distance_matrix: pd.DataFrame = field(init=False)
    asset_direction_matrix: pd.DataFrame = field(init=False)

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        self._calculate_reanalysis_columns()
        self._set_index_columns()
        self._validate_frequency()

        # Remove the non-product-specific reanalysis key if it exists
        # TODO: Find where this is actually entering the missing/dtype dictionaries
        [d.pop("reanalysis") for d in self._errors.values() if "reanalysis" in d]

        # Check the errors againts the analysis requirements
        error_message = _compose_error_message(
            self._errors, metadata=self.metadata, analysis_types=self.analysis_type
        )
        if error_message != "":
            raise ValueError(error_message)

        # Post-validation data manipulations
        self.calculate_asset_geometries()
        if self.asset is not None:
            self.parse_asset_geometry()
            self.calculate_asset_distance_matrix()
            self.calculate_asset_direction_matrix()

        if self.scada is not None:
            self.calculate_turbine_energy()

        # Change the column names to the -25 convention for easier use in the rest of the code base
        self.update_column_names()

    @scada.validator
    @meter.validator
    @tower.validator
    @status.validator
    @curtail.validator
    @asset.validator
    @logged_method_call
    def data_validator(self, instance: attrs.Attribute, value: pd.DataFrame | None) -> None:
        """Validator function for each of the data buckets in ``PlantData`` that checks
        that the appropriate columns exist for each dataframe, each column is of the
        right type, and that the timestamp frequencies are appropriate for the given
        ``analysis_type``.

        Args:
            instance (:obj:`attrs.Attribute`): The ``attrs.Attribute`` details
            value (:obj:`pd.DataFrame | None`): The attribute's user-provided value. A
                dictionary of dataframes is expected for reanalysis data only.
        """
        name = instance.name
        if self.analysis_type == [None]:
            logger.info(f"Skipping data validation for {name} because `analysis_type=None`.")
            return
        if value is None:
            columns = list(getattr(self.metadata, name).col_map.values())
            self._errors["missing"].update({name: columns})
            self._errors["dtype"].update({name: columns})

        else:
            self._errors["missing"].update(self._validate_column_names(category=name))
            self._errors["dtype"].update(self._validate_dtypes(category=name))

    @reanalysis.validator
    @logged_method_call
    def reanalysis_validator(
        self, instance: attrs.Attribute, value: dict[str, pd.DataFrame] | None
    ) -> None:
        """Validator function for the reanalysis data that checks for both matching reanalysis
        product keys in the ``PlantMetaData.reanalysis`` metadata definition, and the following:
        appropriate columns exist for each dataframe, each column is of the right type,
        and that the timestamp frequencies are appropriate for the given
        ``analysis_type``.

        Args:
            instance (:obj:`attrs.Attribute`): The :py:attr:`attrs.Attribute` details.
            value (:obj:`dict[str, pd.DataFrame]` | None): The attribute's user-provided value. A
                dictionary of dataframes is expected for reanalysis data only.
        """
        name = instance.name
        if value is not None:
            meta_products = [*self.metadata.reanalysis]
            data_products = [*value]
            if missing := set(data_products).difference(meta_products):
                raise KeyError(
                    f"Reanalysis meta data definitions were not provided for the following"
                    f" reanalysis data products: {missing}"
                )

        if self.analysis_type == [None]:
            logger.info(f"Skipping data validation for {name} because `analysis_type=None`.")
            return

        if value is None:
            for product, metadata in self.metadata.reanalysis.items():
                _name = f"{name}-{product}"
                columns = list(metadata.col_map.values())
                self._errors["missing"].update({_name: columns})
                self._errors["dtype"].update({_name: columns})

        else:
            self._errors["missing"].update(self._validate_column_names(category=name))
            self._errors["dtype"].update(self._validate_dtypes(category=name))

    def __generate_text_repr(self):
        """Generates a text summary of the core internal data."""
        repr = []
        for attribute in self.__attrs_attrs__:
            if not attribute.repr:
                continue

            name = attribute.name
            value = self.__getattribute__(name)
            if name == "analysis_type":
                repr.append(f"{name}: {value}")
            elif name in ("scada", "meter", "tower", "status", "curtail"):
                repr.append(f"\n{name}")
                repr.append("-" * len(name))
                if value is None:
                    repr.append("no data")
                else:
                    _repr = value.describe().T
                    repr.append(
                        tabulate(_repr, headers=_repr.columns, floatfmt=",.3f", tablefmt="grid")
                    )
            elif name == "reanalysis":
                repr.append(f"\n{name}")
                repr.append("-" * len(name))
                if "product" in value:
                    repr.append("no data")
                else:
                    for product, df in value.items():
                        repr.append(f"\n{product}")

                        _repr = df.describe().T
                        repr.append(
                            tabulate(_repr, headers=_repr.columns, floatfmt=",.3f", tablefmt="grid")
                        )
            elif name == "asset":
                repr.append(f"\n{name}")
                repr.append("-" * len(name))
                if value is None:
                    repr.append("no data")
                else:
                    value = value.drop(columns=["geometry"])
                    repr.append(
                        tabulate(value, headers=value.columns, floatfmt=",.3f", tablefmt="grid")
                    )
        return "\n".join(repr)

    def __generate_markdown_repr(self):
        """Generates a markdown-friendly summary of the core internal data."""
        new_line = "\n"

        repr = [
            "PlantData",
            new_line,
            "**analysis_type**",
            *[f"- {el}" for el in self.analysis_type],
            new_line,
        ]

        data = (
            "no data" if self.asset is None else self.asset.drop(columns=["geometry"]).to_markdown()
        )
        repr.extend(["**asset**", new_line, data, new_line])

        data = "no data" if self.scada is None else self.scada.describe().T.to_markdown()
        repr.extend(["**scada**", new_line, data, new_line])

        data = "no data" if self.meter is None else self.meter.describe().T.to_markdown()
        repr.extend(["**meter**", new_line, data, new_line])

        data = "no data" if self.tower is None else self.tower.describe().T.to_markdown()
        repr.extend(["**tower**", new_line, data, new_line])

        data = "no data" if self.status is None else self.status.describe().T.to_markdown()
        repr.extend(["**status**", new_line, data, new_line])

        data = "no data" if self.curtail is None else self.curtail.describe().T.to_markdown()
        repr.extend(["**curtail**", new_line, data, new_line])

        repr.extend(["**reanalysis**", new_line])

        if "product" in self.reanalysis:
            repr.append("no data")
        for name, df in self.reanalysis.items():
            data = df.describe().T.to_markdown()
            repr.extend([f"**{name}**", new_line, data, new_line])

        return (new_line).join(repr)

    def __str__(self):
        """The string summary."""
        return self.__generate_text_repr()

    def markdown(self):
        """A markdown-formatted version of the ``__str__``."""
        display(Markdown(self.__generate_markdown_repr()))

    def __repr__(self):
        """A context-aware summary generator for printing out the objects."""
        is_terminal = sys.stderr.isatty()
        if is_terminal:
            return self.__generate_text_repr()
        else:
            return repr(display(Markdown(self.__generate_markdown_repr())))

    @logged_method_call
    def _set_index_columns(self) -> None:
        """Sets the index value for each of the `PlantData` objects that are not `None`."""
        with attrs.validators.disabled():
            if self.scada is not None:
                time_col = self.metadata.scada.col_map["time"]
                id_col = self.metadata.scada.col_map["asset_id"]
                self.scada[time_col] = pd.DatetimeIndex(self.scada[time_col])
                self.scada = self.scada.set_index([time_col, id_col])
                self.scada.index.names = ["time", "asset_id"]

            if self.meter is not None:
                time_col = self.metadata.meter.col_map["time"]
                self.meter[time_col] = pd.DatetimeIndex(self.meter[time_col])
                self.meter = self.meter.set_index([time_col])
                self.meter.index.name = "time"

            if self.status is not None:
                time_col = self.metadata.status.col_map["time"]
                id_col = self.metadata.status.col_map["asset_id"]
                self.status[time_col] = pd.DatetimeIndex(self.status[time_col])
                self.status = self.status.set_index([time_col, id_col])
                self.status.index.names = ["time", "asset_id"]

            if self.tower is not None:
                time_col = self.metadata.tower.col_map["time"]
                id_col = self.metadata.tower.col_map["asset_id"]
                self.tower[time_col] = pd.DatetimeIndex(self.tower[time_col])
                self.tower = self.tower.set_index([time_col, id_col])
                self.tower.index.names = ["time", "asset_id"]

            if self.curtail is not None:
                time_col = self.metadata.curtail.col_map["time"]
                self.curtail[time_col] = pd.DatetimeIndex(self.curtail[time_col])
                self.curtail = self.curtail.set_index([time_col])
                self.curtail.index.name = "time"

            if self.asset is not None:
                id_col = self.metadata.asset.col_map["asset_id"]
                self.asset = self.asset.set_index([id_col])
                self.asset.index.name = "asset_id"

            if self.reanalysis is not None:
                for name in self.reanalysis:
                    time_col = self.metadata.reanalysis[name].col_map["time"]
                    self.reanalysis[name][time_col] = pd.DatetimeIndex(
                        self.reanalysis[name][time_col]
                    )
                    self.reanalysis[name] = self.reanalysis[name].set_index([time_col])
                    self.reanalysis[name].index.name = "time"

    @logged_method_call
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
        """Property that returns a dictionary of the data contained in the ``PlantData`` object.

        Returns:
            (:obj:`dict[str, pd.DataFrame]`): A mapping of the data type's name and the ``DataFrame``.
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

    @logged_method_call
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
            with_openoa_col_names (bool, optional): Use the PlantData column names (``True``), or
                convert the column names back to the originally provided values. Defaults to True.
            metadata (str, optional): File name (without extension) to be used for the metadata.
                Defaults to "metadata".
            scada (str, optional): File name (without extension) to be used for the SCADA data.
                Defaults to "scada".
            meter (str, optional): File name (without extension) to be used for the meter data.
                Defaults to "meter".
            tower (str, optional): File name (without extension) to be used for the tower data.
                Defaults to "tower".
            asset (str, optional): File name (without extension) to be used for the asset data.
                Defaults to "scada".
            status (str, optional): File name (without extension) to be used for the status data.
                Defaults to "status".
            curtail (str, optional): File name (without extension) to be used for the curtailment
                data. Defaults to "curtail".
            reanalysis (str, optional): Base file name (without extension) to be used for the
                reanalysis data, where each dataset will use the name provided to form the following
                file name: {save_path}/{reanalysis}_{name}. Defaults to "reanalysis".
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
            scada_fn = (save_path / scada).with_suffix(".csv")
            self.scada.reset_index(drop=False).to_csv(scada_fn, index=False)
            logger.info(f"SCADA data saved to: {scada_fn}")

        if self.status is not None:
            status_fn = (save_path / status).with_suffix(".csv")
            self.status.reset_index(drop=False).to_csv(status_fn, index=False)
            logger.info(f"Status data saved to: {status_fn}")

        if self.tower is not None:
            tower_fn = (save_path / tower).with_suffix(".csv")
            self.tower.reset_index(drop=False).to_csv(tower_fn, index=False)
            logger.info(f"Tower data saved to: {tower_fn}")

        if self.meter is not None:
            meter_fn = (save_path / meter).with_suffix(".csv")
            self.meter.reset_index(drop=False).to_csv(meter_fn, index=False)
            logger.info(f"Meter data saved to: {meter_fn}")

        if self.curtail is not None:
            curtail_fn = (save_path / curtail).with_suffix(".csv")
            self.curtail.reset_index(drop=False).to_csv(curtail_fn, index=False)
            logger.info(f"SCADA data saved to: {curtail_fn}")

        if self.asset is not None:
            asset_fn = (save_path / asset).with_suffix(".csv")
            self.asset.reset_index(drop=False).to_csv(asset_fn, index=False)
            logger.info(f"Asset data saved to: {asset_fn}")

        if self.reanalysis is not None:
            for name, df in self.reanalysis.items():
                reanalysis_fn = (save_path / f"{reanalysis}_{name}").with_suffix(".csv")
                df.reset_index(drop=False).to_csv(reanalysis_fn, index=False)
                logger.info(f"{name} reanalysis data saved to: {reanalysis_fn}")

    @logged_method_call
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
            if category != "all" and category != name:
                # Skip any irrelevant columns if not processing all data types
                continue
            if name == "reanalysis":
                # If no reanalysis data, get the default key from ReanalysisMetaData
                if df is None:
                    sub_name = [*column_map[name]][0]
                    missing_cols[f"{name}-{sub_name}"] = column_validator(
                        df, column_names=column_map[name][sub_name]
                    )
                    continue
                for sub_name, df in df.items():
                    logger.info(f"Validating column names in the {sub_name} {name} data")
                    missing_cols[f"{name}-{sub_name}"] = column_validator(
                        df, column_names=column_map[name][sub_name]
                    )
            else:
                logger.info(f"Validating column names in the {name} data")
                missing_cols[name] = column_validator(df, column_names=column_map[name])
        return missing_cols

    @logged_method_call
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
                column_map[name] = {}
                for sub_name in column_name_map[name]:
                    column_map[name][sub_name] = dict(
                        zip(
                            column_name_map[name][sub_name].values(),
                            column_dtype_map[name][sub_name].values(),
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
                if df is None:
                    # If no reanalysis data, get the default key from ReanalysisMetaData
                    sub_name = [*column_map[name]][0]
                    error_cols[f"{name}-{sub_name}"] = dtype_converter(
                        df, column_types=column_map[name][sub_name]
                    )
                    continue
                for sub_name, df in df.items():
                    logger.info(f"Validating the data types in the {sub_name} {name} data")
                    error_cols[f"{name}-{sub_name}"] = dtype_converter(
                        df, column_types=column_map[name][sub_name]
                    )
            else:
                logger.info(f"Validating the data types in the {name} data")
                error_cols[name] = dtype_converter(df, column_types=column_map[name])
        return error_cols

    @logged_method_call
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
                    logger.info(f"Validating the frequency of the {sub_name} {name} data")
                    is_valid = frequency_validator(freq, frequency_requirements.get(name), True)
                    is_valid |= frequency_validator(freq, frequency_requirements.get(name), False)
                    if not is_valid:
                        invalid_freq.update({f"{name}-{sub_name}": freq})
            else:
                logger.info(f"Validating the frequency of the {name} data")
                is_valid = frequency_validator(freq, frequency_requirements.get(name), True)
                is_valid |= frequency_validator(freq, frequency_requirements.get(name), False)
                if not is_valid:
                    invalid_freq.update({name: freq})

        return invalid_freq

    @logged_method_call
    def validate(self, metadata: dict | str | Path | PlantMetaData | None = None) -> None:
        """Secondary method to validate the plant data objects after loading or changing
        data with option to provide an updated `metadata` object/file as well

        Args:
            metadata (Optional[dict]): Updated metadata object, dictionary, or file to
                create the updated metadata for data validation, which should align with
                the mapped column names during initialization.

        Raises:
            ValueError: Raised at the end if errors are caught in the validation steps.
        """
        logger.info("Post-intialization data validation")
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

        error_message = _compose_error_message(self._errors, self.metadata, self.analysis_type)
        if error_message:
            raise ValueError(error_message)
        self.update_column_names()

    @logged_method_call
    def _calculate_reanalysis_columns(self) -> None:
        """Calculates extra variables such as wind direction from the provided
        reanalysis data if they don't already exist.
        """
        if self.reanalysis is None:
            return

        logger.info("Calculating extra variables for the reanalysis data")
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
                # .values to fix an issue where df[u] and df[v] with ANY NaN values
                # would cause df[wd] to be all NaN.
                df[wd] = met.compute_wind_direction(df[u], df[v]).values

            dens = col_map["WMETR_AirDen"]
            sp = col_map["WMETR_EnvPres"]
            temp = col_map["WMETR_EnvTmp"]
            has_sp_temp = (sp in df) & (temp in df)
            if dens not in df and has_sp_temp:
                df[dens] = met.compute_air_density(df[temp], df[sp])

            reanalysis[name] = df
        self.reanalysis = reanalysis

    @logged_method_call
    def parse_asset_geometry(
        self,
        reference_system: str | None = None,
        utm_zone: int | None = None,
        reference_longitude: float | None = None,
    ) -> None:
        """Calculate UTM coordinates from latitude/longitude.

        The UTM system divides the Earth into 60 zones, each 6deg of longitude in width. Zone 1
        covers longitude 180deg to 174deg W; zone numbering increases eastward to zone 60, which
        covers longitude 174deg E to 180deg. The polar regions south of 80deg S and north of 84deg N
        are excluded.

        Ref: http://geopandas.org/projections.html

        Args:
            reference_system (:obj:`str`, optional): Used to define the coordinate reference system
                (CRS). If None is used, then the `metadata.reference_system` value will be used.
                Defaults to the European Petroleum Survey Group (EPSG) code 4326 to be used with
                the World Geodetic System reference system, WGS 84.
            utm_zone (:obj:`int`, optional): UTM zone.  If None is used, then the
                `metadata.utm_zone` value will be used. Defaults to the being calculated from
                :py:attr:`reference_longitude`.
            reference_longitude (:obj:`float`, optional): Reference longitude for calculating the
                UTM zone. If None is used, then the `metadata.reference_longitude` value will be
                used. Defaults to the mean of `asset.longitude`.

        Returns: None
            Sets the asset "geometry" column.
        """
        # Check for metadata inputs
        if utm_zone is None:
            utm_zone = self.metadata.utm_zone
        if reference_longitude is None:
            reference_longitude = self.metadata.reference_longitude
        if reference_system is None:
            reference_system = self.metadata.reference_system

        # Calculate the UTM Zone as needed
        logger.info("Parsing the geometry of the asset coordinate data")
        if utm_zone is None:
            if reference_longitude is None:
                longitude = self.asset[self.metadata.asset.longitude].mean()
            utm_zone = int(np.floor((180 + longitude) / 6.0)) + 1

        to_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        transformer = Transformer.from_crs(reference_system.upper(), to_crs)
        lats, lons = transformer.transform(
            self.asset[self.metadata.asset.latitude].values,
            self.asset[self.metadata.asset.longitude].values,
        )

        self.asset["geometry"] = [Point(lat, lon) for lat, lon in zip(lats, lons)]

    @logged_method_call
    def update_column_names(self, to_original: bool = False) -> None:
        """Renames the columns of each dataframe to the be the keys from the
        `metadata.xx.col_map` that was passed during initialization.

        Args:
            to_original (bool, optional): An indicator to map the column names back to
                the originally passed values. Defaults to False.
        """
        meta = self.metadata
        reverse = not to_original  # flip the boolean to correctly map between the col_map entries

        if to_original:
            logger.info("Converting column names back to their original naming convention")
        else:
            logger.info("Converting column names to OpenOA conventions")

        with attrs.validators.disabled():
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

    @logged_method_call
    def calculate_turbine_energy(self) -> None:
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
            return self.scada.index.get_level_values("asset_id").unique()
        return self.asset.loc[self.asset["type"] == "turbine"].index.values

    @property
    def n_turbines(self) -> int:
        """The number of turbines contained in the data."""
        return self.turbine_ids.size

    def turbine_df(self, turbine_id: str) -> pd.DataFrame:
        """Filters `scada` on a single `turbine_id` and returns the filtered data frame.

        Args:
            turbine_id (str): The asset_id of the turbine to retrieve its data.

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
            return self.tower.index.get_level_values("asset_id").unique()
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

    @logged_method_call
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
        distance_array = distance.values
        np.fill_diagonal(distance_array, np.inf)
        distance.loc[:, :] = distance_array
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

    @logged_method_call
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
        direction_array = direction.values
        np.fill_diagonal(direction_array, np.inf)
        direction.loc[:, :] = direction_array
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

    def calculate_asset_geometries(self) -> None:
        """Calculates the asset distances and parses the asset geometries. This is intended for use
        during initialization and for when asset data is added after initialization
        """
        if self.asset is not None:
            self.parse_asset_geometry()
            self.calculate_asset_distance_matrix()
            self.calculate_asset_direction_matrix()

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

        return list(self.asset.loc[self.asset["type"] == "turbine"].index[freestream_indices])

    @logged_method_call
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

    def nearest_turbine(self, asset_id: str) -> str:
        """Finds the nearest turbine to the provided `asset_id`.

        Args:
            asset_id (str): A valid `asset` `asset_id`.

        Returns:
            str: The turbine `asset_id` closest to the provided `asset_id`.
        """
        if "nearest_turbine_id" not in self.asset.columns:
            self.calculate_nearest_neighbor()
        return self.asset.loc[asset_id, "nearest_turbine_id"].values[0]

    def nearest_tower(self, asset_id: str) -> str:
        """Finds the nearest tower to the provided `asset_id`.

        Args:
            asset_id (str): A valid `asset` `asset_id`.

        Returns:
            str: The tower `asset_id` closest to the provided `asset_id`.
        """
        if "nearest_tower_id" not in self.asset.columns:
            self.calculate_nearest_neighbor()
        return self.asset.loc[asset_id, "nearest_tower_id"].values[0]

    @classmethod
    def from_entr(cls, *args, **kwargs):
        try:
            from entr.plantdata import from_entr
        except ModuleNotFoundError:
            raise NotImplementedError(
                "The entr python package was not found. Please install py-entr by visiting https://github.com/entralliance/py-entr and following the instructions."
            )

        return from_entr(*args, **kwargs)


# **********************************************************
# Define additional class methods for custom loading methods
# **********************************************************

# Add the method for fetching and attaching the EIA plant data to the project
setattr(PlantData, "attach_eia_data", attach_eia_data)
