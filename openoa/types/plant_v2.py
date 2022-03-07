from __future__ import annotations

import io
import os
import json
import itertools
from typing import Callable, Sequence
from dataclasses import dataclass

import attr
import numpy as np
import pandas as pd
from attr import define, fields, fields_dict
from dateutil.parser import parse

from openoa.types import timeseries_table

from .asset import AssetData
from .reanalysis import ReanalysisData


# PlantData V2 with Attrs Dataclass
METADATA_DTYPE = "dtype"
METADATA_UNITS = "units"


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
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
        default=dict(
            time=np.datetime64,
            id=str,
            power=np.float64,
            windspeed=np.float64,
            wind_direction=np.float64,
            status=str,
            pitch=np.float64,
            temp=np.float64,
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
            temp="C",
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
    time: str = attr.ib(
        default="time", metadata={METADATA_DTYPE: np.datetime64, METADATA_UNITS: "datetime64[ns]"}
    )
    power: str = attr.ib(
        default="power", metadata={METADATA_DTYPE: np.float64, METADATA_UNITS: "kW"}
    )
    energy: str = attr.ib(
        default="energy", metadata={METADATA_DTYPE: np.float64, METADATA_UNITS: "kW"}
    )

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    col_map: dict = attr.ib(init=False)
    # dtypes: dict = attr.ib(
    #     default=dict(
    #         time=np.datetime64,
    #         power=np.float64,
    #         energy=np.float64,
    #     ),
    #     init=False,  # don't allow for user input
    # )
    # units: dict = attr.ib(
    #     default=dict(
    #         time="datetim64[ns]",
    #         power="kW",
    #         energy="kW",
    #     ),
    #     init=False,  # don't allow for user input
    # )

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
    curtailment_pct: str = attr.ib(default="curtailment_pct")
    availability_pct: str = attr.ib(default="availability_pct")
    net_energy: str = attr.ib(default="net_energy")

    # Data about the columns
    frequency: str = attr.ib(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
        default=dict(
            time=np.datetime64,
            curtailment_pct=np.float64,
            availability_pct=np.float64,
            net_energy=np.float64,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = attr.ib(
        default=dict(
            time="datetim64[ns]",
            curtailment_pct=None,
            availability_pct=None,
            net_energy="kW",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            curtailment_pct=self.curtailment_pct,
            availability_pct=self.availability_pct,
            net_energy=self.net_energy,
        )


@define(auto_attribs=True)
class AssetMetaData(FromDictMixin):
    # DataFrame columns
    id: str = attr.ib(default="id")
    latitude: str = attr.ib(default="latitude")
    longitude: str = attr.ib(default="longitude")
    rated_power: str = attr.ib(default="rated_power")
    type: str = attr.ib(default="type")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
        default=dict(
            id=str,
            latitude=np.float64,
            longitude=np.float64,
            rated_power=np.float64,
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
            type=self.type,
        )


@define(auto_attribs=True)
class ReanalysisMetaData(FromDictMixin):
    # DataFrame columns
    time: str = attr.ib(default="time")
    windspeed: str = attr.ib(default="windspeed")
    wind_direction: str = attr.ib(default="wind_direction")
    rho: str = attr.ib(default="rho")

    # Data about the columns
    frequency: str = attr.ib(default="10T")

    # Parameterizations that should not be changed
    # Prescribed mappings, datatypes, and units for in-code reference.
    col_map: dict = attr.ib(init=False)
    dtypes: dict = attr.ib(
        default=dict(
            time=np.datetime64,
            windspeed=np.float64,
            wind_direction=np.float64,
            rho=np.float64,
        ),
        init=False,  # don't allow for user input
    )
    units: dict = attr.ib(
        default=dict(
            time="datetim64[ns]",
            windspeed="m/s",
            wind_direction="deg",
            rho="kg/m^3",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict(
            time=self.time,
            windspeed=self.windspeed,
            wind_direction=self.wind_direction,
            rho=self.rho,
        )


# def validate_longitude(instance, attribute, value):

#     if value > x or value < y:
#         raise ValueError("Bad longitude, must be between x and y")


@define(auto_attribs=True)
class PlantMetaData(FromDictMixin):
    """Composese the individual metadata/validation requirements from each of the
    individual data "types" that can compose a `PlantData` object.
    """

    scada: SCADAMetaData = attr.ib(default={}, converter=SCADAMetaData.from_dict)
    meter: MeterMetaData = attr.ib(default={}, converter=MeterMetaData.from_dict)
    tower: TowerMetaData = attr.ib(default={}, converter=TowerMetaData.from_dict)
    status: StatusMetaData = attr.ib(default={}, converter=StatusMetaData.from_dict)
    curtail: CurtailMetaData = attr.ib(default={}, converter=CurtailMetaData.from_dict)
    asset: AssetMetaData = attr.ib(default={}, converter=AssetMetaData.from_dict)
    reanalysis: ReanalysisMetaData = attr.ib(default={}, converter=ReanalysisMetaData.from_dict)


def convert_to_list(
    value: Sequence | str | int | float,
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

    if isinstance(value, (str, int, float)):
        value = [value]
    if manipulation is not None:
        return [manipulation(el) for el in value]
    return list(value)


my_data = {
    "metadata": {
        "scada": {
            "time": "Date_time",
            "id": "Asset_ID",
            "freq": "10T",
        }
    },
    "scada": pd.DataFrame,
}


def column_validator(df: pd.DataFrame, column_names={}) -> None | list[str]:
    """Validates the column names"""
    missing = set(column_names.values()).difference(df.columns)
    if missing:
        return list(missing)
    return None


def _get_dtypes(cls, columns: list = []) -> dict:
    return {col: fields_dict(cls)[col].metadata[METADATA_DTYPE] for col in columns}


def dtype_converter(df: pd.DataFrame, column_types={}) -> None | list[str]:
    """Validates the column data types"""
    errors = []
    for column, new_type in column_types.items():
        try:
            df[column] = df[column].astype(new_type)
        except:  # noqa: disable=E722
            errors.append(column)

    if errors:
        return errors
    return None


@define(auto_attribs=True)
class PlantDataV3:
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
        default={}, converter=PlantMetaData.from_dict, on_setattr=[attr.converters, attr.validators]
    )
    scada: pd.DataFrame | None = attr.ib(default=None, on_setattr=[attr.validators])
    meter: pd.DataFrame | None = attr.ib(default=None, on_setattr=[attr.validators])
    tower: pd.DataFrame | None = attr.ib(default=None, on_setattr=[attr.validators])
    status: pd.DataFrame | None = attr.ib(default=None, on_setattr=[attr.validators])
    curtail: pd.DataFrame | None = attr.ib(default=None, on_setattr=[attr.validators])
    asset: pd.DataFrame | None = attr.ib(default=None, on_setattr=[attr.validators])
    reanalysis: pd.DataFrame | None = attr.ib(default=None, on_setattr=[attr.validators])
    analysis_type: list[str] | None = attr.ib(default=["all"], converter=convert_to_list)

    @scada.validator
    def scada_format_validator(self, instance: attr.Attribute, value: pd.DataFrame | None):
        if value is None:
            return None

        missing_cols = [
            col for col in self.metadata.scada.col_map.values() if col not in value.columns
        ]
        bad_dtypes = []
        for key, col in self.metadata.scada.col_map.items():
            try:
                dtype = self.metadata.scada.dtypes[key]
                value[col] = value[col].astype(dtype)
            except:  # noqa: disable=E722
                bad_dtypes.append(col)

        errors = []
        if len(missing_cols) > 0:
            errors.append(
                ValueError(f"Missing the following columns in the `scada` inputs: {missing_cols}.")
            )
        if len(bad_dtypes) > 0:
            errors.append(
                ValueError(
                    f"The following columns in `scada` were unable to converted to the correct data dtypes: {bad_dtypes}"
                )
            )

    @meter.validator  # noqa: disable=F821
    def meter_format_validator(self, instance: attr.Attribute, value: pd.DataFrame | None):
        if value is None:
            return None
        missing_cols = column_validator(value, column_names=self.metadata.meter.col_map)

        dtypes = _get_dtypes(MeterMetaData, self.metadata.meter.col_map)
        data_error_cols = dtype_converter(value, column_types=dtypes)

        errors = []
        if missing_cols is not None:
            errors.append(
                ValueError(f"Missing the following columns in the `meter` inputs: {missing_cols}.")
            )

        if data_error_cols is not None:
            errors.append(
                ValueError(
                    "The following `meter` columns could not be converted properly, ",
                    f"please check the data: {data_error_cols}",
                )
            )
        if errors:
            raise Exception(errors)

    # Not necessary, but could provide an additional way in
    @classmethod
    def from_entr(
        cls: PlantDataV3,
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

    plant = PlantDataV3()

    plant.scada.df = pd.read_sql(scada_query, conn)

    conn.close()

    return plant


# PlantData V2 with Python Dataclass
# requirements:
# - Holds 7 dataframes with data about one wind plant
# - Optionally validates data with respect to a schema
# - Can support loading data from multiple sources and saving itself to disk
@dataclass
class PlantDataV2:
    scada: pd.DataFrame
    meter: pd.DataFrame
    tower: pd.DataFrame
    status: pd.DataFrame
    curtail: pd.DataFrame
    asset: pd.DataFrame
    reanalysis: pd.DataFrame

    name: str
    version: float = 2

    def __init__(self):
        self._dataframe_field_names = [
            "scada",
            "meter",
            "tower",
            "status",
            "curtail",
            "asset",
            "reanalysis",
        ]

    def _get_dataframes(self) -> dict[str : pd.DataFrame]:
        return {name: getattr(self, name) for name in self._dataframe_fields}

    def validate(self, schema, fail_if_contains_extra_data=False):
        """Validate this plant data object against a schema. Returns True if valid, Rasies an exception if not valid.

        Example Usage:
        ```
        # Plant is automatically validated when an analysis is run
        openoa.AEP(plant).run()

        # Manually validate with a schema
        schema = openoa.AEP.input_schema # schema is a python dict object
        plant.validate(schema)
        ```
        """
        errors = []

        dataframes = self._get_dataframes()
        for field in schema["fields"]:
            field_df = dataframes[field["name"]]

            # Check the dataframe contains the right columns:
            expected_tags = set([field.name for field in field["fields"]])
            present_tags = set(field_df.columns)

            # Missing tags
            missing_tags = expected_tags - present_tags
            if len(missing_tags) > 0:
                errors.append(f"Table {field['name']} missing tags {missing_tags}")

            # Extra tags
            if fail_if_contains_extra_data:
                extra_tags = present_tags - expected_tags
                if len(extra_tags > 0):
                    errors.append(f"Table {field['name']} contains extra tags {extra_tags}")

            # Special validator for scada
            if field["name"] == "scada":
                pass

        if len(errors > 0):
            for error in errors:
                print(error)
            raise ValueError(f"Plant {self.name} failed validation")
        else:
            return True

    def __repr__(self):
        print(f"PlantData V{self.version}")
        print(f"\tPlant: {self.name}")
        print("=======================================")
        missing_tables = ["scada", "meter", "tower", "status", "curtail", "asset", "reanalysis"]

        if self.asset is not None and self.asset.shape[0] > 0:
            missing_tables.remove("asset")
            print("\tAsset Table:")
            print(f"\t\tNumber of Assets: {self.asset.shape[0]}")

        if self.scada is not None and self.scada.shape[0] > 0:
            missing_tables.remove("scada")
            print("\tScada Table:")
            print(f"\t\tNumber of Rows: {self.scada.shape[0]}")
            print(f"\t\tNumber of Columns: {self.scada.shape[1]}")
            print(f"\t\tTags: {self.scada.columns}")

        print(f"Missing or Empty Tables: {missing_tables}")

    def save(self, path):
        pass

    @classmethod
    def from_save(cls, path):
        pass

    @classmethod
    def from_entr(
        cls,
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

        plant = cls()

        plant.scada.df = pd.read_sql(scada_query, conn)

        conn.close()

        return plant

    @classmethod
    def from_pandas(cls, scada, meter, status, tower, asset, curtail, reanalysis):
        """
        from_pandas

        Create a PlantData object from a collection of Pandas data frames.

        Args:
            scada:
            meter:
            status:
            tower:
            asset:
            curtail:
            reanalysis:

        Returns:
            plant(PlantData): An OpenOA PlantData object.
        """
        plant = cls()

        plant.scada = scada
        plant.meter = meter
        plant.status = status
        plant.tower = tower
        plant.asset = asset
        plant.curtail = curtail
        plant.reanalysis = reanalysis

        plant.validate()


def from_plantdata_v1(plant_v1: PlantData):
    plant_v2 = PlantDataV2()
    plant_v2.scada = plant_v1.scada._df
    plant_v2.asset = plant_v1.asset._df
    plant_v2.meter = plant_v1.meter._df
    plant_v2.tower = plant_v1.tower._df
    plant_v2.status = plant_v1.status._df
    plant_v2.curtail = plant_v1.curtail._df
    plant_v2.reanalysis = plant_v1.reanalysis._df

    # copy any other data members to their new location

    # validate(plant_v2)

    return plant_v2


# PlantData
class PlantData(object):
    """Data object for operational wind plant data.

    This class holds references to all tables associated with a wind plant. The tables are grouped by type:
        - PlantData.scada
        - PlantData.meter
        - PlantData.tower
        - PlantData.status
        - PlantData.curtail
        - PlantData.asset
        - PlantData.reanalysis

    Each table must have columns following the following convention:
        -

    The PlantData object can serialize all of these structures and reload them
    them from the cache as needed.

    The underlying datastructure is a TimeseriesTable, which is agnostic to the underlying
    engine and can be implemented with Pandas, Spark, or Dask (for instance).

    Individual plants will extend this object with their own
    prepare() and other methods.
    """

    def __init__(self, path, name, engine="pandas", toolkit=["pruf_analysis"], schema=None):
        """
        Create a plant data object without loading any data.

        Args:
            path(string): path where data should be read/written
            name(string): uniqiue name for this plant in case there's multiple plant's data in the directory
            engine(string): backend engine - pandas, spark or dask
            toolkit(list): the _tool_classes attribute defines a list of toolkit modules that can be loaded

        Returns:
            New object
        """
        if not schema:
            dir = os.path.dirname(os.path.abspath(__file__))
            schema = dir + "/plant_schema.json"
        with open(schema) as schema_file:
            self._schema = json.load(schema_file)

        self._scada = timeseries_table.TimeseriesTable.factory(engine)
        self._meter = timeseries_table.TimeseriesTable.factory(engine)
        self._tower = timeseries_table.TimeseriesTable.factory(engine)
        self._status = timeseries_table.TimeseriesTable.factory(engine)
        self._curtail = timeseries_table.TimeseriesTable.factory(engine)
        self._asset = AssetData(engine)
        self._reanalysis = ReanalysisData(engine)
        self._name = name
        self._path = path
        self._engine = engine

        self._version = 1

        self._status_labels = ["full", "unavailable"]

        self._tables = [
            "_scada",
            "_meter",
            "_status",
            "_tower",
            "_asset",
            "_curtail",
            "_reanalysis",
        ]

    def amend_std(self, dfname, new_fields):
        """
        Amend a dataframe standard with new or changed fields. Consider running ensure_columns afterward to
        automatically create the new required columns if they don't exist.

        Args:
            dfname (string): one of scada, status, curtail, etc.
            new_fields (dict): set of new fields and types in the same format as _scada_std to be added/changed in
            the std

        Returns:
            New data field standard
        """

        k = "_%s_std" % (dfname,)
        setattr(
            self, k, dict(itertools.chain(iter(getattr(self, k).items()), iter(new_fields.items())))
        )

    def get_time_range(self):
        """Get time range as tuple

        Returns:
            (tuple):
                start_time(datetime): start time
                stop_time(datetime): stop time
        """
        return (self._start_time, self._stop_time)

    def set_time_range(self, start_time, stop_time):
        """Set time range given two unparsed timestamp strings

        Args:
            start_time(string): start time
            stop_time(string): stop time

        Returns:
            (None)
        """
        self._start_time = parse(start_time)
        self._stop_time = parse(stop_time)

    def save(self, path=None):
        """Save out the project and all JSON serializeable attributes to a file path.

        Args:
            path(string): Location of new directory into which plant will be saved. The directory should not
            already exist. Defaults to self._path

        Returns:
            (None)
        """
        if path is None:
            raise RuntimeError("Path not specified.")

        os.mkdir(path)

        meta_dict = {}
        for ca, ci in self.__dict__.items():
            if ca in self._tables:
                ci.save(path, ca)
            elif ca in ["_start_time", "_stop_time"]:
                meta_dict[ca] = str(ci)
            else:
                meta_dict[ca] = ci

        with io.open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as outfile:
            outfile.write(str(json.dumps(meta_dict, ensure_ascii=False)))

    def load(self, path=None):
        """Load this project and all associated data from a file path

        Args:
            path(string): Location of plant data directory. Defaults to self._path

        Returns:
            (None)
        """
        if not path:
            path = self._path

        for df in self._tables:
            getattr(self, df).load(path, df)

        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            with io.open(os.path.join(path, "metadata.json"), "r") as infile:
                meta_dict = json.load(infile)
                for ca, ci in meta_dict.items():
                    if ca in ["_start_time", "_stop_time"]:
                        ci = parse(ci)
                    setattr(self, ca, ci)

    def ensure_columns(self):
        """@deprecated Ensure all dataframes contain necessary columns and format as needed"""
        raise NotImplementedError("ensure_columns has been deprecated. Use plant.validate instead.")

    def validate(self, schema=None):

        """Validate this plant data object against its schema. Returns True if valid, Rasies an exception if not valid."""

        if not schema:
            schema = self._schema

        for field in schema["fields"]:
            if field["type"] == "timeseries":
                attr = "_{}".format(field["name"])
                if not getattr(self, attr).is_empty():
                    getattr(self, attr).validate(field)

        return True

    def merge_asset_metadata(self):
        """Merge metadata from the asset table into the scada and tower tables"""
        if not (self._scada.is_empty()) and (len(self._asset.turbine_ids()) > 0):
            self._scada.pandas_merge(
                self._asset.df,
                [
                    "latitude",
                    "longitude",
                    "rated_power_kw",
                    "id",
                    "nearest_turbine_id",
                    "nearest_tower_id",
                ],
                "left",
                on="id",
            )
        if not (self._tower.is_empty()) and (len(self._asset.tower_ids()) > 0):
            self._tower.pandas_merge(
                self._asset.df,
                [
                    "latitude",
                    "longitude",
                    "rated_power_kw",
                    "id",
                    "nearest_turbine_id",
                    "nearest_tower_id",
                ],
                "left",
                on="id",
            )

    def prepare(self):
        """Prepare this object for use by loading data and doing essential preprocessing."""
        self.ensure_columns()
        if not ((self._scada.is_empty()) or (self._tower.is_empty())):
            self._asset.prepare(self._scada.unique("id"), self._tower.unique("id"))
        self.merge_asset_metadata()

    @property
    def scada(self):
        return self._scada

    @property
    def meter(self):
        return self._meter

    @property
    def tower(self):
        return self._tower

    @property
    def reanalysis(self):
        return self._reanalysis

    @property
    def status(self):
        return self._status

    @property
    def asset(self):
        return self._asset

    @property
    def curtail(self):
        return self._curtail

    @classmethod
    def from_entr(
        cls,
        thrift_server_host="localhost",
        thrift_server_port=10000,
        database="entr_warehouse",
        wind_plant="",
        aggregation="",
        date_range=None,
    ):
        """
        from_entr

        Load a PlantData object from data in an entr_warehouse.

        Args:
            thrift_server_host(str): URL of the Apache Thrift server
            thrift_server_port(int): Port of the Apache Thrift server
            database(str): Name of the Hive database
            wind_plant(str): Name of the wind plant you'd like to load
            aggregation: Not yet implemented
            date_range: Not yet implemented

        Returns:
            plant(PlantData): An OpenOA PlantData object.
        """
        from pyhive import hive

        plant = cls(
            database, wind_plant
        )  # Passing in database as the path and wind_plant as the name for now.

        conn = hive.Connection(host=thrift_server_host, port=thrift_server_port)

        scada_query = f"""SELECT Wind_turbine_name as Wind_turbine_name,
                Date_time as Date_time,
                cast(P_avg as float) as P_avg,
                cast(Power_W as float) as Power_W,
                cast(Ws_avg as float) as Ws_avg,
                Wa_avg as Wa_avg,
                Va_avg as Va_avg,
                Ya_avg as Ya_avg,
                Ot_avg as Ot_avg,
                Ba_avg as Ba_avg

        FROM {database}.{wind_plant}
        """

        plant.scada.df = pd.read_sql(scada_query, conn)

        conn.close()

        return plant

    @classmethod
    def from_pandas(cls, scada, meter, status, tower, asset, curtail, reanalysis):
        """
        from_pandas

        Create a PlantData object from a collection of Pandas data frames.

        Args:
            scada:
            meter:
            status:
            tower:
            asset:
            curtail:
            reanalysis:

        Returns:
            plant(PlantData): An OpenOA PlantData object.
        """
        plant = cls()

        plant.scada.df = scada
        plant.meter.df = meter
        plant.status.df = status
        plant.tower.df = tower
        plant.asset.df = asset
        plant.curtail.df = curtail
        plant.reanalysis.df = reanalysis

        plant.validate()
