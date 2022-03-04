from __future__ import annotations

import io
import os
import json
import itertools
from typing import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.sharedctypes import Value

import attr
import numpy as np
import pandas as pd
from attr import define
from dateutil.parser import parse

from openoa.types import timeseries_table

from .asset import AssetData
from .reanalysis import ReanalysisData


ENTR_SCHEMA = {"""the ENTR data column mapping."""}


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
            stauts=str,
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
            stauts=None,
            pitch="deg",
            temp="C",
        ),
        init=False,  # don't allow for user input
    )

    def __attrs_post_init__(self) -> None:
        self.col_map = dict()


@define(auto_attribs=True)
class MeterMetaData(FromDictMixin):
    col: str


@define(auto_attribs=True)
class TowerMetaData(FromDictMixin):
    col: str


@define(auto_attribs=True)
class StatusMetaData(FromDictMixin):
    col: str
    levels: int = 1


@define(auto_attribs=True)
class CurtailMetaData(FromDictMixin):
    col: str


@define(auto_attribs=True)
class AssetMetaData(FromDictMixin):
    col: str


@define(auto_attribs=True)
class ReanalysisMetaData(FromDictMixin):
    col: str


# def validate_longitude(instance, attribute, value):

#     if value > x or value < y:
#         raise ValueError("Bad longitude, must be between x and y")


@define(auto_attribs=True)
class PlantMetaData(FromDictMixin):
    """Composese the individual metadata/validation requirements from each of the
    individual data "types" that can compose a `PlantData` object.
    """

    scada: SCADAMetaData = attr.ib(converter=SCADAMetaData.from_dict)
    meter: MeterMetaData = attr.ib(converter=MeterMetaData.from_dict)
    tower: TowerMetaData = attr.ib(converter=TowerMetaData.from_dict)
    status: StatusMetaData = attr.ib(converter=StatusMetaData.from_dict)
    curtail: CurtailMetaData = attr.ib(converter=CurtailMetaData.from_dict)
    asset: AssetMetaData = attr.ib(converter=AssetMetaData.from_dict)
    reanalysis: ReanalysisMetaData = attr.ib(converter=ReanalysisMetaData.from_dict)


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
            "date_time_col": "Date_time",
        }
    },
    "scada": pd.DataFrame,
}


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
        converter=PlantMetaData.from_dict, on_setattr=[attr.converters, attr.validators]
    )
    scada: pd.DataFrame | None = attr.ib(on_setattr=[attr.validators])
    meter: pd.DataFrame = attr.ib(on_setattr=[attr.validators])
    tower: pd.DataFrame = attr.ib(on_setattr=[attr.validators])
    status: pd.DataFrame = attr.ib(on_setattr=[attr.validators])
    curtail: pd.DataFrame = attr.ib(on_setattr=[attr.validators])
    asset: pd.DataFrame = attr.ib(on_setattr=[attr.validators])
    reanalysis: pd.DataFrame = attr.ib(on_setattr=[attr.validators])
    analysis_type: list[str] = attr.ib(default=["all"], converter=convert_to_list)

    @scada.validator  # noqa: disable=F821
    def scada_column_validator(self, instance: attr.Attribute, value: pd.DataFrame | None):
        if value is None:
            # The data value isn't required for the analysis type
            return

        self.scada = self.scada.rename(columns=self.metadata.scada.col_map)
        missing_cols = [
            col for col in self.metadata.scada.col_map.values() if col not in value.columns
        ]
        if len(missing_cols) > 0:
            raise ValueError(
                f"Missing the following columns in the `scada` inputs: {missing_cols}."
            )

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


def validate(plant, schema):
    pass


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

    plant = PlantDataV2()

    plant.scada.df = pd.read_sql(scada_query, conn)

    conn.close()

    validate(plant)

    return plant


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
