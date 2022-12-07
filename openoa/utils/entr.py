"""
ENTR OpenOA Toolkit for OpenOA V3 API

Provides low-level utility functions to load data from ENTR warehouse into PlantData objects using the PyHive (for remote warehouses) or PySpark (for local spark clusters and UDFs) libraries.

With an existing OpenOA PlantData object, you can use the functions from this file as follows:

```
conn = PyHiveEntrConnection(thrift_server_host, thrift_server_port)
conn = PySparkEntrConnection()

plant_metadata:dict = entr.load_metadata(conn, plant_name)
asset:pd.DataFrame, asset_metadata:dict = entr.load_asset(conn, plant_metadata, cols=...)
scada:pd.DataFrame, scada_metadata:dict = entr.load_scada(conn, plant_metadata, cols=...)
... curtailment, meter, reanalysis...

plant_metadata["scada"] = scada_metadata
plant_metadata["asset] = asset_metadata

plant = PlantData(plant_metadata, scada, asset...)

```

This usage pattern is implemented in PlantData.from_entr()

Most functions in this file follow a pattern:

    load_TABLE_meta:
        Function that loads data from the metadata tables associated with a given table.

    load_TABLE_prepare:
        Function that takes raw data from the ENTR warehouse table, and does the necessary transformations so they are ready to load into OpenOA.

    load_TABLE:
        Function that loads all data from a given table.

In some cases, check_metadata_row is called to assert properties about a table's metadata.
"""

import pandas as pd
import openoa.utils.unit_conversion as un
import openoa.utils.met_data_processing as met
from dataclasses import dataclass

## --- DATABASE CONNECTIONS ---

class EntrConnection:
    _conn: object

    def pandas_query(self, query_string:str) -> pd.DataFrame:
        pass

class PySparkEntrConnection(EntrConnection):

    def __init__(self):
        """
        Get PySpark-Based Connection object for ENTR Warehouse.
        """
        from pyspark.sql import SparkSession
        self._conn = SparkSession.builder\
            .appName("entr_openoa_connector")\
            .config("spark.sql.warehouse.dir", "/home/jovyan/warehouse")\
            .config("spark.hadoop.javax.jdo.option.ConnectionURL", "jdbc:derby:;databaseName=/home/jovyan/warehouse/metastore_db;create=true")\
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
            .enableHiveSupport()\
            .getOrCreate()
        self._conn.sql("use entr_warehouse")

    def pandas_query(self, query_string:str) -> pd.DataFrame:
        """
        Query the PySpark-Based ENTR Warehouse, returning a pandas dataframe.

        Args:
            query_string(:obj:`str`): Spark SQL Query
        
        Returns:
            :obj:`pandas.DataFrame`: Result of the query.
        """
        return self._conn.sql(query_string).toPandas()

class PyHiveEntrConnection(EntrConnection):

    def __init__(self,thrift_server_host:str="localhost",thrift_server_port:int=10000):
        """
        Get PyHive-Based Connection object for ENTR Warehouse. This connection object at self._conn is DBAPI2 compatible.

        Args:
            thrift_server_host(:obj:`str`): URL of Apache Thrift2 server
            thrift_server_port(:obj:`int`): Port of Apache Thrift2 server
        """
        from pyhive import hive
        self._conn = hive.Connection(host=thrift_server_host, port=thrift_server_port)

    def pandas_query(self, query_string:str) -> pd.DataFrame:
        """
        Query the PyHive-Based ENTR Warehouse, returning a pandas dataframe.

        Args:
            query_string(:obj:`str`): SQL Query
        
        Returns:
            :obj:`pandas.DataFrame`: Result of the query.
        """
        return pd.read_sql(query_string, self._conn)

## --- PLANT LEVEL METADATA ---

def load_metadata(conn:EntrConnection, plant_name:str) -> dict:
    ## Plant Metadata
    metadata_query = f"""
    SELECT
        plant_id,
        plant_name,
        latitude,
        longitude,
        plant_capacity,
        number_of_turbines,
        turbine_capacity
    FROM
        entr_warehouse.dim_asset_wind_plant
    WHERE
        plant_name = "{plant_name}";
    """
    metadata = conn.pandas_query(metadata_query)

    assert len(metadata)<2, f"Multiple plants matching name {plant_name}"
    assert len(metadata)>0, f"No plant matching name {plant_name}"

    metadata_dict = {
        "latitude": metadata["latitude"][0],
        "longitude": metadata["longitude"][0],
        "capacity": metadata["plant_capacity"][0],
        "number_of_turbines": metadata["number_of_turbines"][0],
        "turbine_capacity": metadata["turbine_capacity"][0],
        "_entr_plant_id": metadata["plant_id"][0]
    }
    return metadata_dict

## --- ASSET ---

def load_asset(conn:EntrConnection, plant_metadata:dict):
    asset_query = f"""
    SELECT
        plant_id,
        wind_turbine_id,
        wind_turbine_name,
        latitude,
        longitude,
        elevation,
        hub_height,
        rotor_diameter,
        rated_power,
        manufacturer,
        model
    FROM
        entr_warehouse.dim_asset_wind_turbine
    WHERE
        plant_id = {plant_metadata['_entr_plant_id']};
    """
    return conn.pandas_query(asset_query)

## --- SCADA ---

def load_scada_meta(conn:EntrConnection, plant_metadata:dict):
    # Query the warehouse for any non-uniform metadata
    query = f"""
    SELECT
        interval_s,
        value_type,
        value_units
    FROM
        entr_warehouse.openoa_wtg_scada_tag_metadata
    WHERE
        entr_tag_name = 'WTUR.W';
    """
    scada_meta_df = conn.pandas_query(query)

    freq, _, _ = check_metadata_row(scada_meta_df.iloc[0], allowed_freq=['10T'], allowed_types=["average"], allowed_units=["W","Wh"])

    # Build the metadata dictionary
    scada_metadata = {
        "frequency": freq
    }

    return scada_metadata

def load_scada(conn, plant):

    load_scada_meta(conn, plant)
    
    scada_query = f"""
    SELECT
        entr_warehouse.openoa_wtg_scada.wind_turbine_name,
        date_time,
        `WROT.BlPthAngVal`,
        `WTUR.W`,
        `WMET.HorWdSpd`,
        `WMET.HorWdDirRel`,
        `WMET.EnvTmp`,
        `WNAC.Dir`,
        `WMET.HorWdDir`,
        `WTUR.SupWh`
    FROM
        entr_warehouse.openoa_wtg_scada
    WHERE
        plant_id = {plant._entr_plant_id};
    """
    plant.scada.df = pd.read_sql(scada_query, conn)

    load_scada_prepare(plant)

def load_scada_prepare(plant):
    
    plant._scada.df['time'] = pd.to_datetime(plant._scada.df['date_time'],utc=True).dt.tz_localize(None)

    # # Remove duplicated timestamps and turbine id
    plant._scada.df = plant._scada.df.drop_duplicates(subset=['time','wind_turbine_name'],keep='first')

    # # Set time as index
    plant._scada.df.set_index('time',inplace=True,drop=False)

    plant._scada.df = plant._scada.df[(plant._scada.df["WMET.EnvTmp"]>=-15.0) & (plant._scada.df["WMET.EnvTmp"]<=45.0)]

    # # Convert pitch to range -180 to 180.
    plant._scada.df["WROT.BlPthAngVal"] = plant._scada.df["WROT.BlPthAngVal"] % 360
    plant._scada.df.loc[plant._scada.df["WROT.BlPthAngVal"] > 180.0,"WROT.BlPthAngVal"] \
        = plant._scada.df.loc[plant._scada.df["WROT.BlPthAngVal"] > 180.0,"WROT.BlPthAngVal"] - 360.0

    # # Calculate energy
    plant._scada.df['energy_kwh'] = plant._scada.df['WTUR.SupWh'] / 1000.0

    # # Note: there is no vane direction variable defined in -25, so
    # # making one up
    scada_map = {
                "date_time"                 : "time",
                "wind_turbine_name"    : "id",
                "WTUR.W"              : "wtur_W_avg",

                "WMET.HorWdSpd"          : "wmet_wdspd_avg",
                "WMET.HorWdDirRel"       : "wmet_HorWdDir_avg",
                "WMET.HorWdDir"          : "wmet_VaneDir_avg",
                "WNAC.Dir"               : "wyaw_YwAng_avg",
                "WMET.EnvTmp"            : "wmet_EnvTmp_avg",
                "WROT.BlPthAngVal"       : "wrot_BlPthAngVal1_avg",
                }

    plant._scada.df.rename(scada_map, axis="columns", inplace=True)

def check_metadata_row(row, allowed_freq=["10T"], allowed_types=["sum"], allowed_units=["kWh"]):
    """
    Check the result of an ENTR Warehouse metadata query to make sure the values conform to our expectation.
    """
    accepted_freq = None
    freq_long_str = f"{row['interval_s']} sec"
    freq_timedelta = pd.Timedelta(freq_long_str)
    for freq in allowed_freq:
        if freq_timedelta == pd.Timedelta(freq): 
            accepted_freq = freq
            break
    assert accepted_freq is not None, f"Unsupported time frequency {freq_long_str} does not match any allowed frequencies {allowed_freq}"

    assert row["value_type"] in allowed_types, f"Unsupported value type {row['value_type']}"
    assert row["value_units"] in allowed_units, f"Unsupported value type {row['value_units']}"

    return accepted_freq, row["value_type"], row["value_units"]

## --- CURTAILMENT ---

def load_curtailment_meta(conn, plant):
    query = f"""
    SELECT
        interval_s,
        value_type,
        value_units
    FROM
        entr_warehouse.openoa_curtailment_and_availability_tag_metadata
    WHERE
        entr_tag_name in ('IAVL.DnWh', 'IAVL.ExtPwrDnWh')
    """
    meter_meta_df = pd.read_sql(query, conn)
    freq, _, _ = check_metadata_row(meter_meta_df.iloc[0], allowed_freq=['10T'], allowed_types=["sum"], allowed_units=["kWh"])
    plant._curtail_freq = freq

def load_curtailment(conn, plant):

    load_curtailment_meta(conn, plant)

    query = f"""
    SELECT
        date_time,
        `IAVL.DnWh`,
        `IAVL.ExtPwrDnWh`
    FROM
        entr_warehouse.openoa_curtailment_and_availability
    WHERE
        plant_id = {plant._entr_plant_id}
    ORDER BY
        date_time;
    """
    plant.curtail.df = pd.read_sql(query, conn)

    load_curtailment_prepare(plant)

def load_curtailment_prepare(plant):

    curtail_map = {
        'IAVL.DnWh':'availability_kwh',
        'IAVL.ExtPwrDnWh':'curtailment_kwh',
        'date_time':'time'
    }

    plant._curtail.df.rename(curtail_map, axis="columns", inplace=True)

    # Create datetime field
    plant._curtail.df['time'] = pd.to_datetime(plant._curtail.df.time).dt.tz_localize(None)
    plant._curtail.df.set_index('time',inplace=True,drop=False)


## --- METER ---

def load_meter_meta(conn, plant):
    query = f"""
    SELECT
        interval_s,
        value_type,
        value_units
    FROM
        entr_warehouse.openoa_revenue_meter_tag_metadata
    WHERE
        entr_tag_name = 'MMTR.SupWh'
    """
    meter_meta_df = pd.read_sql(query, conn)

    # Parse frequency
    freq, _, _ = check_metadata_row(meter_meta_df.iloc[0], allowed_freq=['10T'], allowed_types=["sum"], allowed_units=["kWh"])
    plant._meter_freq = freq

def load_meter(conn, plant):

    load_meter_meta(conn, plant)

    meter_query = f"""
    SELECT
        date_time,
        `MMTR.SupWh`
    FROM
        entr_warehouse.openoa_revenue_meter
    WHERE
        plant_id = {plant._entr_plant_id};
    """
    plant.meter.df = pd.read_sql(meter_query, conn)

    load_meter_prepare(plant)

def load_meter_prepare(plant):

    plant._meter.df['time'] = pd.to_datetime(plant._meter.df["date_time"]).dt.tz_localize(None)
    plant._meter.df.set_index('time',inplace=True,drop=False)

    meter_map = {
        "MMTR.SupWh": "energy_kwh"
    }

    plant._meter.df.rename(meter_map, axis="columns", inplace=True)

## --- REANALYSIS ---

def load_reanalysis_meta(conn, plant):
    pass

def load_reanalysis(conn, plant, reanalysis_products):

    #load_reanalysis_meta(conn, plant)
    if reanalysis_products is None:
        return ## No reanalysis products were requested

    for product in reanalysis_products:

        product_query_string = product.upper()

        reanalysis_query = f"""
        SELECT
            date_time,
            `WMETR.HorWdSpdU`,
            `WMETR.HorWdSpdV`,
            `WMETR.EnvTmp`,
            `WMETR.EnvPres`,
            `WMETR.HorWdSpd`,
            `WMETR.HorWdDir`,
            `WMETR.AirDen`
        FROM
            entr_warehouse.openoa_reanalysis
        WHERE
            plant_id = {plant._entr_plant_id} AND
            reanalysis_dataset_name = "{product_query_string}";
        """
        plant.reanalysis._product[product.lower()].df = pd.read_sql(reanalysis_query, conn)

        load_reanalysis_prepare(plant, product=product)

def load_reanalysis_prepare(plant, product):

    # CASE: MERRA2
    if product.lower() == "merra2":
        
        # calculate wind direction from u, v
        plant._reanalysis._product['merra2'].df["winddirection_deg"] \
            = met.compute_wind_direction(plant._reanalysis._product['merra2'].df["WMETR.HorWdSpdU"], \
            plant._reanalysis._product['merra2'].df["WMETR.HorWdSpdV"])

        plant._reanalysis._product['merra2'].rename_columns({"time":"date_time",
                                    "windspeed_ms": "WMETR.HorWdSpd",
                                    "u_ms": "WMETR.HorWdSpdU",
                                    "v_ms": "WMETR.HorWdSpdV",
                                    "temperature_K": "WMETR.EnvTmp",
                                    "rho_kgm-3": "WMETR.AirDen"})
        #plant._reanalysis._product['merra2'].normalize_time_to_datetime("%Y-%m-%d %H:%M:%S")
        plant._reanalysis._product['merra2'].df['time'] = pd.to_datetime(plant._reanalysis._product['merra2'].df['time']).dt.tz_localize(None)
        plant._reanalysis._product['merra2'].df.set_index('time',inplace=True,drop=False)

    # CASE: ERA5
    elif product.lower() == 'era5':

        # calculate wind direction from u, v
        plant._reanalysis._product['era5'].df["winddirection_deg"] \
            = met.compute_wind_direction(plant._reanalysis._product['era5'].df["WMETR.HorWdSpdU"], \
            plant._reanalysis._product['era5'].df["WMETR.HorWdSpdV"])

        plant._reanalysis._product['era5'].rename_columns({"time":"date_time",
                                    "windspeed_ms": "WMETR.HorWdSpd",
                                    "u_ms": "WMETR.HorWdSpdU",
                                    "v_ms": "WMETR.HorWdSpdV",
                                    "temperature_K": "WMETR.EnvTmp",
                                    "rho_kgm-3": "WMETR.AirDen"})
        #plant._reanalysis._product['era5'].normalize_time_to_datetime("%Y-%m-%d %H:%M:%S")
        plant._reanalysis._product['era5'].df['time'] = pd.to_datetime(plant._reanalysis._product['era5'].df['time']).dt.tz_localize(None)
        plant._reanalysis._product['era5'].df.set_index('time',inplace=True,drop=False)

### UDFs

# from pyspark.sql.functions import udf
# from pyspark.sql.types import BooleanType
# from pyspark.sql import SQLContext
# from operational_analysis.toolkits.filters import range_flag

# openoa_range_flag_filter = udf(range_flag, BooleanType())

# def register_all(context: SQLContext):
#     context.udf.register("openoa_range_flag_filter", openoa_range_flag_filter)

