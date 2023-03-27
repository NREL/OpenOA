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

plant_metadata["scada": scada_metadata
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
import pickle

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
            .config("spark.driver.memory", "4g")\
            .config("spark.executor.memory", "4g")\
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
        float(latitude) as latitude,
        float(longitude) as longitude,
        float(elevation) as elevation,
        float(hub_height) as hub_height,
        float(rotor_diameter) as rotor_diameter,
        float(rated_power) as rated_power,
        manufacturer,
        model
    FROM
        entr_warehouse.dim_asset_wind_turbine
    WHERE
        plant_id = {plant_metadata['_entr_plant_id']};
    """
    asset_df = conn.pandas_query(asset_query)

    asset_metadata = {
        "elevation":"elevation",
        "id":"wind_turbine_name",
        "latitude":"latitude",
        "longitude":"longitude",
        "rated_power":"rated_power",
        "rotor_diameter":"rotor_diameter"
    }

    return asset_df, asset_metadata

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
        "frequency": freq,
        "id": "wind_turbine_name",
        "power": "WTUR.W",
        "pitch": "WROT.BlPthAngVal",
        "temperature": "WMET.EnvTmp",
        "time": "time",
        "wind_direction": "WMET.HorWdDir",
        "windspeed": "WMET.HorWdSpd"
    }

    return scada_metadata

def load_scada(conn:EntrConnection, plant_metadata:dict):

    scada_metadata = load_scada_meta(conn, plant_metadata)
    
    scada_query = f"""
    SELECT
        entr_warehouse.openoa_wtg_scada.wind_turbine_name,
        date_time,
        float(`WROT.BlPthAngVal`) as `WROT.BlPthAngVal`,
        float(`WTUR.W`) as `WTUR.W`,
        float(`WMET.HorWdSpd`) as `WMET.HorWdSpd`,
        float(`WMET.HorWdDirRel`) as `WMET.HorWdDirRel`,
        float(`WMET.EnvTmp`) as `WMET.EnvTmp`,
        float(`WNAC.Dir`) as `WNAC.Dir`,
        float(`WMET.HorWdDir`) as `WMET.HorWdDir`,
        float(`WTUR.SupWh`) as `WTUR.SupWh`
    FROM
        entr_warehouse.openoa_wtg_scada
    WHERE
        plant_id = {plant_metadata['_entr_plant_id']};
    """
    scada_df = conn.pandas_query(scada_query)
    
    scada_df['time'] = pd.to_datetime(scada_df['date_time'],utc=True).dt.tz_localize(None)

    # # Remove duplicated timestamps and turbine id
    scada_df = scada_df.drop_duplicates(subset=['time','wind_turbine_name'],keep='first')

    # # Set time as index
    scada_df.set_index('time',inplace=True,drop=False)

    scada_df = scada_df[(scada_df["WMET.EnvTmp"]>=-15.0) & (scada_df["WMET.EnvTmp"]<=45.0)]

    # # Convert pitch to range -180 to 180.
    scada_df["WROT.BlPthAngVal"] = scada_df["WROT.BlPthAngVal"] % 360
    scada_df.loc[scada_df["WROT.BlPthAngVal"] > 180.0,"WROT.BlPthAngVal"] \
        = scada_df.loc[scada_df["WROT.BlPthAngVal"] > 180.0,"WROT.BlPthAngVal"] - 360.0

    # # Calculate energy
    scada_df['energy_kwh'] = scada_df['WTUR.SupWh'] / 1000.0

    return scada_df, scada_metadata


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

def load_curtailment_meta(conn:EntrConnection, plant_metadata:dict) -> dict:
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
    curtail_meta_df = conn.pandas_query(query)
    freq, _, _ = check_metadata_row(curtail_meta_df.iloc[0], allowed_freq=['10T'], allowed_types=["sum"], allowed_units=["kWh"])

    # Build the metadata dictionary
    curtail_metadata = {
        "frequency": freq,
        "availability": 'IAVL.DnWh',
        "curtailment": 'IAVL.ExtPwrDnWh',
        "time": "date_time"
    }

    return curtail_metadata

def load_curtailment(conn:EntrConnection, plant_metadata:dict):

    curtail_metadata = load_curtailment_meta(conn, plant_metadata)

    query = f"""
    SELECT
        date_time,
        float(`IAVL.DnWh`) as `IAVL.DnWh`,
        float(`IAVL.ExtPwrDnWh`) as `IAVL.ExtPwrDnWh`
    FROM
        entr_warehouse.openoa_curtailment_and_availability
    WHERE
        plant_id = {plant_metadata['_entr_plant_id']}
    ORDER BY
        date_time;
    """

    curtail_df = conn.pandas_query(query)

    # Create datetime field
    curtail_df['date_time'] = pd.to_datetime(curtail_df["date_time"]).dt.tz_localize(None)
    curtail_df.set_index('date_time',inplace=True,drop=False)

    return curtail_df, curtail_metadata


## --- METER ---

def load_meter_meta(conn:EntrConnection, plant_metadata:dict) -> dict:
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
    meter_meta_df = conn.pandas_query(query)

    # Parse frequency
    freq, _, _ = check_metadata_row(meter_meta_df.iloc[0], allowed_freq=['10T'], allowed_types=["sum"], allowed_units=["kWh"])

    # Build the metadata dictionary
    meter_metadata = {
        "frequency": freq,
        "energy": "MMTR.SupWh",
        "time": "date_time"
    }

    return meter_metadata

def load_meter(conn:EntrConnection, plant_metadata:dict):

    meter_metadata = load_meter_meta(conn, plant_metadata)

    meter_query = f"""
    SELECT
        date_time,
        float(`MMTR.SupWh`) as `MMTR.SupWh`
    FROM
        entr_warehouse.openoa_revenue_meter
    WHERE
        plant_id = {plant_metadata['_entr_plant_id']};
    """
    meter_df = conn.pandas_query(meter_query)

    meter_df['date_time'] = pd.to_datetime(meter_df["date_time"]).dt.tz_localize(None)
    meter_df.set_index('date_time',inplace=True,drop=False)

    return meter_df, meter_metadata

## --- REANALYSIS ---


def load_reanalysis(conn:EntrConnection, plant_metadata:dict, reanalysis_products):

    #load_reanalysis_meta(conn, plant)
    if reanalysis_products is None:
        return ## No reanalysis products were requested

    reanalysis_df_dict = {}
    reanalysis_meta_dict = {}

    for product in reanalysis_products:

        product_query_string = product.upper()

        reanalysis_query = f"""
        SELECT
            date_time,
            float(`WMETR.HorWdSpdU`) as `WMETR.HorWdSpdU`,
            float(`WMETR.HorWdSpdV`) as `WMETR.HorWdSpdV`,
            float(`WMETR.EnvTmp`) as `WMETR.EnvTmp`,
            float(`WMETR.EnvPres`) as `WMETR.EnvPres`,
            float(`WMETR.HorWdSpd`) as `WMETR.HorWdSpd`,
            float(`WMETR.HorWdDir`) as `WMETR.HorWdDir`,
            float(`WMETR.AirDen`) as `WMETR.AirDen`
        FROM
            entr_warehouse.openoa_reanalysis
        WHERE
            plant_id = {plant_metadata['_entr_plant_id']} AND
            reanalysis_dataset_name = "{product_query_string}";
        """
        reanalysis_df = conn.pandas_query(reanalysis_query)

        reanalysis_df["winddirection_deg"] = met.compute_wind_direction(reanalysis_df["WMETR.HorWdSpdU"], reanalysis_df["WMETR.HorWdSpdV"])
        reanalysis_df['date_time'] = pd.to_datetime(reanalysis_df['date_time']).dt.tz_localize(None)
        reanalysis_df.set_index('date_time',inplace=True,drop=False)

        reanalysis_metadata = {
            "frequency": "H", #TODO: Read this from Metadata tables
            "surface_pressure": "WMETR.EnvPres",
            "temperature": "WMETR.EnvTmp",
            "time": "date_time",
            "windspeed_u": "WMETR.HorWdSpdU",
            "windspeed_v": "WMETR.HorWdSpdV"
        }

        reanalysis_df_dict[product] = reanalysis_df
        reanalysis_meta_dict[product] = reanalysis_metadata

    return reanalysis_df_dict, reanalysis_meta_dict


### Multi-Plant AEP Map

def aep_spark_map_udf(aep_spark_map_udf_broadcast_metadata, plant_name, df):
    """
    Pyspark UDF parial, applied via applyInPandas
    """
    from openoa.analysis import MonteCarloAEP
    from openoa.plant import PlantData

    fct_pandas = df
    plant_name = plant_name[0] # Spark assigns group key as a tuple, which would generalize to multiple group keys

    #global aep_spark_map_udf_broadcast_metadata
    plant_meta = aep_spark_map_udf_broadcast_metadata.value[plant_name]

    openoa_era5 = fct_pandas[(fct_pandas["table"]=="reanalysis") &\
                               (fct_pandas["reanalysis_dataset_name"]=="ERA5")]\
                            .pivot(index="date_time", columns="entr_tag_name", values="tag_value")\
                            .reset_index()

    # OpenOA MERRA2 table
    openoa_merra2 = fct_pandas[(fct_pandas["table"]=="reanalysis") &\
                               (fct_pandas["reanalysis_dataset_name"]=="MERRA2")]\
                            .pivot(index="date_time", columns="entr_tag_name", values="tag_value")\
                            .reset_index()

    # OpenOA Curtail table
    # TODO: We are resampling the availability and curtailment values to monthly. This prevents an index error in OpenOA when these variables have different frequencies.
    openoa_curtail = fct_pandas[(fct_pandas["table"]=="plant") &\
                               (fct_pandas["entr_tag_name"].isin(["IAVL.DnWh", "IAVL.ExtPwrDnWh"]))]\
                            .pivot(index="date_time", columns="entr_tag_name", values="tag_value")\
                            .resample("1M").sum()\
                            .reset_index()

    #penOA Meter table
    openoa_meter = fct_pandas[(fct_pandas["table"]=="plant") &\
                               (fct_pandas["entr_tag_name"].isin(["MMTR.SupWh"]))]\
                            .pivot(index="date_time", columns="entr_tag_name", values="tag_value")\
                            .reset_index()

    # Build the OpenOA Plant Data Object
    project = PlantData(
        analysis_type="MonteCarloAEP",  # No validation desired at this point in time
        metadata=plant_meta,
        meter=openoa_meter,
        curtail=openoa_curtail,
        reanalysis={"era5": openoa_era5, "merra2": openoa_merra2},
    )
    # Run the MonteCarloAEP
    pa = MonteCarloAEP(project, reanalysis_products = ['era5', 'merra2'])
    pa.run(num_sim=1000)
    # Serialize and return DDL "plant_name string, aep_GWh float"
    # return pd.DataFrame([{"plant_name": plant_name,
    #                      "aep_GWh": pa.results.aep_GWh.mean()}]) # TODO: Return the whole series, and other numbers in this structure, as a complex type
    # TODO: Also return availability and curtaulment estimates
    return pd.DataFrame([{
        "plant_name": plant_name,
        "results": pickle.dumps(pa.results)
    }])


def aep_spark_map_build_metadata(conn:PySparkEntrConnection, plants:list):
    """
    Driver-side function to build metadata dictionaries required for OpenOA constructor
    """
    import pyspark.sql.functions as f

    spark = conn._conn # Private varaible _conn defined in PySparkEntrConnection class.
    metadata_dict = {}

    dim_asset_wind_plant = spark.table("dim_asset_wind_plant").filter(f.col("plant_name").isin(plants)).toPandas()

    for plant in plants:

        plant_meta = dim_asset_wind_plant[dim_asset_wind_plant["plant_name"]==plant]

        metadata_dict[plant] = {
            "latitude": float(plant_meta.latitude),
            "longitude": float(plant_meta.longitude),
            "capacity": float(plant_meta.plant_capacity),
            "number_of_turbines": int(plant_meta.number_of_turbines),
            "turbine_capacity": float(plant_meta.turbine_capacity),
            "_entr_plant_id": int(plant_meta.plant_id)
        }

        metadata_dict[plant]["reanalysis"] = {}

        metadata_dict[plant]["reanalysis"]["era5"] =  {
            "frequency": "H", #TODO: Read this from Metadata tables
            "surface_pressure": "WMETR.EnvPres",
            "temperature": "WMETR.EnvTmp",
            "time": "date_time",
            "windspeed_u": "WMETR.HorWdSpdU",
            "windspeed_v": "WMETR.HorWdSpdV",
            "density":  "WMETR.AirDen"
        }

        metadata_dict[plant]["reanalysis"]["merra2"] =  {
            "frequency": "H", #TODO: Read this from Metadata tables
            "surface_pressure": "WMETR.EnvPres",
            "temperature": "WMETR.EnvTmp",
            "time": "date_time",
            "windspeed_u": "WMETR.HorWdSpdU",
            "windspeed_v": "WMETR.HorWdSpdV",
            "density":  "WMETR.AirDen"
        }

        metadata_dict[plant]["curtail"] = {
            "frequency": "1M", #TODO: Read this from Metadata tables
            "availability": 'IAVL.DnWh',
            "curtailment": 'IAVL.ExtPwrDnWh',
            "time": "date_time"
        }

        metadata_dict[plant]["meter"] = {
            "frequency": "10T", #TODO: Read this from Metadata tables
            "energy": "MMTR.SupWh",
            "time": "date_time"
        }

    return metadata_dict



from functools import partial

def aep_spark_map(conn:PySparkEntrConnection, plants:list):
    """
    Map the OpenOA AEP Method over plants listed in argument list by name.

    Arguments:
        conn:PysparkEntrConnection
        plants:list(str)
    """
    import pyspark.sql.functions as f
    spark = conn._conn # Private varaible _conn defined in PySparkEntrConnection class.

    # Retrieve Metadata into a broadcast variable
    print("Collecting metadata for query")

    plant_meta = aep_spark_map_build_metadata(conn, plants)
    print(plant_meta)
    aep_spark_map_udf_broadcast_metadata = spark.sparkContext.broadcast(plant_meta)
    # plants_ids = [int(plant_meta.value[plant_meta.value['plant_name']==name]['plant_id'].values[0]) for name in plants]

    ## Build main spark query
    print("Building the main query")

    # Build query for facts needed to run AEP
    fct_entr_plant_data = spark.table("fct_entr_plant_data")\
        .withColumn("table", f.lit("plant"))\

    fct_entr_reanalysis_data = spark.table("fct_entr_reanalysis_data")\
        .withColumn("table", f.lit("reanalysis"))

    fct_all = fct_entr_plant_data\
        .unionByName(fct_entr_reanalysis_data, allowMissingColumns=True)

    # Join facts with tag names
    dim_entr_tag_list = spark.table("dim_entr_tag_list")
    fct_all = fct_all.join(dim_entr_tag_list.select(f.col("entr_tag_name"), f.col("entr_tag_id")), on="entr_tag_id", how="left")

    # Join facts with reanalysis dataset names
    dim_asset_reanalysis_dataset = spark.table("dim_asset_reanalysis_dataset")
    fct_all = fct_all.join(dim_asset_reanalysis_dataset.select(f.col("reanalysis_dataset_id"), f.col("reanalysis_dataset_name")), on="reanalysis_dataset_id", how="left")

    # Join facts with plant name
    dim_asset_wind_plant = spark.table("dim_asset_wind_plant")
    fct_all = fct_all.join(dim_asset_wind_plant.select(f.col("plant_id"), f.col("plant_name")), on="plant_id", how="left")

    # Project the fact table down to only the columns we need for the AEP query
    # TODO: Programatically set these columns based on plant metadata
    required_tags = ["IAVL.ExtPwrDnWh", # curtail.curtailment
                 "IAVL.DnWh", # curtail.availability
                 "MMTR.SupWh", # meter.energy (need to convert from power?)
                 "WMETR.HorWdSpd", # reanalysis.windspeed
                 "WMETR.AirDen", # reanalysis.density
                 "WMETR.HorWdSpdU", # reanalysis.windspeed_u
                 "WMETR.HorWdSpdV", # reanalysis.windspeed_v
                 "WMETR.EnvTmp"] # reanalysis.temperature

    # Filter based on plant IDs and tag name, project to columns we need
    fct_all = fct_all\
        .filter(f.col("plant_name").isin(plants))\
        .filter(f.col("entr_tag_name").isin(required_tags))\
        .select(["plant_name", "plant_id", "entr_tag_name", "date_time", "tag_value", "reanalysis_dataset_name", "table"])

    ## Execute main query, running the aep_spark_map_udf in parallel across the Spark cluster. One instance for each plant_id.
    print("Running the main query")
    #res = fct_all.groupby("plant_name").applyInPandas(partial(aep_spark_map_udf, aep_spark_map_udf_broadcast_metadata), schema="plant_name string, aep_GWh float").toPandas()

    res = fct_all.groupby("plant_name")\
        .applyInPandas(partial(aep_spark_map_udf, aep_spark_map_udf_broadcast_metadata), schema="plant_name string, results binary")\
        .toPandas()

    res["results"] = res["results"].apply(pickle.loads)

    res = [(row[0], row[1]) for _,row in res.iterrows()]

    # Remove metadata from spark executors
    aep_spark_map_udf_broadcast_metadata.unpersist()

    return res

### UDFs

# from pyspark.sql.functions import udf
# from pyspark.sql.types import BooleanType
# from pyspark.sql import SQLContext
# from operational_analysis.toolkits.filters import range_flag

# openoa_range_flag_filter = udf(range_flag, BooleanType())

# def register_all(context: SQLContext):
#     context.udf.register("openoa_range_flag_filter", openoa_range_flag_filter)

#aep_spark_map_udf_broadcast_metadata = None