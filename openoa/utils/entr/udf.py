import pandas as pd
import pickle
from openoa.utils.entr.connection import PySparkEntrConnection

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
            "WMETR_EnvPres": "WMETR.EnvPres",
            "WMETR_EnvTmp": "WMETR.EnvTmp",
            "time": "date_time",
            "WMETR_HorWdSpdU": "WMETR.HorWdSpdU",
            "WMETR_HorWdSpdV": "WMETR.HorWdSpdV",
            "WMETR_AirDen":  "WMETR.AirDen"
        }

        metadata_dict[plant]["reanalysis"]["merra2"] =  {
            "frequency": "H", #TODO: Read this from Metadata tables
            "WMETR_EnvPres": "WMETR.EnvPres",
            "WMETR_EnvTmp": "WMETR.EnvTmp",
            "time": "date_time",
            "WMETR_HorWdSpdU": "WMETR.HorWdSpdU",
            "WMETR_HorWdSpdV": "WMETR.HorWdSpdV",
            "WMETR_AirDen":  "WMETR.AirDen"
        }

        metadata_dict[plant]["curtail"] = {
            "frequency": "1M", #TODO: Read this from Metadata tables
            "IAVL_DnWh": 'IAVL.DnWh',
            "IAVL_ExtPwrDnWh": 'IAVL.ExtPwrDnWh',
            "time": "date_time"
        }

        metadata_dict[plant]["meter"] = {
            "frequency": "10T", #TODO: Read this from Metadata tables
            "MMTR_SupWh": "MMTR.SupWh",
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