from openoa.utils import met_data_processing as met
from openoa.utils import unit_conversion as un
from openoa import PlantData
from openoa.utils.entr.connection import EntrConnection, PySparkEntrConnection
from time import perf_counter
import pandas as pd

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



# Todo: Demonstrate working openoav3 constructor with Hive
# Todo: Add Pyspark constructor with option to preserve pyspark dataframe

def from_entr(
    cls,
    plant_name:str,
    analysis_type:str=None,
    connection:EntrConnection=None,
    reanalysis_products:list[str]=["merra2", "era5"]
)->PlantData:
    """
    from_entr
        Load a PlantData object from data in an entr_warehouse.
    
    Args:
            connection_type(str): pyspark|pyhive
            thrift_server_host(str): URL of the Apache Thrift server
            thrift_server_port(int): Port of the Apache Thrift server
            database(str): Name of the Hive database
            wind_plant(str): Name of the wind plant you'd like to load
            reanalysis_products(list[str]): Reanalysis products to load from the warehouse.
            aggregation: Not yet implemented
            date_range: Not yet implemented
    
    Returns:
            plant(PlantData): An OpenOA PlantData object.
    """

    tic = perf_counter()

    if connection is None:
        connection = PySparkEntrConnection()

    toc = perf_counter()
    print(f"{toc-tic} sec\tENTR Connection obtained")

    tic = perf_counter()
    plant_metadata = load_metadata(connection, plant_name)
    asset_df, asset_metadata = load_asset(connection, plant_metadata)
    scada_df, scada_metadata = load_scada(connection, plant_metadata)
    curtail_df, curtail_metadata = load_curtailment(connection, plant_metadata)
    meter_df, meter_metadata = load_meter(connection, plant_metadata)
    reanalysis_df_dict, reanalysis_metadata_dict = load_reanalysis(connection, plant_metadata, reanalysis_products)
    toc = perf_counter()
    print(f"{toc-tic} sec\tData loaded from Warehouse into Python")


    combined_metadata = plant_metadata.copy()
    combined_metadata["asset"] = asset_metadata
    combined_metadata["scada"] = scada_metadata
    combined_metadata["curtail"] = curtail_metadata
    combined_metadata["meter"] = meter_metadata
    combined_metadata["reanalysis"] = reanalysis_metadata_dict

    tic = perf_counter()

    plant = PlantData(
        analysis_type=analysis_type,  # No validation desired at this point in time
        metadata=combined_metadata,
        scada=scada_df,
        meter=meter_df,
        curtail=curtail_df,
        asset=asset_df,
        reanalysis=reanalysis_df_dict,
    )

    toc = perf_counter()
    print(f"{toc-tic} sec\tPlantData Object Creation")

    return plant
