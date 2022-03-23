"""
ENTR OpenOA Toolkit
Provides utility functions to load data from ENTR warehouse into PlantData objects
"""

import pandas as pd

_conn = None

def get_connection(thrift_server_host,thrift_server_port):
    """
    If connection is not valid, get a new connection and return it.
    """
    global _conn
    if _conn is None:
        from pyhive import hive
        _conn = hive.Connection(host=thrift_server_host, port=thrift_server_port)
    return _conn

def do_query(conn, query):
    df = pd.read_sql(query, conn)
    return df

def load_metadata(conn, plant):
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
        entr_warehouse.dim_asset_plant
    WHERE
        plant_name = "{plant.name}";
    """
    metadata = pd.read_sql(metadata_query, conn)

    assert len(metadata)<2, f"Multiple plants matching name {wind_plant}"
    assert len(metadata)>0, f"No plant matching name {wind_plant}"

    plant.latitude = metadata["latitude"][0]
    plant.longitude = metadata["longitude"][0]
    plant.capacity = metadata["plant_capacity"][0]
    plant._entr_plant_id = metadata["plant_id"][0]

def load_asset(conn, plant):
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
        plant_id = {plant._entr_plant_id};
    """
    #plant._asset = pyspark.sql(asset_query).to_pandas()
    plant._asset = pd.read_sql(asset_query, conn)

def load_scada(conn, plant):
    scada_query = f"""
    SELECT
        entr_warehouse.rpt_openoa_wtg_scada.wind_turbine_name,
        date_time,
        `WROT.BlPthAngVal`,
        `WTUR.W`,
        `WMET.HorWdSpd`,
        `WMET.HorWdDirRel`,
        `WMET.EnvTmp`,
        `WNAC.Dir`,
        `WMET.HorWdDir`
    FROM
        entr_warehouse.rpt_openoa_wtg_scada
        LEFT JOIN
            entr_warehouse.dim_asset_wind_turbine
        ON entr_warehouse.rpt_openoa_wtg_scada.wind_turbine_name = entr_warehouse.dim_asset_wind_turbine.wind_turbine_name
    WHERE
        plant_id = {plant._entr_plant_id};
    """
    plant.scada.df = pd.read_sql(scada_query, conn)

def check_metadata_row(row, allowed_freq=["10T"], allowed_types=["sum"], allowed_units=["kWh"]):
    
    accepted_freq = None
    freq_long_str = f"{row['interval_n']} {row['interval_units']}"
    freq_timedelta = pd.Timedelta(freq_long_str)
    for freq in allowed_freq:
        if freq_timedelta == pd.Timedelta(freq): 
            accepted_freq = freq
            break
    assert accepted_freq is not None, f"Unsupported time frequency {freq_long_str} does not match any allowed frequencies {allowed_freq}"

    assert row["value_type"] in allowed_types, f"Unsupported value type {row['value_type']}"
    assert row["value_units"] in allowed_units, f"Unsupported value type {row['value_units']}"

    return accepted_freq, row["value_type"], row["value_units"]

def load_curtailment_meta(conn, plant):
    query = f"""
    SELECT
        interval_n,
        interval_units,
        value_type,
        value_units
    FROM
        entr_warehouse.rpt_openoa_curtailment_and_availability_tag_metadata
    WHERE
        entr_tag_name in ('IAVL.DnWh', 'IAVL.ExtPwrDnWh')
    """
    meter_meta_df = pd.read_sql(query, conn)
    # TODO: Check metadata using check_metadata_row 

def load_curtailment(conn, plant):

    load_curtailment_meta(conn, plant)

    query = f"""
    SELECT
        date_time,
        `IAVL.DnWh`,
        `IAVL.ExtPwrDnWh`
    FROM
        entr_warehouse.rpt_openoa_curtailment_and_availability
        LEFT JOIN
            entr_warehouse.dim_asset_plant
        ON entr_warehouse.rpt_openoa_curtailment_and_availability.plant_name = entr_warehouse.dim_asset_plant.plant_name
    WHERE
        plant_id = {plant._entr_plant_id}
    ORDER BY
        date_time;
    """
    plant.curtail.df = pd.read_sql(query, conn)


def load_meter_meta(conn, plant):
    query = f"""
    SELECT
        interval_n,
        interval_units,
        value_type,
        value_units
    FROM
        entr_warehouse.rpt_openoa_revenue_meter_tag_metadata
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
        entr_warehouse.rpt_openoa_revenue_meter
        LEFT JOIN
            entr_warehouse.dim_asset_plant
        ON entr_warehouse.rpt_openoa_revenue_meter.plant_name = entr_warehouse.dim_asset_plant.plant_name
    WHERE
        plant_id = {plant._entr_plant_id};
    """
    plant.meter.df = pd.read_sql(meter_query, conn)

def load_openoa_project_from_warehouse(cls, thrift_server_host="localhost",
                       thrift_server_port=10000,
                       database="entr_warehouse",
                       wind_plant="La Haute Borne",
                       aggregation="",
                       date_range=None,
                       conn=None):
    plant = cls(database, wind_plant) ## Passing in database as the path and wind_plant as the name for now.
        
    plant.name = wind_plant

    conn = get_connection(thrift_server_host, thrift_server_port)

    load_metadata(conn, plant)
    load_asset(conn, plant)
    load_scada(conn, plant)
    load_curtailment(conn, plant)
    load_meter(conn, plant)

    return plant