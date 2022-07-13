"""
ENTR OpenOA Toolkit
Provides utility functions to load data from ENTR warehouse into PlantData objects
"""

import pandas as pd
import operational_analysis.toolkits.unit_conversion as un
import operational_analysis.toolkits.met_data_processing as met

_conn = None

def get_connection(thrift_server_host,thrift_server_port):
    """
    Using the host and port to get a connection to the thrift server.
    If a connection is already instantiated, it will return that same connection.

    Returns: PyHive.Connection
    """
    global _conn
    if _conn is None:
        from pyhive import hive
        _conn = hive.Connection(host=thrift_server_host, port=thrift_server_port)
    return _conn

def do_query(conn, query):
    """
    Use the connection object to run a query, returning a dataframe.
    """
    df = pd.read_sql(query, conn)
    return df

## --- PLANT LEVEL METADATA ---

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
        entr_warehouse.dim_asset_wind_plant
    WHERE
        plant_name = "{plant.name}";
    """
    metadata = pd.read_sql(metadata_query, conn)

    assert len(metadata)<2, f"Multiple plants matching name {wind_plant}"
    assert len(metadata)>0, f"No plant matching name {wind_plant}"

    plant.latitude = metadata["latitude"][0]
    plant.longitude = metadata["longitude"][0]
    plant._plant_capacity = metadata["plant_capacity"][0]
    plant._num_turbines = metadata["number_of_turbines"][0]
    plant._turbine_capacity = metadata["turbine_capacity"][0]
    plant._entr_plant_id = metadata["plant_id"][0]

## --- ASSET ---

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

## --- SCADA ---

def load_scada_meta(conn, plant):
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
    meter_meta_df = pd.read_sql(query, conn)

    # Parse frequency
    freq, _, _ = check_metadata_row(meter_meta_df.iloc[0], allowed_freq=['10T'], allowed_types=["average"], allowed_units=["W","Wh"])
    plant._scada_freq = freq

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


