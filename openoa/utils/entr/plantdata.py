from openoa.utils import met_data_processing as met
from openoa.utils import unit_conversion as un
from openoa import PlantData
from openoa.plant import ANALYSIS_REQUIREMENTS
from openoa.utils.entr.connection import EntrConnection, PySparkEntrConnection
from time import perf_counter
import pandas as pd
from typing import Union

import logging

## --- PLANT LEVEL METADATA ---

def load_plant_metadata(conn:EntrConnection, plant_name:str) -> dict:
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

def load_plant_assets(conn:EntrConnection, plant_id):
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
        plant_id = {plant_id};
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


## --- Load Data from Fact Table and Pivot it

# def load_tag_meta_from_warehouse(conn:EntrConnection, plant_name:str, columns:list[str]):
#     pass

# def load_tag_values_from_warehouse(conn:EntrConnection, plant_name:str, columns:list[str]):
#     pass

## --- OpenOA Report Table and Metadata

entr_tables_dict = {
    "curtail": "openoa_curtailment_and_availability",
    "reanalysis": "openoa_reanalysis",
    "meter": "openoa_revenue_meter",
    "scada": "openoa_wtg_scada"
 }

def load_openoa_rpt_table(conn:EntrConnection, entr_plant_id:str, table_name:str, columns:list[str], reanalysis=None) -> pd.DataFrame:

    # Table name query fragment
    table_query_fragment = entr_tables_dict[table_name]

    # Column projection query fragment
    column_query_fragment = ",".join([f"float(`{column.replace('_','.')}`) as {column} " for column in columns])
    column_query_fragment += ", date_time as time"
    if table_name == "scada": ## Only scada has Turbine Name column
        column_query_fragment += ",entr_warehouse.openoa_wtg_scada.wind_turbine_name as WTUR_TurNam"

    # Filter query fragment
    filter_query_fragment = f"plant_id = {entr_plant_id}"
    if table_name == "reanalysis":
        filter_query_fragment += f" AND reanalysis_dataset_name = \"{reanalysis.upper()}\""

    # Build full query from fragments
    query = f"SELECT {column_query_fragment} FROM {table_query_fragment} WHERE {filter_query_fragment} ORDER BY time;"
    logging.debug(query)

    # Execute query in database, return pandas dataframe
    print(query)
    df = conn.pandas_query(query)

    return df


def load_openoa_rpt_table_tag_metadata(conn:EntrConnection, plant_name:str, table_name:str, columns:list[str], reanalysis=None) -> dict:
    # TODO: What about filtering by plant id? The current schema assumes all plants will have the same metadata.

    # Always the same three metadata columns
    column_query_fragment = "interval_s, value_type, value_units"

    # Metadata table name
    table_query_fragment = entr_tables_dict[table_name] + "_tag_metadata"

    # Filter by tags of interest
    tag_names_sql = ",".join([f'"{column}"' for column in columns])
    print(tag_names_sql)

    filter_query_fragment = f"entr_tag_name in ({tag_names_sql})"
    if table_name == "reanalysis":
        filter_query_fragment += f"AND reanalysis_dataset_name == \"{reanalysis.upper()}\""

    # Build simple select query
    query = f"SELECT {column_query_fragment} FROM {table_query_fragment} WHERE {filter_query_fragment};"

    # Execute query in database, return pandas dataframe
    df = conn.pandas_query(query)

    return df

def load_plant_reanalysis():
    reanalysis_metadata_dict = {}
    reanalysis_table_dict = {}
    for product in reanalysis_products:
        reanalysis_metadata_dict[product] = load_openoa_rpt_table_tag_metadata(...)
        reanalysis_table_dict[product] = load_openoa_rpt_table(...)
    return reanalysis_table_dict, reanalysis_metadata_dict


def from_entr(
    plant_name:str,
    schema:Union[str,dict]=None,
    connection:EntrConnection=None,
    reanalysis_products:list[str]=["merra2", "era5"]
)->PlantData:
    """
    from_entr
        Load a PlantData object from data in an entr_warehouse.
    
    Args:
        plant_name: Name of wind plant.
        schema: Schema dictionary. If this is a string, it will use schema defined in openoa.plant.ANALYSIS_REQUIREMENTS matching this name.
        connection: EntrConnection object to use to access the database.
        reanalysis_products: List of names of reanalysis products to retrieve.
    
    Returns:
        plant(PlantData): An OpenOA PlantData object.
    """
    tic = perf_counter()

    if connection is None:
        connection = PySparkEntrConnection()

    toc = perf_counter()
    logging.debug(f"{toc-tic} sec\tENTR Connection obtained")
    tic = perf_counter()

    # Get plant level metadata, including the plant_id, from the plant name string.
    plant_metadata = load_plant_metadata(connection, plant_name)
    plant_id = plant_metadata["_entr_plant_id"]

    # Grab schema from openoa.plant if it was provided as a string
    if type(schema) == str:
        analysis_type = schema
        schema = ANALYSIS_REQUIREMENTS[schema]
    assert type(schema) == dict, "schema must be a dictionary, or the name of an openoa analysis"

    combined_metadata = plant_metadata.copy()
    combined_tables = {}
    # Load meter, scada, availability, and curtailment tables into combined_tables and combined_metadata
    print(schema)
    for table,spec in schema.items():
        print(table)
        columns = spec["columns"]
        if table == "reanalysis":
            combined_tables["reanalysis"] = {}
            combined_metadata["reanalysis"] = {}
            for reanalysis_product in reanalysis_products:
                combined_tables["reanalysis"][reanalysis_product] = load_openoa_rpt_table(connection, plant_id, "reanalysis", spec["columns"], reanalysis=reanalysis_product)
                combined_metadata["reanalysis"][reanalysis_product] = load_openoa_rpt_table_tag_metadata(connection, plant_id, "reanalysis", spec["columns"], reanalysis=reanalysis_product)
        elif table == "asset":
            combined_tables[table], combined_metadata[table] = load_plant_assets(connection, plant_id)
        else:
            combined_metadata[table] = load_openoa_rpt_table_tag_metadata(connection, plant_id, table, spec["columns"])
            combined_tables[table] = load_openoa_rpt_table(connection, plant_id, table, spec["columns"])

    ## TODO: Any pre-processing?

    toc = perf_counter()
    logging.debug(f"{toc-tic} sec\tData loaded from Warehouse into Python")
    tic = perf_counter()

    print("METADATA:")
    print(combined_metadata)
    print("TABLES:")
    print(combined_tables)

    plant = PlantData(
        analysis_type=analysis_type,
        metadata=combined_metadata,
        **combined_tables
    )

    toc = perf_counter()
    logging.debug(f"{toc-tic} sec\tPlantData Object Created")

    return plant




# ## --- SCADA ---

# def load_scada_meta(conn:EntrConnection, plant_metadata:dict):
#     # Query the warehouse for any non-uniform metadata
#     query = f"""
#     SELECT
#         interval_s,
#         value_type,
#         value_units
#     FROM
#         entr_warehouse.openoa_wtg_scada_tag_metadata
#     WHERE
#         entr_tag_name = 'WTUR.W';
#     """
#     scada_meta_df = conn.pandas_query(query)

#     freq, _, _ = check_metadata_row(scada_meta_df.iloc[0], allowed_freq=['10T'], allowed_types=["average"], allowed_units=["W","Wh"])

#     # Build the metadata dictionary
#     scada_metadata = {
#         "frequency": freq,
#         "WTUR_TurNam": "wind_turbine_name",
#         "WTUR_W": "WTUR.W",
#         "WROT_BlPthAngVal": "WROT.BlPthAngVal",
#         "WMET_EnvTmp": "WMET.EnvTmp",
#         "time": "time",
#         "WMET_HorWdDir": "WMET.HorWdDir",
#         "WMET_HorWdSpd": "WMET.HorWdSpd"
#     }

#     return scada_metadata

# def load_scada(conn:EntrConnection, plant_metadata:dict):

#     scada_metadata = load_scada_meta(conn, plant_metadata)
    
#     scada_query = f"""
#     SELECT
#         entr_warehouse.openoa_wtg_scada.wind_turbine_name,
#         date_time,
#         float(`WROT.BlPthAngVal`) as `WROT.BlPthAngVal`,
#         float(`WTUR.W`) as `WTUR.W`,
#         float(`WMET.HorWdSpd`) as `WMET.HorWdSpd`,
#         float(`WMET.HorWdDirRel`) as `WMET.HorWdDirRel`,
#         float(`WMET.EnvTmp`) as `WMET.EnvTmp`,
#         float(`WNAC.Dir`) as `WNAC.Dir`,
#         float(`WMET.HorWdDir`) as `WMET.HorWdDir`,
#         float(`WTUR.SupWh`) as `WTUR.SupWh`
#     FROM
#         entr_warehouse.openoa_wtg_scada
#     WHERE
#         plant_id = {plant_metadata['_entr_plant_id']};
#     """
#     scada_df = conn.pandas_query(scada_query)
    
#     scada_df['time'] = pd.to_datetime(scada_df['date_time'],utc=True).dt.tz_localize(None)

#     # # Remove duplicated timestamps and turbine id
#     scada_df = scada_df.drop_duplicates(subset=['time','wind_turbine_name'],keep='first')

#     # # Set time as index
#     scada_df.set_index('time',inplace=True,drop=False)

#     scada_df = scada_df[(scada_df["WMET.EnvTmp"]>=-15.0) & (scada_df["WMET.EnvTmp"]<=45.0)]

#     # # Convert pitch to range -180 to 180.
#     scada_df["WROT.BlPthAngVal"] = scada_df["WROT.BlPthAngVal"] % 360
#     scada_df.loc[scada_df["WROT.BlPthAngVal"] > 180.0,"WROT.BlPthAngVal"] \
#         = scada_df.loc[scada_df["WROT.BlPthAngVal"] > 180.0,"WROT.BlPthAngVal"] - 360.0

#     # # Calculate energy
#     scada_df['energy_kwh'] = scada_df['WTUR.SupWh'] / 1000.0

#     return scada_df, scada_metadata


# def check_metadata_row(row, allowed_freq=["10T"], allowed_types=["sum"], allowed_units=["kWh"]):
#     """
#     Check the result of an ENTR Warehouse metadata query to make sure the values conform to our expectation.
#     """
#     accepted_freq = None
#     freq_long_str = f"{row['interval_s']} sec"
#     freq_timedelta = pd.Timedelta(freq_long_str)
#     for freq in allowed_freq:
#         if freq_timedelta == pd.Timedelta(freq): 
#             accepted_freq = freq
#             break
#     assert accepted_freq is not None, f"Unsupported time frequency {freq_long_str} does not match any allowed frequencies {allowed_freq}"

#     assert row["value_type"] in allowed_types, f"Unsupported value type {row['value_type']}"
#     assert row["value_units"] in allowed_units, f"Unsupported value type {row['value_units']}"

#     return accepted_freq, row["value_type"], row["value_units"]

# ## --- CURTAILMENT ---

# def load_curtailment_meta(conn:EntrConnection, plant_metadata:dict) -> dict:
#     query = f"""
#     SELECT
#         interval_s,
#         value_type,
#         value_units
#     FROM
#         entr_warehouse.openoa_curtailment_and_availability_tag_metadata
#     WHERE
#         entr_tag_name in ('IAVL.DnWh', 'IAVL.ExtPwrDnWh')
#     """
#     curtail_meta_df = conn.pandas_query(query)
#     freq, _, _ = check_metadata_row(curtail_meta_df.iloc[0], allowed_freq=['10T'], allowed_types=["sum"], allowed_units=["kWh"])

#     # Build the metadata dictionary
#     curtail_metadata = {
#         "frequency": freq,
#         "IAVL_DnWh": 'IAVL.DnWh',
#         "IAVL_ExtPwrDnWh": 'IAVL.ExtPwrDnWh',
#         "time": "date_time"
#     }

#     return curtail_metadata

# def load_curtailment(conn:EntrConnection, plant_metadata:dict):

#     curtail_metadata = load_curtailment_meta(conn, plant_metadata)

#     query = f"""
#     SELECT
#         date_time,
#         float(`IAVL.DnWh`) as `IAVL.DnWh`,
#         float(`IAVL.ExtPwrDnWh`) as `IAVL.ExtPwrDnWh`
#     FROM
#         entr_warehouse.openoa_curtailment_and_availability
#     WHERE
#         plant_id = {plant_metadata['_entr_plant_id']}
#     ORDER BY
#         date_time;
#     """

#     curtail_df = conn.pandas_query(query)

#     # Create datetime field
#     curtail_df['date_time'] = pd.to_datetime(curtail_df["date_time"]).dt.tz_localize(None)
#     curtail_df.set_index('date_time',inplace=True,drop=False)

#     return curtail_df, curtail_metadata


# ## --- METER ---

# def load_meter_meta(conn:EntrConnection, plant_metadata:dict) -> dict:
#     query = f"""
#     SELECT
#         interval_s,
#         value_type,
#         value_units
#     FROM
#         entr_warehouse.openoa_revenue_meter_tag_metadata
#     WHERE
#         entr_tag_name = 'MMTR.SupWh'
#     """
#     meter_meta_df = conn.pandas_query(query)

#     # Parse frequency
#     freq, _, _ = check_metadata_row(meter_meta_df.iloc[0], allowed_freq=['10T'], allowed_types=["sum"], allowed_units=["kWh"])

#     # Build the metadata dictionary
#     meter_metadata = {
#         "frequency": freq,
#         "MMTR_SupWh": "MMTR.SupWh",
#         "time": "date_time"
#     }

#     return meter_metadata

# def load_meter(conn:EntrConnection, plant_metadata:dict):

#     meter_metadata = load_meter_meta(conn, plant_metadata)

#     meter_query = f"""
#     SELECT
#         date_time,
#         float(`MMTR.SupWh`) as `MMTR.SupWh`
#     FROM
#         entr_warehouse.openoa_revenue_meter
#     WHERE
#         plant_id = {plant_metadata['_entr_plant_id']};
#     """
#     meter_df = conn.pandas_query(meter_query)

#     meter_df['date_time'] = pd.to_datetime(meter_df["date_time"]).dt.tz_localize(None)
#     meter_df.set_index('date_time',inplace=True,drop=False)

#     return meter_df, meter_metadata

# ## --- REANALYSIS ---


# def load_reanalysis(conn:EntrConnection, plant_metadata:dict, reanalysis_products):

#     #load_reanalysis_meta(conn, plant)
#     if reanalysis_products is None:
#         return ## No reanalysis products were requested

#     reanalysis_df_dict = {}
#     reanalysis_meta_dict = {}

#     for product in reanalysis_products:

#         product_query_string = product.upper()

#         reanalysis_query = f"""
#         SELECT
#             date_time,
#             float(`WMETR.HorWdSpdU`) as `WMETR.HorWdSpdU`,
#             float(`WMETR.HorWdSpdV`) as `WMETR.HorWdSpdV`,
#             float(`WMETR.EnvTmp`) as `WMETR.EnvTmp`,
#             float(`WMETR.EnvPres`) as `WMETR.EnvPres`,
#             float(`WMETR.HorWdSpd`) as `WMETR.HorWdSpd`,
#             float(`WMETR.HorWdDir`) as `WMETR.HorWdDir`,
#             float(`WMETR.AirDen`) as `WMETR.AirDen`
#         FROM
#             entr_warehouse.openoa_reanalysis
#         WHERE
#             plant_id = {plant_metadata['_entr_plant_id']} AND
#             reanalysis_dataset_name = "{product_query_string}";
#         """
#         reanalysis_df = conn.pandas_query(reanalysis_query)

#         reanalysis_df["winddirection_deg"] = met.compute_wind_direction(reanalysis_df["WMETR.HorWdSpdU"], reanalysis_df["WMETR.HorWdSpdV"])
#         reanalysis_df['date_time'] = pd.to_datetime(reanalysis_df['date_time']).dt.tz_localize(None)
#         reanalysis_df.set_index('date_time',inplace=True,drop=False)

#         reanalysis_metadata = {
#             "frequency": "H", #TODO: Read this from Metadata tables
#             "WMETR_EnvPres": "WMETR.EnvPres",
#             "WMETR_EnvTmp": "WMETR.EnvTmp",
#             "time": "date_time",
#             "WMETR_HorWdSpdU": "WMETR.HorWdSpdU",
#             "WMETR_HorWdSpdV": "WMETR.HorWdSpdV"
#         }

#         reanalysis_df_dict[product] = reanalysis_df
#         reanalysis_meta_dict[product] = reanalysis_metadata

#     return reanalysis_df_dict, reanalysis_meta_dict