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


