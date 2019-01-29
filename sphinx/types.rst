.. types:

Project Data
==================

New data imported into the operational_analysis toolkit can take advantage of the data structures in the `types` module.
The base class is operational_analysis.types.PlantData, which represents all known pieces of data about a given wind plant.


Schemas
-------

Operational Data
^^^^^^^^^^^^^^^^

PlantData.scada

==================== ====================
 Field Name           Data Type
==================== ====================
 time                 datetime64[ns]
 id                   string
 power_kw             float64
 windspeed_ms         float64
 winddirection_deg    float64
 status_label         string
 pitch_deg            float64
 temp_c               float64
==================== ====================

PlantData.meter

==================== ====================
 Field Name           Data Type
==================== ====================
 time                 datetime64[ns]
 power_kw             float64
 energy_kw            float64
==================== ====================


PlantData.tower

==================== ====================
 Field Name           Data Type
==================== ====================
 time                 datetime64[ns]
 id                   float64
==================== ====================

PlantData.curtail

==================== ====================
 Field Name           Data Type
==================== ====================
 time                 datetime64[ns]
 curtailment_pct      float64
 availability_pct     float64
 net_energy           float64
==================== ====================


PlantData.status

==================== ====================
 Field Name           Data Type
==================== ====================
 time                 datetime64[ns]
 id                   string
 status_id            int64
 status_code          int64
 status_text          string
==================== ====================


PlantData.asset

==================== ====================
 Field Name           Data Type
==================== ====================
 id                   string
 latitude             float64
 longitude            float64
 rated_power_kw       float64
 type                 string
==================== ====================


Reanalysis Products
^^^^^^^^^^^^^^^^^^^

Reanalysis products are included as Plant Data objects and, regardless
of data source, have a standardized set of field names and types (see below).
That said, the data sources are obviously different, as are the methods use to 
calculate these standard fields from the raw datasets. These methods are described here.

PlantData.reanalysis.product["merra2"]

MERRA-2 data are based on the single-level diagnostic data available here:

https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_V5.12.4/summary?keywords=%22MERRA-2%22

Wind speed and direction are taken directly from the diagnostic 50-m u- and v-wind 
fields provided in this dataset. Air density at 50m is calculated using temperature
and pressure estimations at 50m and the ideal gas law. Temperature at 50m is estimated by taking the 10-m
temperature data provided by this dataset and assuming a constant lapse rate of -9.8 
degrees Celsius per vertical kilometer. Pressure at 50m is extrapolated from surface pressure
data provided in this dataset using the hypsometric equation.

==================== ====================
 Field Name           Data Type
==================== ====================
 time                 datetime64[ns]
 windspeed_ms         float64
 winddirection_deg    float64
 rho_kgm-3            float64
==================== ====================

PlantData.reanalysis.product["ncep2"]

NCEP-2 data are based on the single-level diagnostic data available here:

https://rda.ucar.edu/datasets/ds091.0/

Wind speed and direction are taken directly from the diagnostic 10-m u- and v-wind 
fields provided in this dataset. Air density at 10m is calculated using temperature
and pressure estimations at 10m and the ideal gas law. Temperature at 10m is estimated by taking the 2-m
temperature data provided by this dataset and assuming a constant lapse rate of -9.8 
degrees Celsius per vertical kilometer. Pressure at 10m is extrapolated from surface pressure
data provided in this dataset using the hypsometric equation.

==================== ====================
 Field Name           Data Type
==================== ====================
 time                 datetime64[ns]
 windspeed_ms         float64
 winddirection_deg    float64
 rho_kgm-3            float64
==================== ====================


PlantData.reanalysis.product["erai"]

ERA-interim data are based on the model-level data available here:

https://rda.ucar.edu/datasets/ds627.0/

Model levels are based on sigma coordinates (i.e. fractions of surface pressure). From this dataset, we 
extract temperature, u-wind, and v-wind at the 58th model level, which is on average about 72m above ground level 
(https://www.ecmwf.int/en/forecasts/documentation-and-support/60-model-levels). We also extract surface pressure 
data. Air density at the 58th model level is calculated using temperature data extracted at that level and an estimation 
of pressure at that level using the ideal gas law. Pressure at the 58th model level is extrapolated from surface pressure
data provided in this dataset using the hypsometric equation.

==================== ====================
 Field Name           Data Type
==================== ====================
 time                 datetime64[ns]
 windspeed_ms         float64
 winddirection_deg    float64
 rho_kgm-3            float64
==================== ====================



PlantData
----------------------------------------

.. automodule:: operational_analysis.types.plant
    :members:
    :no-undoc-members:
    :show-inheritance:


AssetData
----------------------------------------

.. automodule:: operational_analysis.types.asset
    :members:
    :no-undoc-members:
    :show-inheritance:


ReanalysisData
----------------------------------------

.. automodule:: operational_analysis.types.reanalysis
    :members:
    :no-undoc-members:
    :show-inheritance:
