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

PlantData.reanalysis.product["merra2"]

==================== ====================
 Field Name           Data Type
==================== ====================
 time                 datetime64[ns]
 windspeed_ms         float64
 winddirection_deg    float64
 rho_kgm-3            float64
==================== ====================

PlantData.reanalysis.product["ncep2"]

==================== ====================
 Field Name           Data Type
==================== ====================
 time                 datetime64[ns]
 windspeed_ms         float64
 winddirection_deg    float64
 rho_kgm-3            float64
==================== ====================


PlantData.reanalysis.product["erai"]

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
