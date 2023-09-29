.. types:

PlantData and the Meta Data APIs
################################

PlantData API
*************

The ``PlantData`` object is the primary engine for OpenOA's data storage, validation, and analysis.

.. autoclass:: openoa.plant.PlantData
    :members:
    :no-undoc-members:
    :show-inheritance:


PlantMetaData API
*****************

Without the metadata schema provided in each data category's metadata class, and compiled through
``PlantMetaData`` the data standardization provided by ``PlantData`` would not be possible.

.. autoclass:: openoa.schema.PlantMetaData
    :members:
    :no-undoc-members:
    :show-inheritance:

.. autoclass:: openoa.schema.SCADAMetaData
    :members:
    :no-undoc-members:
    :show-inheritance:

.. autoclass:: openoa.schema.MeterMetaData
    :members:
    :no-undoc-members:
    :show-inheritance:

.. autoclass:: openoa.schema.TowerMetaData
    :members:
    :no-undoc-members:
    :show-inheritance:

.. autoclass:: openoa.schema.CurtailMetaData
    :members:
    :no-undoc-members:
    :show-inheritance:

.. autoclass:: openoa.schema.StatusMetaData
    :members:
    :no-undoc-members:
    :show-inheritance:

.. autoclass:: openoa.schema.AssetMetaData
    :members:
    :no-undoc-members:
    :show-inheritance:

.. autoclass:: openoa.schema.ReanalysisMetaData
    :members:
    :no-undoc-members:
    :show-inheritance:
