.. methods:

Analysis Methods
################

Analysis Methods work on :py:attr:`openoa.plant.PlantData` objects to produce high level analyses,
such as the long term AEP. These methods rely on the more generic utils modules, by chaining them
together to create reproducible analysis workflows.

Plant Level Analysis
********************

.. autoclass:: openoa.analysis.aep.MonteCarloAEP
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: openoa.analysis.eya_gap_analysis.EYAGapAnalysis
    :members:
    :undoc-members:
    :show-inheritance:

Turbine Level Analysis
**********************

.. autoclass:: openoa.analysis.turbine_long_term_gross_energy.TurbineLongTermGrossEnergy
    :members:
    :undoc-members:
    :show-inheritance:

Electrical Losses Analysis
**************************

.. autoclass:: openoa.analysis.electrical_losses.ElectricalLosses
    :members:
    :undoc-members:
    :show-inheritance:
