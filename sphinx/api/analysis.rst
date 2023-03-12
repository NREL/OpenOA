.. methods:

Analysis Methods
################

Analysis Methods work on :py:attr:`openoa.plant.PlantData` objects to produce high level analyses,
such as the long term AEP. These methods rely on the more generic utils modules, by chaining them
together to create reproducible analysis workflows.

.. autoclass:: openoa.analysis.aep.MonteCarloAEP
    :members:

.. autoclass:: openoa.analysis.turbine_long_term_gross_energy.TurbineLongTermGrossEnergy
    :members:

.. autoclass:: openoa.analysis.electrical_losses.ElectricalLosses
    :members:

.. autoclass:: openoa.analysis.eya_gap_analysis.EYAGapAnalysis
    :members:

.. autoclass:: openoa.analysis.wake_losses.WakeLosses
    :members:
