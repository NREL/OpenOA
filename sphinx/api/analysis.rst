.. methods:

Analysis Methods
################

Analysis Methods work on :py:attr:`openoa.plant.PlantData` objects to produce high level analyses,
such as the long term AEP. These methods rely on the more generic utils modules, by chaining them
together to create reproducible analysis workflows.

All models use mixin classes to provide three additional methods:

- ``cls.from_dict(data_dictionary)``, which allows the use of creating a class from a dictionary of
  inputs that can be shared across workflows if particular settings work better, or users don't
  wish to use a standard class definition interface.
- ``cls.set_values(run_parameter_dictionary)``, which enables users to set any or all of the
  analysis parameters that are allowed to be manually set post-initialization (see
  ``cls.run_parameters`` for specific parameter listings per class, or the documentation of ``cls.run()``).
- ``cls.reset_defaults(which="""None, a single parameter, or a list of parameters""")``, which allows
  a user to reset all of the analysis parameters back the class defaults.

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

.. autoclass:: openoa.analysis.yaw_misalignment.StaticYawMisalignment
    :members:
