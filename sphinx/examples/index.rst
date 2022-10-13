.. examples:

Examples
########

All notebooks are located at /examples in the OpenOA repository, and can be modified and run on
`Binder`_.

In each of the following examples we'll be providing more insight into the different functionalities
of OpenOA.

.. * :ref:`00: Intro to PlantData <examplesout#Intro-to-the-OpenOA-PlantData-and-QA-Methods>`
* Introduction and QA Example

  * Reiterates some of the essential concepts of the of the :py:attr:`openoa.plant.PlantData` and
    :py:attr:`openoa.plant.PlantMetaData` classes
  * Shows how we formulated the :py:mod:`examples/project_ENGIE.py` data cleaning and loading
    scripts
  * Highlights the QA methods available in :py:mod:`openoa.utils.qa.py` and how they work on real
    data

* Utils Examples

  * Walks through the use of various plotting and analysis methods
  * Introduces some of the building blocks of how analyses are composed

* MonteCarloAEP Example

  * Introduces the annual energy production (AEP) class, and how to estimate the uncertainty using
    a Monte Carlo approach
  * Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
    how to customize them, and how to use them through the :py:mod:`openoa.utils.plotting.py`
    interface

* Turbine Ideal Energy Example

  * Introduces the turbine long term gross energy estimation workflow
  * Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
    how to customize them, and how to use them through the :py:mod:`openoa.utils.plotting.py`
    interface

* Electrical Losses Example

  * Introduces the electrical losses analysis workflow
  * Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
    how to customize them, and how to use them through the :py:mod:`openoa.utils.plotting.py`
    interface

* EYA Gap Analysis Example

  * Ties together the previous examples to estimate energy production and potential losses
  * Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
    how to customize them, and how to use them through the :py:mod:`openoa.utils.plotting.py`
    interface



.. toctree::
    :maxdepth: 1

    examplesout


.. _Binder: https://mybinder.org/v2/gh/NREL/OpenOA/main?filepath=examples
