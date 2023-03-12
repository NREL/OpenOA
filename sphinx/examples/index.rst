.. examples:

Examples
########

All notebooks are located at /examples in the OpenOA repository, and can be modified and run on
`Binder`_.


Overview
********

In each of the following examples we'll be providing more insight into the different functionalities
of OpenOA. In each notebook, OpenOA is demonstrated using two years of operational data for the La Haute Borne wind power plant from the ENGIE open data set (https://opendata-renewables.engie.com). The examples start by introducing the :py:attr:`openoa.plant.PlantData` class and quality assurance methods in the :py:mod:`openoa.utils.qa.py` utils module, illustrating how a PlantData object is created for the La Haute Borne data set (the rest of the examples use this PlantData object to demonstrate OpenOA analysis and utils methods). Next, several utils module use cases are demonstrated, such as power curve fitting and plotting. Three analysis methods are then demonstrated: MonteCarloAEP (long-term AEP analysis), TurbineLongTermGrossEnergy (turbine ideal energy), and ElectricalLosses. Next, the EYAGapAnalysis class is used to perform a gap analysis using the estimated operational long-term AEP, turbine ideal energy, electrical losses, and availability losses together with corresponding example pre-construction estimates. Finally, the WakeLosses method for estimating operational wind plant and wind turbine-level wake losses is demonstrated.


Introduction and QA Example
===========================

* Reiterates some of the essential concepts of the of the :py:attr:`openoa.plant.PlantData` and
  :py:attr:`openoa.plant.PlantMetaData` classes
* Shows how we formulated the :py:mod:`examples/project_ENGIE.py` data cleaning and loading
  scripts
* Highlights the QA methods available in :py:mod:`openoa.utils.qa.py` and how they work on real
  data

Utils Examples
==============

* Walks through the use of various plotting and analysis methods
* Introduces some of the building blocks of how analyses are composed

MonteCarloAEP Example
=====================

* Introduces the annual energy production (AEP) class, and how to estimate the uncertainty using
  a Monte Carlo approach
* Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
  how to customize them, and how to use them through the :py:mod:`openoa.utils.plotting.py`
  interface

Turbine Ideal Energy Example
============================

* Introduces the turbine long term gross energy estimation workflow
* Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
  how to customize them, and how to use them through the :py:mod:`openoa.utils.plotting.py`
  interface

Electrical Losses Example
=========================

* Introduces the electrical losses analysis workflow
* Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
  how to customize them, and how to use them through the :py:mod:`openoa.utils.plotting.py`
  interface

EYA Gap Analysis Example
========================

* Ties together the previous examples to estimate energy production and potential losses
* Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
  how to customize them, and how to use them through the :py:mod:`openoa.utils.plotting.py`
  interface

Wake Loss Analysis Example
==========================

* Introduces the operational wake loss estimation class and workflow
* Demonstrates the estimation of wake losses based on turbine-level SCADA data during the
  period of record as well as the long-term corrected wake losses incorporating historical
  reanalysis wind resource data
* Illustrates the estimation of wake losses at the wind plant level as well as for each wind
  turbine with and without uncertainty quantification
* Demonstrates methods for plotting wake losses as a function of wind direction and wind speed


Table of Contents
*****************

.. toctree::
    :maxdepth: 3

    examplesout


.. _Binder: https://mybinder.org/v2/gh/NREL/OpenOA/develop_v3?filepath=examples
