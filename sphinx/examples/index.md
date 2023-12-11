# Examples

All notebooks are located at /examples in the OpenOA repository, and can be modified and run on
[Binder](https://mybinder.org/v2/gh/NREL/OpenOA/develop_v3?filepath=examples).

```{important}
Be sure to install OpenOA using the `examples` modifier (`pip install "openoa[examples]"`). This
will install all of the dependencies used by the additional methods contained within the examples
notebooks.
```

## Overview

In each of the following examples we'll be providing more insight into the different functionalities
of OpenOA. In each notebook, OpenOA is demonstrated using two years of operational data for the La Haute Borne wind power plant from the ENGIE open data set (https://opendata-renewables.engie.com). The examples start by introducing the {py:attr}`openoa.plant.PlantData` class and quality assurance methods in the {py:mod}`openoa.utils.qa` utils module, illustrating how a PlantData object is created for the La Haute Borne data set (the rest of the examples use this PlantData object to demonstrate OpenOA analysis and utils methods). Next, several utils module use cases are demonstrated, such as power curve fitting and plotting. Three analysis methods are then demonstrated: MonteCarloAEP (long-term AEP analysis), TurbineLongTermGrossEnergy (turbine ideal energy), and ElectricalLosses. Next, the EYAGapAnalysis class is used to perform a gap analysis using the estimated operational long-term AEP, turbine ideal energy, electrical losses, and availability losses together with corresponding example pre-construction estimates. Two additional analysis methods are then demonstrated: the WakeLosses method for estimating operational wind plant and wind turbine-level wake losses, and the StaticYawMisalignment method for estimating static yaw misalignment for individual wind turbines as a function of wind speed.

## Intro to OpenOA `PlantData` and the QA Methods [[link]](00_intro_to_plant_data.ipynb)

* Reiterates some of the essential concepts of the of the {py:attr}`openoa.plant.PlantData` and
  {py:attr}`openoa.plant.PlantMetaData` classes
* Shows how we formulated the `examples/project_ENGIE.py` data cleaning and loading
  scripts
* Highlights the QA methods available in {py:mod}`openoa.utils.qa` and how they work on real
  data

## Demonstrating the Utils With the ENGIE Open Data [[link]](01_utils_examples.ipynb)

* Walks through the use of various plotting and analysis methods
* Introduces some of the building blocks of how analyses are composed

## Gap Analysis Step 1a: Estimate the AEP and Its Uncertainty [[link]](02a_plant_aep_analysis.ipynb)

* Introduces the annual energy production (AEP) class, and how to estimate the uncertainty using
  a Monte Carlo approach
* Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
  how to customize them, and how to use them through the {py:mod}`openoa.utils.plot`
  interface

## Gap Analysis Step 1b: Estimate the AEP and Its Uncertainty Using Cubico Open Data [[link]](02b_plant_aep_analysis_cubico.ipynb)

```{important}
Be sure to install OpenOA using the `examples` and `reanalysis` modifiers for this notebook
(`pip install "openoa[examples,reanalysis]"`).
```

* Introduces the annual energy production (AEP) class, and how to estimate the uncertainty using
  a Monte Carlo approach
* Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
  how to customize them, and how to use them through the {py:mod}`openoa.utils.plot`
  interface

## Gap Analysis Step 1c: Alternative Methods for Calculating the AEP [[link]](02c_augmented_plant_aep_analysis.ipynb)

* Building from the previous example, the augmented capabilities for calculating AEP using
  a Monte Carlo framework for calculating the AEP are demonstrated
* Demonstrates how to change the regression model and additional variables that can be considered
  for analyses

## Gap Analysis Step 2: Calculate the Turbine Ideal Energy [[link]](03_turbine_ideal_energy.ipynb)

* Introduces the turbine long term gross energy estimation workflow
* Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
  how to customize them, and how to use them through the {py:mod}`openoa.utils.plot`
  interface

## Gap Analysis Step 3: Estimate Electrical Losses [[link]](04_electrical_losses.ipynb)

* Introduces the electrical losses analysis workflow
* Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
  how to customize them, and how to use them through the {py:mod}`openoa.utils.plot`
  interface

## Gap Analysis Step 4: Compare the Energy Yield Assessment to the Operational Assessment (Gap Analysis) [[link]](05_eya_gap_analysis.ipynb)

* Ties together the previous examples to estimate energy production and potential losses
* Demonstrates some of the supplementary tools to analysis, such as the plotting routines, and
  how to customize them, and how to use them through the {py:mod}`openoa.utils.plot`
  interface

## Estimate Operational Wake Losses [[link]](06_wake_loss_analysis.ipynb)

* Introduces the operational wake loss estimation class and workflow
* Demonstrates the estimation of wake losses based on turbine-level SCADA data during the
  period of record as well as the long-term corrected wake losses incorporating historical
  reanalysis wind resource data
* Illustrates the estimation of wake losses at the wind plant level as well as for each wind
  turbine with and without uncertainty quantification
* Demonstrates methods for plotting wake losses as a function of wind direction and wind speed

## Estimate Yaw Misalignment [[link]](07_static_yaw_misalignment.ipynb)

* Introduces the static yaw misalignment estimation class and workflow
* Demonstrates the estimation of yaw misalignment based on turbine-level SCADA data during
  the period of record
* Illustrates the estimation of wake losses for each wind turbine with and without uncertainty quantification
* Demonstrates methods for plotting yaw misalignment as a function of wind vane angle and normalized power

## Table of Contents

```{toctree}
00_intro_to_plant_data
01_utils_examples
02a_plant_aep_analysis
02b_plant_aep_analysis_cubico
02c_augmented_plant_aep_analysis
03_turbine_ideal_energy
04_electrical_losses
05_eya_gap_analysis
06_wake_loss_analysis
07_static_yaw_misalignment
```
