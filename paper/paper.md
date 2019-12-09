---
title: 'OpenOA: An Open-Source Codebase For Operational Analysis of Wind Farms'
tags:
  - Python
  - wind energy
  - operational analysis
  - data analysis
  - standardization
authors:
  - name: Mike Optis
    affiliation: 1
  - name: Jordan Perr-Sauer
    orcid: 0000-0003-0872-7098
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Caleb Phillips
    affiliation: 1
  - name: Anna E. Craig
    affiliation: 1
  - name: Travis Kemper
    affiliation: 1
  - name: Joseph C.Y. Lee
    orcid: 0000-0003-1897-6290
    affiliation: 1
  - name: Monte Lunacek
    affiliation: 1
  - name: Shuangwen Shang
    affiliation: 1
  - name: Lindy Williams
    affiliation: 1
  - name: Eric Simley
    affiliation: 1
  - name: John Meissner
    affiliation: 1
  - name: Jason M. Fields
    affiliation: 1
affiliations:
 - name: National Renewable Energy Laboratory, Golden, CO, USA
   index: 1
date: 20 December 2019
bibliography: paper.bib
---

# Summary

OpenOA is an open source Python package which implements operational analysis (OA) methods for wind plants.
The development of OpenOA started internally at the National Renewable Energy Laboratory (NREL) to support the lab's
efforts in the Wind Plant Performance and Prediction (WP3) benchmarking initiative, which is a key risk reduction
activity of the Performance, Risk, Uncertainty, and Finance project under the Atmosphere to Electrons initiative [@A2EWebsite].
The goal of WP3 is to provide an independent benchmark of bias in pre-construction energy yield assessment (EYA),
and to understand the sources of uncertainty therein.

OpenOA was originally scoped to calculate the operational annual energy production (AEP) of case study wind power
plants, providing a baseline to which bias in EYA can be measured.
OpenOA has since expand its scope to support additional types of analyses including turbine performance and reliability
and to support data from wind plants outside the initial set of test projects.
Released publicly in September 2018, the OpenOA repository contains numerous examples worked out in Jupyter notebooks,
along with publicly available data which can be used to run the built in unit and integration tests.

# Operational Analysis
Operational analyses consume data from various sources, including from supervisory control and data acquisition (SCADA)
systems and meteorological reanalysis products such as weather models.
These data are used to perform a wide variety of assessments ranging from the long-term estimates of AEP, diagnosis of
faults and underperformance, benchmarking of performance improvements (e.g., wind sector management, vortex generators),
and building/tuning statistical or physics-based models for various applications (e.g., wake model validation, wind power forecasting).

## Long Term AEP Calculation

The AEP analysis implemented in OpenOA is based on a relatively standard approach within the wind resource assessment
industry, where monthly gross energy for the wind plant (reported energy at the revenue meter corrected for availability
and curtailment losses) is related to a monthly long-term wind resource through a linear regression relationship.
Calculation of AEP involves several steps:

1. Processing of the revenue meter energy, loss estimates, and long-term reanalysis wind resource data.
2. Review of different reanalysis data for suitability.
3. Outlier detection and removal.
4. Flagging and removal of months with high energy losses.
5. Application of linear regression between energy and wind resource and the long-term resource to calculate long-term
gross energy.
6. Estimation of long-term AEP from long-term gross energy and expected future losses.
7. Uncertainty quantification through a Monte Carlo approach in which inputs to and intermediate calculations within
the process are sampled based on their assumed or calculated uncertainties.

An example usage of this method is shown in Figure 1.
Here, revenue meter and reanalysis data attributes from plant data are used with several toolkit modules to calculate
operational AEP for a wind plant.
The details of this particular example are provided in a Jupyter notebook on the GitHub repository.

![](aep_analysis_v3.png)
*Figure 1: Using different OpenOA Toolkits to calculate wind plant AEP using operational data.
In this example, revenue meter and reanalysis data are processed using the data flagging, time series analysis,
and plotting tools module to produce the graphs in the figure.*

## Low Level Toolkits
As of October 2019, there are seven low level modules in OpenOA called Toolkits, which are listed here along with a
general description of their functions.
These modules range from general data processing (flagging, imputation, unit conversion, and time series modules)
and those specifically intended for wind plant data processing (meteorological data processing, power curve fitting, and plotting).
The value of toolkit modules lies in their generality.
Each function was written to operate on array-like objects, such as Pandas Series, Data Frames, and NumPy Arrays.
In this way, the toolkit modules can be applied in a variety of situations, both internal and external to the OpenOA code base.

- Filters: Functions for flagging data based on a range of criteria, such as: Values out of range, frozen signals,
and criteria based on the variance.
- Imputing: Functions for filling in null data with interpolated (imputed) values.
- Meteorological data: Functions for calculating common meteorological variables used in wind resource analysis,
such as: Air density correction, vertical pressure extrapolation, and computation of sheer and veer.
- Time series: Functions for common time series analysis, such as: Missing time-stamp identification, and gap filling.
- Unit conversion: Functions for common unit conversions in wind energy, such as power to energy conversion.
- Power curve: Functions to fit data to a specified wind turbine power curve model
(including parametric and nonparametric forms) and to then apply the power curve to wind speed data.
- Plotting tools: Functions to produce common wind resource and energy-related visualizations (e.g., the wind rose).

An example of toolkit use is shown in Figure 2.
Here, several power curve models are fit to filtered wind speed and power data for a specific turbine.
As shown in the figure, SCADA data can be processed by several toolkit modules to perform the flagging and
removal of outlier data, the fitting of the power curve, and the plotting of results.
The steps of this particular example are provided in detail as a Jupyter notebook on the GitHub repository.

![](pc_analysis_v2.png)
*Figure 2: Using different OpenOA modules to calculate idealized power curves for a sample wind turbine. In this example,
raw SCADA data are filtered for outlier data (shown in red) using a bin filter from the data flagging toolkit.
Three power curves are computed using the power curve toolkit and are plotted against the filtered data in the bottom
right figure.*

# Towards Industry Standards
OpenOA implements a data standard for wind plant SCADA data based on the International Electrotechnical Commission (IEC) standard 61400-25.
The standard defines naming conventions and data types for variables which are encountered in wind plant data.
OpenOA aims to boost this adoption by providing an internal data model based upon the 61400-25 standard.
Users are required to build a mapping of their data source to OpenOA's data model by extending the built in "PlantData" class.

For operational assessments, there are only limited standards covering specific applications: IEC 61400-12 IEC
61400-12-1:2017 addresses turbine power curve testing and IEC 61400-26 IEC 61400-26- 3:2016 addresses the derivation
and categorization of availability loss metrics.
Notably lacking standards are AEP estimates, reliability and performance metrics, and fault and underperformance diagnosis.
In fact, very little documentation of OA best practices exists beyond these standards, and seems to be limited
to a consultant report [@lindvall2016], an academic thesis [@khatab2017], and some conference proceedings [@lunacek2018].

Moving forward, we see a role for OpenOA in fostering standards development for the methods of wind plant OA.
We believe significant efficiency gains can be achieved by providing a public repository for the collection and
dissemination of OA methods and best practices.
Furthermore, we believe that a standard operational analysis will benefit those who rely upon these analysis for
investment decisions, as a standardized analysis will produce more consistent results.

# Fostering Collaboration
The intent of OpenOA is to foster collaboration and methods sharing in a wind industry that has historically been very
protective of methods and data.
Much of an OA analysis could be standard and uncontroversial, and by creating a public repository for the collection and
dissemination of OA methods and best practices, significant efficiency gains can be achieved.
OpenOA currently implements a long-term corrected AEP analysis, and provides low level functions which are helpful in this computation.
Over time, we hope that OpenOA will become a reference implementation for OA methods from which a published standard
(IEC or otherwise) may quickly follow.
To our knowledge, this approach of fostering a collaborative repository of methods that are tested and used prior to
developing a published standard is new to the wind industry, and may indeed prove more efficient than the multi-year
approach of writing a standard from scratch.

# Acknowledgements
This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308.
Funding provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office, within the Atmosphere to Electrons research program.
The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government.
The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

# References
