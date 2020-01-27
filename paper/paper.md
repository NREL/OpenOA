---
title: 'OpenOA: An Open-Source Codebase For Operational Analysis of Wind Farms'
tags:
  - Python
  - wind energy
  - operational analysis
  - data analysis
  - standardization
authors:
  - name: Jordan Perr-Sauer
    orcid: 0000-0003-0872-7098
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Mike Optis
    orcid: 0000-0001-5617-6134
    affiliation: 1
  - name: Jason M. Fields
    affiliation: 1
  - name: Lindy Williams
    affiliation: 1
  - name: Joseph C.Y. Lee
    orcid: 0000-0003-1897-6290
    affiliation: 1
  - name: Austin Todd
    affiliation: 1
  - name: Caleb Phillips
    affiliation: 1
  - name: Monte Lunacek
    affiliation: 1
  - name: Anna Craig
    affiliation: 1
  - name: Travis Kemper
    affiliation: 1
  - name: Shuangwen Sheng
    affiliation: 1
  - name: Eric Simley
    affiliation: 1
  - name: John Meissner
    affiliation: 1
affiliations:
 - name: National Renewable Energy Laboratory, Golden, CO, USA
   index: 1
date: 20 December 2019
bibliography: paper.bib
---

# Summary

OpenOA is an open source Python package which implements operational analysis (OA) methods for wind energy plants.
The goal of OpenOA is collaboration and methods sharing in a wind industry that has historically been very protective of methods and data [@McCann2018].
Over time, we hope that OpenOA will become a reference implementation for OA methods.

The development of OpenOA started internally at the National Renewable Energy Laboratory (NREL) to support the lab's
efforts in the Wind Plant Performance and Prediction (WP3) benchmarking initiative [^wp3website], which is a key risk reduction
activity of the Performance, Risk, Uncertainty, and Finance project under the Atmosphere to Electrons initiative.
The goal of WP3 is to provide an independent benchmark of bias in pre-construction energy yield assessment (EYA),
and to understand the sources of uncertainty therein

[^wp3website]: \url{https://a2e.energy.gov/projects/wp3}

OpenOA was originally scoped to calculate the operational annual energy production (AEP) of case study wind power plants, providing a baseline to which EYA bias can be measured.
OpenOA has since expanded its scope to support additional types of analyses including turbine performance and electrical losses and to support data from wind plants outside the initial set of test projects.
Released publicly in September 2018, the OpenOA repository contains numerous examples demonstrated in Jupyter notebooks along with publicly available data which can be used to run the built in unit and integration tests. The software has also been used in study on the impact of analyst choice in the outcome of OAs [@craig2018].

# Importance of Operational Analysis
Operational analysis uses collected data from wind farms to perform assessments ranging from the diagnosis of
faults and underperformance, benchmarking of performance improvements (e.g., wind sector management,
vortex generators), long-term estimates of AEP, and building
statistical or physics-based models for various applications (e.g.,, wake model validation, wind power
forecasting). Data sources include the wind farm revenue meter, turbine supervisory, control and data
acquisition (SCADA) systems, on-site meteorological towers, and modeled atmospheric data (e.g., reanalysis data).

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
the process are sampled based on their assumed or calculated error distributions.

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
OpenOA provides an internal data model which is based on a data standard for wind plant SCADA data from the
International Electrotechnical Commission (IEC) standard 61400-25 [@iec25].
The standard defines naming conventions and data types for variables which are encountered in wind plant data.
OpenOA aims to boost this adoption by providing an internal data model based upon the 61400-25 standard.
Users are required to build a mapping of their data source to OpenOA's data model by extending the built in "PlantData" class.

For operational assessments, there are only limited standards covering specific applications: IEC
61400-12 [@iec12] addresses turbine power curve testing and IEC 61400-26 [@iec26] addresses the derivation
and categorization of availability loss metrics.
Notably lacking standards are AEP estimates, reliability and performance metrics, and fault and underperformance diagnosis.
Little documentation of OA best practices exists beyond these standards, to our knowledge. We are aware of a consultant report [@lindvall2016], an academic thesis [@khatab2017], and some conference proceedings [@lunacek2018].

Moving forward, we see a role for OpenOA in fostering reference methods and standards development for the methods of wind plant OA.
We believe significant efficiency gains can be achieved by providing a public repository for the collection and
dissemination of OA methods and best practices.
Furthermore, we believe that a standard operational analysis will benefit those who rely upon these analysis for
daily operations management through to investment decisions. A standardized analysis will produce more consistent results across industry and yield better faster decision making.

# Acknowledgements
This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308.
Funding provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office, within the Atmosphere to Electrons research program.
The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government.
The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

# References
