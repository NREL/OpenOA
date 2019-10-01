---
title: 'OpenOA: An Open-Source Codebase For Operational Analysis of Wind Farms'
tags:
  - Python
  - wind energy
  - operational analysis
  - data analysis
authors:
  - name: Jordan Perr-Sauer
    orcid: 0000-0003-0872-7098
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Mike Optis
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: National Renewable Energy Laboratory, Golden, CO, USA
   index: 1
date: 26 June 2019
bibliography: paper.bib
---

# Summary

Operational analyses use collected data from wind power plants to perform assessments ranging from the diagnosis of faults and underperformance, benchmarking of performance improvements (e.g., wind sector management, vortex generators), long-term estimates of annual energy production (AEP), and building/tuning statistical or physics-based models for various applications (e.g.,, wake model validation, wind power forecasting).

The development of OpenOA started internally at the National Renewable Energy Laboratory (NREL) and was initiated to support the lab's efforts in the Wind Plant Performance and Prediction (WP3) Benchmark, which is a key risk reduction activity of the Performance, Risk, Uncertainty, and Finance project under the Atmosphere to Electrons initiative [@A2EWebsite].
Further funding was provided through the NREL Research Data Initiative.
The goal of WP3 is to provide an independent benchmark of preconstruction EYA bias, and to understand the sources of uncertainty.
OpenOA was originally scoped to calculate the operational AEP of case study wind power plants, to which EYA estimates of those plants provided by consultants could be compared.
OpenOA is now expected to rapidly expand its scope to support additional types of analyses including turbine performance and reliability.

Released publicly in September 2018, OpenOA provides an open-source framework for the operational analysis of wind power plant data.
The intent of this effort is to begin fostering collaboration and methods sharing in a wind industry that has historically been very protective of methods and data.
Much of an OA analysis could be standard and uncontroversial, and by creating a public repository for the collection and dissemination of OA methods and best practices, significant efficiency gains can be achieved.
Furthermore, over time, we hope that OpenOA will become a reference implementation for OA methods from which a published standard (IEC or otherwise) may quickly follow.
To our knowledge, this approach of first fostering a collaborative repository of methods that are tested and used prior to developing a published standard is new to the wind industry, and may indeed prove more efficient than the multiyear approach of beginning immediately with a standard.

# Development of a Functional Wind Plant Data Standard
There have been efforts to standardize wind plant data collection, such as IEC-25.
We base our implementation off the standard, make a reference implementation.

There are four classes within the Types module.
The core of this module, as described in Section \ref{subsec:modular}, is the Plant Data class.
In addition to housing all the wind plant data attributes, this class also includes useful functions for importing raw plant data into a Plant Data object, checks to ensure Plant Data conform to expected schema (e.g.,, column naming conventions), and loading/saving Plant Data to file using flexible file formatsWe believe this functionality will become invaluable to our users, and represents our first step toward an industrywide data exchange format.

The remaining classes are used largely in support of the Plant Data class.
The Timeseries Table class provides a data structure in which the underlying data frame back end can be Pandas or Spark.
This flexibility allows for OpenOA to handle both smaller data sets (i.e.,, Pandas) or very large data sets requiring distributed and parallel processing (i.e.,, Spark).
The Reanalysis Data module allows storage of multiple reanalysis data products as Timeseries Tables within the same class.
The Asset Data module contains a GeoPandas data frame that contains information about the turbines and met towers at the wind plant (e.g., location, rated turbine capacity).

# AEP Calculation Methods
Methods modules are high-level analyses that are generally implemented as a class and compute metrics of scientific interest while relying on the lower-level toolkit functions for their implementation.
Version 1.0 of OpenOA includes one method: the calculation of wind plant AEP using operational data.

The AEP analysis is based on an industry-standard approach in which monthly gross energy for the wind plant (reported energy at the revenue meter corrected for availability and curtailment losses) is related to a monthly long-term wind resource through a linear regression relationship.
Calculation of AEP involves several steps:

- Processing of the revenue meter energy, loss estimates, and long-term reanalysis wind resource data.
- Review of different reanalysis data for suitability.
- Linear regression outlier detection and removal.
- Flagging and removal of high-energy loss months.
- Application of regression relationship of energy and wind resource to the long-term resource to calculate long-term gross energy.
- Estimation of long-term AEP from long-term gross energy and expected future losses.
- Uncertainty quantification through a Monte Carlo approach in which inputs to and intermediate calculations within the process are sampled based on their assumed or calculated uncertainties.

An example usage of this method is shown in Figure \ref{fig:aep_calc}.
Here, revenue meter and reanalysis data attributes from Plant Data are used with several toolkit modules to calculate operational AEP for a wind plant.
The details of this particular example are provided in a Jupyter notebook on the GitHub repository.
\begin{figure}[h]
\begin{center}
\includegraphics[width=15cm]{figures/aep_analysis_v3.png}
\caption{Using different OpenOA modules to calculate wind plant AEP using operational data.
In this example, revenue meter and reanalysis data are processed using several toolkit modules.}
\label{fig:aep_calc}
\end{center}
\end{figure}

# Lower Level Toolkit Functions
There are currently seven different OpenOA toolkits, which are listed in Table \ref{tab:toolkits} along with a general description of their functions.
Toolkit modules range from those used for general data processing (flagging, imputation, unit conversion, and time series modules) and those specifically intended for wind plant data processing (meteorological data processing, power curve fitting, and plotting).

- Filters: Functions for flagging data based on a range of criteria.
- Imputing: Functions for filling in null data with interpolated (imputed) values.
- Meteorological data: Functions for calculating common meteorological variables used in wind resource analysis.
- Time series: Functions for common time series analysis, including missing time-stamp identification and gap filling.
- Unit conversion: Functions for common unit conversions in wind energy (e.g.,, power to energy).
- Power curve: Functions to fit data to a specified wind turbine power curve model (including parametric and nonparametric forms) and to then apply the power curve to wind speed data.
- Plotting tools: Functions to produce common wind resource and energy-related visualizations (e.g.,, wind rose).

An example of toolkit use is shown in Figure \ref{fig:pc_calc}.
Here, several power curve models are fit to filtered wind speed and power data for a specific turbine.
As shown in the figure, data from the Plant Data SCADA attribute and several toolkit modules are used to perform the flagging and removal of outlier data, the fitting of the power curve, and the plotting of results.
The steps of this particular example are provided in detail as a Jupyter notebook\footnote{https://jupyter.org/} on the GitHub repository.

The value of toolkit modules lies in their generality.
Each function was written to operate on array-like objects, such as Pandas Series, Data Frames, and NumPy Arrays.
In this way, the toolkit modules can be applied in a variety of situations that are both internal and external to the OpenOA code base.

# Industry and Community Participation

# Acknowledgements
This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308.
Funding provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office, within the Atmosphere to Electrons research program.
The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government.
The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

# References
