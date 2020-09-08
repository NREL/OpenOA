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
  - name: Nicola Bodini
    orcid: 0000-0002-2550-9853
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
  - name: Robert Hammond
    orcid: 0000-0003-4476-6406
    affiliation: 1
  - name: Nathan Agarwal
    orcid: 0000-0002-2734-5514
    affiliation: 1
  - name: Shawn Sheng
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

<!--
JOSS welcomes submissions from broadly diverse research areas. For this reason, we require that authors include in the paper some sentences that explain the software functionality and domain of use to a non-specialist reader. We also require that authors explain the research applications of the software. The paper should be between 250-1000 words.

Your paper should include:

A list of the authors of the software and their affiliations, using the correct format (see the example below).
A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.
A clear Statement of Need that illustrates the research purpose of the software.
A list of key references, including to other software addressing related needs.
Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.
Acknowledgement of any financial support.
As this short list shows, JOSS papers are only expected to contain a limited set of metadata (see example below), a Statement of Need, Summary, Acknowledgements, and References sections. You can look at an example accepted paper. Given this format, a “full length” paper is not permitted, and software documentation such as API (Application Programming Interface) functionality should not be in the paper and instead should be outlined in the software documentation.


Review Checklist:
Summary: Has a clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided?
A statement of need: Do the authors clearly state what problems the software is designed to solve and who the target audience is?
State of the field: Do the authors describe how this software compares to other commonly-used packages?
Quality of writing: Is the paper well written (i.e., it does not require editing for structure, language, or writing quality)?
References: Is the list of references complete, and is everything cited appropriately that should be cited (e.g., papers, datasets, software)? Do references in the text use the proper citation syntax?

-->

This paper announces OpenOA version 2.0.0, which was released in August of 2020. Originally released to the public in 2018 [@osti_1478526], OpenOA is an open source framework for operational data analysis of wind energy plants, implemented in the Python programming language [@Rossum2009]. OpenOA provides a common data model, high level analysis workflows, and low-level convenience functions that engineers, analysts, and researchers in the wind energy industry can use to facilitate anlytics workflows on operational data sets. With over 50 stars on Github, a dozen contributors, and an active issues forum, OpenOA is becoming a mature project that provides a high level interface to solve practical problems in the wind energy industry.

# Statement of Need

OpenOA was created by and primarily developed by researchers at the National Renewable Energy Laboratory [^nrelwebsite] (NREL) through the Performance, Risk Uncertainty Finance [^wp3website] (PRUF) project, when the team recognized the need to compute a 20-year long term correced AEP metric from operational data as part of a benchmarking study [WP3 benchmark citation]. Due to restrictions on the input SCADA data, the team recognized that open source publication of the code was necessary to foster trust in the results by the participants of the benchmarking study. Furthermore, after talking with our industry partners, it became clear that there was no industry standard code for computing a 20-year long term corrected AEP.

[^nrelwebsite]: \url{https://nrel.gov}
[^wp3website]: \url{https://a2e.energy.gov/projects/wp3}

Operational analysis involves obtaining time-series data from an industrial plant's SCADA system, performing ETL and QC processes on these data, and then computing various metrics that might inform decisions by the plant operator. Since its inception, OpenOA has been used in several published studies at NREL. An early version of the code was used in [@Craig2018] to quantify the uncertainty in analyst choices. In one study, it is used to calculate long-term operational AEP from over 470 wind farms in the US to assess correlation between uncertainty components [@Bodini2020]. OpenOA is used in an upcoming paper analysing the gap between preconstruction estimates of energy production, called the P50 bias [cite gap analysis].

By forming an open source software project, OpenOA hopes to improve the reproducibility of research in this field, provide benchmark implementations of commonly performed data transformation and analysis tasks (to lower the barrier to entry), and finally to serve as a conduit that can deliver state-of-the-art analysis methods from researchers to practitioners.

# Summary

OpenOA V2 is released as a Python package and is freely available under a business-friendly, open-source, BSD license. OpenOA contains documentation, worked out examples in Jupyter notebooks, and a corresponding example dataset from Engie Renewable's La Haute Borne Dataset [@EngieDataset].

The typical user interfaces with OpenOA through its "analysis methods" API. These are Python classes which conform to a common interface (e.g., they implement `__init__`, `prepare`, and `run` methods). Version 2 of OpenOA implements three high level analysis methods. (1) Long term corrected AEP. (2) Electrical losses, and (3) Turbine level losses. Uncertainty quantification is achieved in each analysis using a monte carlo approach. A more detailed description of these analyses are provided in the documentation.

The OpenOA data model is implemented using wrapper classes, called PlantData, that have at least one Pandas data frame [@Mckinney2010]. These classes add convenience functions and a domain-specific schema based on the IEC 6400-25 standard. OpenOA is part of the ENTR alliance consortium, which envisions a complete software stack centered around an open source implementation of this standard. To the author's knowledge, OpenOA offers the first known implementation of this standard published as open source software.

OpenOA depends on scikit-learn [@Pedregosa2011] and numpy [@oliphant2006guide], with graphing functions implemented using matplotlib [@hunter2007matplotlib]. Low level functions are organized in toolkit modules, which operate on Pandas series objects, and are general enough to use across multiple domains. The OpenOA development team prioriotizes the use of best software development practices. Documentation is compiled from the source code and automatically published to ReadTheDocs [^rtdwebsite]. We use Github actions to implement our continuous integration pipeline, including automated unit and regression tests, test coverage reporting via CodeCov, automated packaging and publication to the Pypi package index. We utilize a modified git-flow development workflow, with pull requests and issue tracking on Github driving the development.

[^nrelwebsite]: \url{https://openoa.readthedocs.io}

In conclusion, the OpenOA development team is excited about the future of open science in the wind energy industry. We invite all interested readers to contribute to this project through Github.

# Acknowledgements
The authors would like to acknowledge that Jordan Perr-Sauer and Mike Optis have made an equal contribution to this work.
This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308.
Funding provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office, within the Atmosphere to Electrons research program.
The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government.
The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

# References
