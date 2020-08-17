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

This paper announces OpenOA version 2.0.0, which was released in August of 2020. Originally released to the public in 2018, OpenOA is implemented in the Python programming language [cite python] and provides a framework and set of convenience functions that engineers, analysts, and researchers in the wind energy industry can use to facilitate big data analysis of operational data sets from wind energy plants. With over 50 stars on github, a dozen contributors, and an active Github issues forum, OpenOA is a mature and active project that provides a high level interface to practical problems in this field.

# Statement of Need

Operational analyses are 

OpenOA was created by and primarily developed by researchers at the National Renewable Energy Laboratory [nrel footnote] through the Performance, Risk Uncertainty Finance (PRUF) project. 

OpenOA has also been used to calculate long-term operational AEP from over 470 wind farms in the US to assess correlation between uncertainty components [Bodini2020].

Reproducible research is an important goal of OpenOA.

OpenOA serves as a conduit to deliver state-of-the-art analysis methods between researchers and practitioners.

We believe the reproducibility and efficiency of scientific research will increase if common, open source tools such as OpenOA are utilized.

[Diagram of OA analysis]

# Summary

OpenOA V2 is released as a Python package and is freely available under a business-friendly, open-source, BSD license. OpenOA contains documentation, worked out examples in Jupyter notebooks, and an example dataset from Engie [cite engie].

The typical user interfaces with OpenOA through its "analysis methods" API. These are Python classes which conform to a common interface (e.g., they implement `__init__`, `prepare`, and `run` methods). Version 2 of OpenOA implements three high level analysis methods. (1) Long term corrected AEP. (2) Electrical losses, and (3) Turbine level losses. Uncertainty quantification is achieved in each analysis using a monte carlo approach. A more detailed description of these analyses are provided in the documentation.

The OpenOA data model is implemented using wrapper classes, called PlantData, that have at least one Pandas data frame [cite pandas]. These classes add convenience functions and a domain-specific schema based on the IEC 6400-25 standard. OpenOA is part of the ENTR alliance consortium, which aims to produce a software ecosystem around an implementation of this standard. It is a vision of the OpenOA project, as well as the ENTR alliance, to provide an implementation of this industry standard. To the author's knowledge, OpenOA offers the first known implementation of this standard published as open source software.

OpenOA depends on scikit-learn [cite sklearn] and numpy [cite numpy], with graphing functions implemented using matplotlib [cite matplotlib]. Low level functions are organized in toolkit modules, which operate on Pandas series objects, and be used outside of the analysis methods. 

[Diagram of OpenOA software architecture]

The OpenOA development team prioriotizes the use of best software development practices. Documentation is built using Sphinx [sphinx citation] and compiled automatically using ReadTheDocs [documentation citation]. We use Github actions to implement our continuous integration pipeline, including automated unit and regression tests, test coverage reporting via codecov, automated packaging and publication to the pypi package index, and the use of a modified git-flow development workflow utilizing pull requests and issue tracking on Github.

The OpenOA development team invites you to collaborate on this project.

# Acknowledgements
This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308.
Funding provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office, within the Atmosphere to Electrons research program.
The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government.
The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

# References
