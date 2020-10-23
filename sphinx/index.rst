.. highlight:: rst


OpenOA Operational Analysis Framework
#####################################

.. ::

    # with overline, for parts
    * with overline, for chapters
    =, for sections
    -, for subsections
    ^, for subsubsections
    ", for paragraphs

This library provides a generic framework for working with large timeseries data from wind plants. Its development
has been motivated by the WP3 Benchmarking (PRUF) project, which aims to provide a reference implementaiton for
plant-level performance assessment.

The implementation makes use of a flexible backend, so that data loading, processing, and analysis can be performed
locally (e.g., with Pandas dataframes), in a semi-distributed manner (e.g., with Dask dataframes), or in a fully
distributed matter (e.g., with Spark dataframes).

Data processing and ETL is handled by the PlantData class and by project-specific modules which implement subclasses.
These modules can be used to import, inspect, pre-process, and save the raw data from wind turbines, meters, met towers,
and reanalysis products such as Merra2.

Analysis routines are grouped by purpose into toolkits - which provide an abstract low level API for common
computations, and methods - which provide higher level wind industry specific API. In addition to these provided modules,
anyone can write their own, which is intended to provide natural growth of tools within this framework.

To interact with how each of these components of OpenOA are used, please visit our examples notebooks on
`Binder <https://mybinder.org/v2/gh/NREL/OpenOA/master?filepath=examples>`_, or view them statically on the
`examples page <examplesout.ipynb>`_.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    install.rst
    examplesout
    toolkits.rst
    methods.rst
    types.rst
    contributing.rst
    credit.rst

