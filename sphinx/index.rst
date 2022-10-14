.. highlight:: rst


OpenOA Operational Analysis Framework
#####################################

|Binder Badge| |Gitter Badge| |Journal of Open Source Software Badge| |Documentation Badge| |Tests Badge| |Code Coverage Badge| |pre-commit| |Code style: black| |Imports: isort|

.. ::

    # with overline, for parts
    * with overline, for chapters
    =, for sections
    -, for subsections
    ^, for subsubsections
    ", for paragraphs

Overview
********

This library provides a framework for working with timeseries data from wind plants. Its development
has been motivated by the WP3 Benchmarking (PRUF) project, which aims to provide a reference implementaiton for
plant-level performance assessment.

Data processing and ETL is handled by the PlantData class, which in turn relies on Pandas data frames.
PlantData can be used to import, inspect, pre-process, and save the raw data from wind turbines, meters, met towers,
and reanalysis products such as Merra2.

Analysis routines are provided in analysis classes, which each use the PlantData objects to ingest data.

To interact with how each of these components of OpenOA are used, please visit our examples notebooks on
`Binder <https://mybinder.org/v2/gh/NREL/OpenOA/master?filepath=examples>`_, or view them statically on the
`examples page <examplesout.ipynb>`_.

If you use this software in your work, please cite our JOSS article with the following BibTex::

    @article{Perr-Sauer2021,
        doi = {10.21105/joss.02171},
        url = {https://doi.org/10.21105/joss.02171},
        year = {2021},
        publisher = {The Open Journal},
        volume = {6},
        number = {58},
        pages = {2171},
        author = {Jordan Perr-Sauer and Mike Optis and Jason M. Fields and Nicola Bodini and Joseph C.Y. Lee and Austin Todd and Eric Simley and Robert Hammond and Caleb Phillips and Monte Lunacek and Travis Kemper and Lindy Williams and Anna Craig and Nathan Agarwal and Shawn Sheng and John Meissner},
        title = {OpenOA: An Open-Source Codebase For Operational Analysis of Wind Farms},
        journal = {Journal of Open Source Software}
    }

Table of Contents
*****************

.. toctree::
   :maxdepth: 2

   getting_started/index
   examples/index
   api/index
   credit


.. |Binder Badge| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/NREL/OpenOA/main?filepath=examples
.. |Gitter Badge| image:: https://badges.gitter.im/NREL_OpenOA/community.svg
   :target: https://gitter.im/NREL_OpenOA/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
.. |Journal of Open Source Software Badge| image:: https://joss.theoj.org/papers/d635ef3c3784d49f6e81e07a0b35ff6b/status.svg
   :target: https://joss.theoj.org/papers/d635ef3c3784d49f6e81e07a0b35ff6b
.. |Documentation Badge| image:: https://readthedocs.org/projects/openoa/badge/?version=latest
   :target: https://openoa.readthedocs.io
.. |Tests Badge| image:: https://github.com/NREL/OpenOA/workflows/Tests/badge.svg?branch=develop
.. |Code Coverage Badge| image:: https://codecov.io/gh/NREL/OpenOA/branch/develop/graph/badge.svg
   :target: https://codecov.io/gh/NREL/OpenOA
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |Imports: isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/
