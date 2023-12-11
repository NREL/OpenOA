.. getting_started:

Getting Started
###############

Before installing and diving in, users can interact with our examples and test out the analysis
library on our `Binder page`_.

Quick Start
***********

For most users, getting started should be as easy as either below code blocks, but if you get hung
up on running examples locally, or interacting with the code, you can always interact with our
`Binder page`_

Pip
---

.. code-block:: bash

    pip install OpenOA

Source
------

.. code-block:: bash

    git clone https://github.com/NREL/OpenOA.git
    cd openoa
    pip install -e .

Additional Dependencies
-----------------------

.. important::
    If using Python 3.11, install ``openoa`` only, then reinstall adding the modifiers to reduce
    the amount of time it takes for pip to resolve the dependency stack.

Whether installing from PyPI or source, any combination of the following can be used to install
additional dependencies. For example, the examples requirements can be installed using
`pip install "openoa[examples]"`.

- `develop`: for linting, automated formatting, and testing
- `docs`: for building the documentation
- `examples`: for the full Jupyter Lab suite (also contains `reanalysis` and `nrel-wind`)
- `renalysis`: for accessing and processing MERRA2 and ERA5 data
- `nrel-wind`: for accessing the NREL WIND Toolkit
- `all`: for the complete dependency stack

Installation and Contributing
*****************************
.. toctree::
    :maxdepth: 2

    install
    contributing
    changelog


.. _Binder page: https://mybinder.org/v2/gh/NREL/OpenOA/develop_v3?filepath=examples
