.. _install:


.. ::

    # with overline, for parts
    * with overline, for chapters
    =, for sections
    -, for subsections
    ^, for subsubsections
    ", for paragraphs

Install
#######

Pip
***

OpenOA is listed on PyPI with the same name, and can be installed into your environment with
the following.

.. code-block:: bash

    pip install OpenOA


Alternatively, if you want to install OpenOA and work with any examples, you can install the extras
at the same time with the following.

.. code-block:: bash

    pip install "OpenOA[develop]"

.. important::
    If using Python 3.11, install ``openoa`` only, then reinstall adding the modifiers to reduce
    the amount of time it takes for pip to resolve the dependency stack.

Additional options:
- `develop`: for linting, automated formatting, and testing
- `docs`: for building the documentation
- `examples`: for the full Jupyter Lab suite (also contains `reanalysis` and `nrel-wind`)
- `renalysis`: for accessing and processing MERRA2 and ERA5 data
- `nrel-wind`: for accessing the NREL WIND Toolkit
- `all`: for the complete dependency stack


Now you can verify the version that was installed

.. code-block:: python

    import openoa
    openoa.__version__


From Source
***********

For any development for your own workflows or work that will be contributed back to the library,
OpenOA should be installed from the source on GitHub. For more information on contributing
guidelines and processes, please see the :ref:`contributors guide <contributing>`.

.. code-block:: bash

    git clone https://github.com/NREL/OpenOA.git
    cd openoa
    pip install -e .

    # Extras can also be installed as one or any combination of the following
    pip install -e ".[develop,docs]"

Now you can verify the version that was installed

.. code-block:: python

    import openoa
    openoa.__version__


Common Installation Issues
**************************

- In Windows you may get an error regarding geos_c.dll. To fix this install Shapely using:

.. code-block:: bash

    conda install Shapely


- In Windows, an ImportError regarding win32api can also occur. This can be resolved by fixing the version of pywin32 as follows:

.. code-block:: bash

    pip install --upgrade pywin32==255
