.. _install:


.. ::

    # with overline, for parts
    * with overline, for chapters
    =, for sections
    -, for subsections
    ^, for subsubsections
    ", for paragraphs

Install
*******

Requirements
============

  * Python 2.7.13 (e.g., from Anaconda) with pip

We recommend creating a new virtual environment or Anaconda environment before attempting to install
OpenOA. To create and activate such a new environment with the name "openoa-env" using Anaconda:

.. code::

    conda create --name openoa-env python=2.7
    conda activate openoa-env


Microsoft Windows
^^^^^^^^^^^^^^^^^

For users Microsoft Windows, the Anaconda python distribution is required. The reason is that pip on windows requires
Visual Studio libraries to compile some of the dependencies. This can be resolved by manually installing the following
packages via conda, which installs pre-built binaries of these dependencies, before attempting a pip install of OpenOA.

.. code::

    conda install shapely
    conda install geos
    conda install fiona


If errors about Visual Studio persist, you can try downloading the Microsoft Visual Studio compiler for Python: https://www.microsoft.com/en-us/download/details.aspx?id=44266


Installlation
=============

Clone the repository and install the library and its dependencies using pip:

.. code::

    git clone git@github.com:NREL/OpenOA.git
    pip install ./OpenOA

You should now be able to import operational_analysis from the Python interpreter:

.. code::

    python
    import operational_analysis


Testing
=======

All tests are runnable from setuptools. They are written in the Python unittest framework.

To run unit tests with code coverage reporting:

.. code::

    cd ./OpenOA
    python setup.py test

To run integration tests (longer running, requires data):

.. code::

    python setup.py integrate

To output junit xml from integration test (used for Jenkins testing):

.. code::

    python setup.py integrate -a "--junitxml=./path_to_outputfile.xml"


Development
===========

We provide a frozen environment in a requirements.txt file which can be used to install the precise versions
of each dependency present in our own development environment. We recommend utilizing a fresh virtual environment or
Anaconda root before installing these requirements. To use requirements.txt:

.. code::

    pip install -r ./OpenOA/requirements.txt

Next, we recommend installing OpenOA in editable mode:

.. code::

    pip install -e ./OpenOA
