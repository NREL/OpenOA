OpenOA 
======

[![Build Status](https://travis-ci.org/NREL/OpenOA.svg?branch=master)](https://travis-ci.org/NREL/OpenOA) (Master), [![Build Status](https://travis-ci.org/NREL/OpenOA.svg?branch=develop)](https://travis-ci.org/NREL/OpenOA) (Develop)

[![Documentation Status](https://readthedocs.org/projects/openoa/badge/?version=latest)](https://openoa.readthedocs.io/en/latest/?badge=latest) (Develop)

This library provides a generic framework for working with large timeseries data from wind plants. Its development
has been motivated by the WP3 Benchmarking (PRUF) project, which aims to provide a reference implementation for
plant-level performance assessment.

The implementation makes use of a flexible backend, so that data loading, processing, and analysis can be performed
locally (e.g., with Pandas DataFrames), in a semi-distributed manner (e.g., with Dask DataFrames), or in a fully
distributed matter (e.g., with Spark DataFrames).

Analysis routines are grouped by purpose into methods, and these methods in turn rely on more abstract toolkits.
In addition to the provided analysis methods, anyone can write their own, which is intended to provide natural
growth of tools within this framework.

### Requirements

  * Python 2.7+, 3.6+ (e.g., from Anaconda) with pip

We recommend creating a new virtual environment or Anaconda environment before attempting to install
OpenOA. To create and activate such a new environment with the name "openoa-env" using Anaconda:

```
conda create --name openoa-env python=2.7
conda activate openoa-env
```

#### Microsoft Windows:

For users Microsoft Windows, the Anaconda python distribution is required. The reason is that pip on windows requires
Visual Studio libraries to compile some of the dependencies. This can be resolved by manually installing the following
packages via conda, which installs pre-built binaries of these dependencies, before attempting a pip install of OpenOA.

```
conda install shapely
conda install geos
conda install fiona
```

If errors about Visual Studio persist, you can try downloading the Microsoft Visual Studio compiler for Python: https://www.microsoft.com/en-us/download/details.aspx?id=44266


### Installation:

Clone the repository and install the library and its dependencies using pip:

```
git clone https://github.com/NREL/OpenOA.git
pip install ./OpenOA
```

You should now be able to import operational_analysis from the Python interpreter:

```
python
>>> import operational_analysis
```

### Testing

All tests are runnable from setuptools. They are written in the Python unittest framework.

To run unit tests with code coverage reporting:

```
cd ./OpenOA
python setup.py test
```

To run integration tests (longer running, requires data) first unzip the example data:

```
cd OpenOA/examples/operational_AEP_analysis/data
unzip eia_example_data.zip

cd OpenOA/examples/turbine_analysis/data
unzip example_20180829.zip

cd OpenOA
```

Then, you can run the integration test:

```
python setup.py integrate
```

To output junit xml from integration test (used for Jenkins testing):

```
python setup.py integrate -a "--junitxml=./path_to_outputfile.xml"
```



### Documentation

Documentation is automatically built by, and visible through, [Read The Docs](http://openoa.readthedocs.io/).

You can build the documentation with [sphinx](http://www.sphinx-doc.org/en/stable/):

```
pip install sphinx_rtd_theme ipython m2r

cd sphinx
make html
```


### Development

We provide a frozen environment in a requirements.txt file which can be used to install the precise versions
of each dependency present in our own development environment. We recommend utilizing a fresh virtual environment or
Anaconda root before installing these requirements. To use requirements.txt:

```
pip install -r ./OpenOA/requirements.txt
```

Next, we recommend installing OpenOA in editable mode:

```
pip install -e ./OpenOA
```


### Credit

Alphabetically:
Anna Craig,
Jason Fields,
Travis Kemper,
Joseph Lee,
Monte Lunacek,
John Meissner,
Mike Optis,
Jordan Perr-Sauer,
Caleb Phillips,
Eliot Quon,
Sheungwen Sheng,
Eric Simley, and
Lindy Williams.
