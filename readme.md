<img src="https://github.com/NREL/OpenOA/blob/develop/Open%20OA%20Final%20Logos/Color/Open%20OA%20Color%20Transparent%20Background.png?raw=true" alt="OpenOA" width="300"/>

![](https://github.com/NREL/OpenOA/workflows/Tests/badge.svg?branch=develop) ![](https://readthedocs.org/projects/openoa/badge/?version=latest) [![codecov](https://codecov.io/gh/NREL/OpenOA/branch/develop/graph/badge.svg)](https://codecov.io/gh/NREL/OpenOA)

-----

This library provides a framework for working with large timeseries data from wind plants, such as SCADA.
Its development has been motivated by the WP3 Benchmarking (PRUF) project,
which aims to provide a reference implementation for plant-level performance assessment.

Analysis routines are grouped by purpose into methods,
and these methods in turn rely on more abstract toolkits.
In addition to the provided analysis methods,
anyone can write their own, which is intended to provide natural
growth of tools within this framework.

The library is written around Pandas Data Frames, utilizing a flexible backend
so that data loading, processing, and analysis could be performed using other libraries,
such as Dask and Spark, in the future.

### Requirements

  * Python 3.6+ (e.g., from Anaconda) with pip.

We recommend creating a new virtual environment or Anaconda environment before attempting to install
OpenOA. To create and activate such a new environment with the name "openoa-env" using Anaconda:

```
conda create --name openoa-env python=3
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

### Extracting Example Data

The integration tests and example notebooks require the example data to be extracted from a zip archive and to be placed in a specific folder.
You can do this using the following command:

```
unzip examples/data/la_haute_borne.zip -d examples/data/la_haute_borne/
```

### Testing

All tests are runnable using pytest. They are written in the Python unittest framework.

To run all tests with code coverage reporting:

```
pytest -o python_files=test/*.py --cov=operational_analysis
```

To run unit tests only (does not require example data)

```
pytest -o python_files=test/test_*.py --cov=operational_analysis
```



### Documentation

Documentation is automatically built by, and visible through, [Read The Docs](http://openoa.readthedocs.io/).

You can build the documentation with [sphinx](http://www.sphinx-doc.org/en/stable/):

```
cd sphinx
pip install -r requirements.txt
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


### Contributors

Alphabetically:
Nathan Agarwal,
Nicola Bodini,
Anna Craig,
Jason Fields,
Travis Kemper,
Joseph Lee,
Monte Lunacek,
John Meissner,
Mike Optis,
Jordan Perr-Sauer,
Sebastian Pfaffel,
Caleb Phillips,
Eliot Quon,
Sheungwen Sheng,
Eric Simley, and
Lindy Williams.
