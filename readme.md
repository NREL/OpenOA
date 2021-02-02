<img src="https://github.com/NREL/OpenOA/blob/develop/Open%20OA%20Final%20Logos/Color/Open%20OA%20Color%20Transparent%20Background.png?raw=true" alt="OpenOA" width="300"/>

![](https://github.com/NREL/OpenOA/workflows/Tests/badge.svg?branch=develop) [![](https://readthedocs.org/projects/openoa/badge/?version=latest)](https://openoa.readthedocs.io) [![codecov](https://codecov.io/gh/NREL/OpenOA/branch/develop/graph/badge.svg)](https://codecov.io/gh/NREL/OpenOA)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NREL/OpenOA/master?filepath=examples)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

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

If you would like to try out the code before installation or simply explore the possibilities, please see our examples on [Binder](https://mybinder.org/v2/gh/NREL/OpenOA/master?filepath=examples).

### Requirements

  * Python 3.6-3.8 with pip.

OpenOA should be compatible with newer versions of Python, but one of its dependencies, Shapely, does not yet have binary wheels in pip for Python 3.9 on Mac.

We strongly recommend using the Anaconda Python distribution and creating a new conda environment for OpenOA. You can download Anaconda through [their website.](https://www.anaconda.com/products/individual)

After installing Anaconda, create and activate a new conda environment with the name "openoa-env":

```
conda create --name openoa-env python=3.8
conda activate openoa-env
```

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

### Development

Development dependencies are provided in a requirements.txt file.

We recommend utilizing a fresh virtual environment or Anaconda root before installing these requirements. To use requirements.txt:

```
pip install -r ./OpenOA/requirements.txt
```

Next, we recommend installing OpenOA in editable mode:

```
pip install -e ./OpenOA
```

#### Extracting Example Data

The example data will be automaticaly extracted as needed by the tests. The following command is provided for reference:

```
unzip examples/data/la_haute_borne.zip -d examples/data/la_haute_borne/
```

In addition, you will need to install the packages required for running the examples with the following command:

```
pip install -r ./OpenOA/examples/requirements.txt
```

#### Testing
Tests are written in the Python unittest framework and are runnable using pytest. To run all tests with code coverage reporting:

```
pytest --cov=operational_analysis
```

To run unit tests only:

```
pytest --ignore=test/regression/ --cov=operational_analysis
```

#### Documentation

Documentation is automatically built by, and visible through, [Read The Docs](http://openoa.readthedocs.io/).

You can build the documentation with [sphinx](http://www.sphinx-doc.org/en/stable/), but will need to ensure [Pandoc is installed](https://pandoc.org/installing.html) on your computer first:

```
cd sphinx
pip install -r requirements.txt
make html
```


### Contributors

Alphabetically:
Nathan Agarwal,
Nicola Bodini,
Anna Craig,
Jason Fields,
Rob Hammond,
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
