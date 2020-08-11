<img src="https://github.com/NREL/OpenOA/blob/develop/Open%20OA%20Final%20Logos/Color/Open%20OA%20Color%20Transparent%20Background.png?raw=true" alt="OpenOA" width="300"/>

![](https://github.com/NREL/OpenOA/workflows/Tests/badge.svg?branch=develop) [![](https://readthedocs.org/projects/openoa/badge/?version=latest)](https://openoa.readthedocs.io) [![codecov](https://codecov.io/gh/NREL/OpenOA/branch/develop/graph/badge.svg)](https://codecov.io/gh/NREL/OpenOA)

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

  * Python 3.6+ with pip.

We strongly recommend using the Anaconda Python distribution and creating a new conda environment for OpenOA. You can download Anaconda through [their website.](https://www.anaconda.com/products/individual)

After installing Anaconda, create and activate a new conda environment with the name "openoa-env":

```
conda create --name openoa-env python=3
conda activate openoa-env
```

#### Special Note for users of Microsoft Windows:

The Anaconda python distribution is *required* for users of Microsoft Windows. This is because the pip package of GDAL for Windows requires Visual Studio to compile some of the dependencies. While advanced users are welcome to explore this option, we find it is easier to install the following packages via Anaconda:

```
conda install shapely
conda install geos
conda install fiona
```

If errors about Visual Studio persist, you can try downloading the [Microsoft Visual Studio compiler for Python](https://www.microsoft.com/en-us/download/details.aspx?id=44266) and compiling GDAL yourself.


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

#### Testing
Tests are written in the Python unittest framework and are runnable using pytest. To run all tests with code coverage reporting:

```
pytest -o python_files=test/*.py --cov=operational_analysis
```

To run unit tests only:

```
pytest -o python_files=test/test_*.py --cov=operational_analysis
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
