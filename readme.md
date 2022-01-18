<img src="https://github.com/NREL/OpenOA/blob/develop/Open%20OA%20Final%20Logos/Color/Open%20OA%20Color%20Transparent%20Background.png?raw=true" alt="OpenOA" width="300"/>

[![Binder Badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NREL/OpenOA/main?filepath=examples) [![Gitter Badge](https://badges.gitter.im/NREL_OpenOA/community.svg)](https://gitter.im/NREL_OpenOA/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Journal of Open Source Software Badge](https://joss.theoj.org/papers/d635ef3c3784d49f6e81e07a0b35ff6b/status.svg)](https://joss.theoj.org/papers/d635ef3c3784d49f6e81e07a0b35ff6b)

[![Documentation Badge](https://readthedocs.org/projects/openoa/badge/?version=latest)](https://openoa.readthedocs.io) ![Tests Badge](https://github.com/NREL/OpenOA/workflows/Tests/badge.svg?branch=develop) [![Code Coverage Badge](https://codecov.io/gh/NREL/OpenOA/branch/develop/graph/badge.svg)](https://codecov.io/gh/NREL/OpenOA)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

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

If you would like to try out the code before installation or simply explore the possibilities, please see our examples on [Binder](https://mybinder.org/v2/gh/NREL/OpenOA/main?filepath=examples).

If you use this software in your work, please cite our JOSS article with the following BibTex:

```
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
```

### Requirements

  * Python 3.6+ with pip.

We strongly recommend using the Anaconda Python distribution and creating a new conda environment for OpenOA. You can download Anaconda through [their website.](https://www.anaconda.com/products/individual)

After installing Anaconda, create and activate a new conda environment with the name "openoa-env":

```
conda create --name openoa-env python=3.8
conda activate openoa-env
```

### Installation

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

#### Common Installation Issues:

- In Windows you may get an error regarding geos_c.dll. To fix this install Shapely using:

```
conda install Shapely
```

- In Windows, an ImportError regarding win32api can also occur. This can be resolved by fixing the version of pywin32 as follows:

```
pip install --upgrade pywin32==255
```

### Development

Development dependencies are provided through the develop extra flag in setup.py. Here, we install OpenOA, with development dependencies, in editable mode, and activate the pre-commit workflow (note: this second step must be done before committing any
changes):

```
pip install -e "./OpenOA[develop]"
pre-commit install
```

Occasionally, you will need to update the dependencies in the pre-commit workflow, which will provide an error when this needs to happen. When it does, this can normally be resolved with the below code, after which you can continue with your normal git workflow:
```
pre-commit autoupdate
git add .pre-commit-config.yaml
```

#### Example Notebooks and Data

The example data will be automaticaly extracted as needed by the tests. To manually extract the example data for use with the example notebooks, use the following command:

```
unzip examples/data/la_haute_borne.zip -d examples/data/la_haute_borne/
```

In addition, you will need to install the packages required for running the examples with the following command:

```
pip install -r ./OpenOA/examples/requirements.txt
```

The example notebooks are located in the `examples` directory. We suggest installing the Jupyter notebook server to run the notebooks interactively. The notebooks can also be viewed statically on [Read The Docs](http://openoa.readthedocs.io/).

```
jupyter notebook
```

#### Testing
Tests are written in the Python unittest framework and are runnable using pytest. There are two types of tests, unit tests (located in `test/unit`) run quickly and are automatically for every pull request to the OpenOA repository. Regression tests (located at `test/regression`) provide a comprehensive suite of scientific tests that may take a long time to run (up to 20 minutes on our machines). These tests should be run locally before submitting a pull request, and are run weekly on the develop and main branches.

To run all unit and regresison tests:
```
pytest
```

To run unit tests only:
```
pytest test/unit
```

To run all tests and generate a code coverage report
```
pytest --cov=operational_analysis
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
Charlie Plumley,
Eliot Quon,
Sheungwen Sheng,
Eric Simley, and
Lindy Williams.
