#!/usr/bin/env python
import io
import os
import re
import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


# Configs ##########

REQUIRED = [
    "statsmodels",
    "scikit_learn>=0.20.1,<1.0",
    "requests>=2.21.0",
    "eia-python>=1.22",
    "pyproj>=2.6.1",
    "shapely>=1.7.1",
    "numpy>=1.15.4",
    "pandas>=0.23.4,<1.3",
    "pygam>=0.8.0",
    "scipy>=1.1.0",
    "statsmodels>=0.11",
    "tqdm>=4.28.1",
    "matplotlib>=2.1.0",
    "bokeh==2.3.*",
    "pytz",
]

TESTS = ["pytest>=5.4.2", "pytest-cov>=2.8.1"]

EXTRAS = {
    "docs": [
        "ipython",
        "m2r2",
        "sphinx",
        "sphinxcontrib-napoleon",
        "sphinx_rtd_theme",
        "nbmerge",
        "nbsphinx",
    ],
    "wtk": ["h5pyd"],
    "develop": [
        "pre-commit",
        "black",
        "isort",
        "flake8",
        "flake8-docstrings",
        "pytest",
        "pytest-cov",
    ],
}


# Read the version from the __init__.py file without importing it
def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def read_file(filename):
    this_directory = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_directory, filename), encoding="utf-8") as f:
        f_text = f.read()
    return f_text


# setup.py main ##########

setup(
    name="OpenOA",
    version=find_version("operational_analysis", "__init__.py"),
    description="A package for collecting and assigning wind turbine metrics",
    long_description=read_file("readme.md"),
    long_description_content_type="text/markdown",
    author="NREL PRUF OA Team",
    author_email="openoa@nrel.gov",
    url="https://github.com/NREL/OpenOA",
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    data_files=[("operational_analysis/types", ["operational_analysis/types/plant_schema.json"])],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    tests_require=TESTS,
    python_requires=">=3.6, <=3.10",
)
