#!/usr/bin/env python
import io
import os
import re
from setuptools import setup
import sys
from setuptools.command.test import test as TestCommand
from setuptools import find_packages

# Configs ##########

# Read the version from the __init__.py file without importing it
def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def read_file(filename):
    this_directory = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_directory, filename), encoding='utf-8') as f:
        f_text = f.read()
    return f_text

# setup.py main ##########

setup(name='OpenOA',
      version=find_version("operational_analysis", "__init__.py"),
      description='A package for collecting and assigning wind turbine metrics',
      long_description=read_file('readme.md'),
      long_description_content_type='text/markdown',
      author='NREL PRUF OA Team',
      author_email='openoa@nrel.gov',
      url='https://github.com/NREL/OpenOA',
      packages=find_packages(exclude=["test"]),
      include_package_data=True,
      data_files=[('operational_analysis/types', ['operational_analysis/types/plant_schema.json'])],
      install_requires=["numpy",
                        "scipy",
                        "pandas",
                        "pygam",
                        "tqdm",
                        "statsmodels",
                        "scikit_learn",
                        "EIA-python",
                        "requests",
                        "pyproj",
                        "shapely"],
      tests_require=['pytest', 'pytest-cov'],
      python_requires='>=3.6'
      )
