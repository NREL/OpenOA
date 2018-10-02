#!/usr/bin/env python
import io
import os
import re
from setuptools import setup
import sys
from setuptools.command.test import test as TestCommand

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


# PyTest Runners ##########

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args + " --cov=operational_analysis"))
        sys.exit(errno)

class PyTestIntegrate(PyTest):

    def run_tests(self):
        import shlex
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args + " -o python_files=int_*.py --cov=operational_analysis"))
        sys.exit(errno)


# setup.py main ##########

setup(name='OpenOA',
      version=find_version("operational_analysis", "__init__.py"),
      description='A package for collecting and assigning wind turbine metrics',
      author='NREL PRUF OA Team',
      author_email='caleb.phillips@nrel.gov',
      url='https://github.com/NREL/OpenOA',
      packages=['operational_analysis'],
      install_requires=["numpy==1.14.0",
                        "scipy==1.0.0",
                        "pandas==0.20.3",
                        "geopandas==0.3.0",
                        "pygam==0.4.0",
                        "tqdm==4.19.5",
                        "statsmodels==0.8.0",
                        "scikit_learn==0.18.1"],
      tests_require=['pytest', 'pytest-cov'],
      python_requires='~=2.7',
      cmdclass={'test': PyTest, 'integrate':PyTestIntegrate},
      license='None'
      )
