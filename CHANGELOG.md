# Changelog
All notable changes to this project will be documented in this file. If you make a notable change to the project, please add a line describing the change to the "unreleased" section. The maintainers will make an effort to keep the [Github Releases](https://github.com/NREL/OpenOA/releases) page up to date with this changelog. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 3.0rc2
- Everything from release candidate 1
- IEC 61400-25 tag names are now used throughout the code base for column naming/calling conventions
- Wake Loss Method now released(!) and available via: `from openoa.analysis import WakeLosses`.

## 3.0rc1
- The package name is changed from `operational_analysis` to `openoa` to be more consistent with how we expect to import OpenOA!
- `PlantData` is now fully based on attrs dataclasses and utilizing the pandas `DataFrame` for all internal data structures
  - `PlantData` can now be imported via `from openoa import PlantData`
  - By using attrs users no longer have to subclass `PlantData` and create their own `PlantData.prepare` method.
  - Users can now bring their own column naming, and provide a metadata definition so columns are mapped under the hood through the `PlantMetaData` class (see Intro Example for more information!)
  - `PlantData.scada` (or similar) is now used in place of accessing the SCADA (or similar) dataframe
  - v2 `ReanalysisData` and `AssetData` methods have been absorbed by `PlantData` in favor of a unified data structure and means to operate on data.
  - v2 `TimeSeriesTable` is removed in favor of a pandas-based API and data usage
- openoa has a new import structure
  - `PlantData` is available at the top level: `from openoa import PlantData`
  - tookits -> utils via `from openoa.utils import xx`
    - pandas_plotting -> plot
    - quality_check_automation -> qa (formerly located in methods)
  - methods -> analysis via `from openoa.analysis import xx`
- Convenience methods such as `PlantData.turbine_ids` or `PlantData.tower_df(tower_id="x")` have been added to address commonly used code patters
- Analysis methods are now available through `from openoa.analysis import <AnalysisClass>`
- A wake loss analysis class `WakeLosses` has been added to estimate operational wake losses using turbine-level SCADA data
  - The new `06_wake_loss_analyis` example notebook demonstrates how to use the wake loss analysis method
- Renamed `compute_shear_v3` to `compute_shear` and deleted old version of `compute_shear`.
- The `utils` subpackage has been cleaned up to take both pandas `DataFrame` and `Series` objects where appropriate, refactors pandas code to be much cleaner for both performance and readability, has more user-friendly error messages, and has more consist outputs
- `openoa.utils.imputing.correlation_matrix_by_id_column` has been renamed to `openoa.utils.imputing.asset_correlation_matrix`
- A new 00_x example notebook is replace the 1a/b QA examples to highlight how the `project_ENGIE.py` methods are created. This creates an example for users to work with and significantly more details on how to use the new `PlantData` and `PlantMetaData` methods.
- Documentation reorganization and cleanup

## [2.3 - 2022-01-18]
- Replaced hard-coded reanalysis dates in plant analysis with automatic valid date selection and added optional user-defined end date argument. Fixed bug in normalization to 30-day months.
- Toolkit added for downloading reanalysis data using the PlanetOS API
- Added hourly resolution to AEP calculation
- Added wind farm plotting function to pandas_plotting toolkit using the Bokeh library
- Split the QC methods into a more generic `WindToolKitQualityControlDiagnosticSuite` class and WTK-specific subclass: `WindToolKitQualityControlDiagnosticSuite`.
- Updated filter algorithms in AEP calculation, now with a proper outlier filter

## [2.2 - 2021-05-28]
- IAV incorporation in AEP calculation
- Set power to 0 for windspeeds above and below cutoff in IEC power curve function.
- Split unit tests from regression tests and updated CI pipeline to run the full regression tests weekly.
- Flake8 with Black code style implemented with git hook to run on commit
- Updated long-term loss calculations to weight by monthly/daily long-term gross energy
- Added wind turbine asset data to example ENGIE project
- Reduce amount of time it takes to run regression tests by decreasing number of monte carlo iterations. Reduce tolerance of float comparisons in plant analysis regression test. Linear regression on daily data is removed from test.
- Bugfixes, such as fixing an improper python version specifier in setup.py and replacing some straggling references to the master branch with main.

## [2.1 - 2021-02-17]
- Modify bootstrapping approach for period of record sampling. Data is now sampled with replacement, across 100% of the POR data.
- Cleaned up dependencies for JOSS review. Adding peer-reviewed JOSS paper.
- Add Binder button to Readme which makes running the example notebooks easier.
- Set maximum python version to 3.8, due to an install issue for dependency Shapely on Mac with Python 3.9.

## [2.0.1 - 2020-10-13]
- Replaced `GeoPandas` functionality with `pyproj` and `Shapely` for coordinate
reference system conversion and distance measurements.
- Moved and renamed tests and updated the documentation accordingly.

## [2.0.0 - 2020-08-11]
- Switch to [semantic versioning](https://semver.org) from this release forward.
- Efficiency improvements in AEP calculation
- Energy Yield Analysis (EYA) added to Operational Assessment (OA) Gap Analysis method
- Uncertainty quantification for electrical losses and longterm turbine gross energy
- Implemented open source Engie example data
- Complete update of example notebooks
- Switch to standard BSD-3 Clause license
- Automated quality control method to assist with data ingestion. Tools in this method include daylight savings time change detection and identification of the diurnal cycle.
- Add electrical losses method
- Method for estimating long-term turbine gross energy (excluding downtime and underperformance losses)
- CI pipeline using Github Actions includes regression testing with Pytest, code coverage reporting via CodeCov, packaging and distribution via Pypi, and automatic documentation using ReadTheDocs.

## [1.1] - 2019-01-29
- Python3 Support
- Addition of reanalysis schemas to the Sphinx documentation
- Easy import of EIA data using new module: Metadata_Fetch
- Updated contributing.md document
- Quality checks for reanalysis data
- Improved installation instructions
- Integration tests are now performed in CI
- Performed PEP8 linting

## [1.0] - 2018-12-06
- Refactor many analysis and toolkit modules to make them conform to a standard API (init, prepare, and run method).
- Timeseries Table is now an integrated component, no sparkplug-datastructures dependency
- Plant Level AEP method w/ Monte Carlo
- Turbine / Scada level toolkits: Filtering, Imputing, Met, Pandas Plotting, Timeseries, Unit Conversion
- Most toolkits and all methods are fully documented in Sphinx.
- Two example notebooks: Operational AEP Analysis and Turbine Analysis
- All toolkits except for Pandas Plotting have > 80% test coverage.
