# Changelog
All notable changes to this project will be documented in this file. If you make a notable change to the project, please add a line describing the change to the "unreleased" section. The maintainers will make an effort to keep the [Github Releases](https://github.com/NREL/OpenOA/releases) page up to date with this changelog. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased
- A static yaw misalignment analysis class `StaticYawMisalignment` has been added to estimate static yaw misalignment as a function of wind speed for individual wind turbines using turbine-level SCADA data
  - The new `07_static_yaw_misalignment` example notebook demonstrates the application of the yaw misalignment method using the example La Haute Borne data
- Added downloader utils module containing functions for downloading generic files from the web, downloading files from Zenodo, and downloading monthly-resolution ERA5 and MERRA2 data
- Added example notebook "02c_plant_aep_analysis_cubico.ipynb" that demonstrates creating a `PlantData` object and running AEP analysis for two Cubico wind plants (Kelmarsh and Penmanshiel) using open data downloaded from Zenodo
- Hard-coded reanalyis product abbreviation requirements in the analysis classes have been moved to check that the provided abbreviations match the reanalysis abbreviations used for the `PlantData.reanalysis` dictionary keys.
- Analysis classes are now attached to `PlantData` at the time of import, maintaining the same behavior as a standalone analysis class import. For example, the following two import patters produce the same results
  ```python
  from openoa import PlantData
  from openoa.analysis import WakeLosses

  project = PlantData(<kwargs>)

  # Original pattern
  wake_classic = WakeLosses(project)

  # New, equivalent pattern
  wake_new = project.WakeLosses()
  ```

- Modern dependency stacks now supported!
  - Upgrading past major versions of Scikit-Learn (1.0) and Pandas (2.0), in conjunction with their own dependencies, caused small divergences in the MonteCarloAEP analysis method with Daily GBM, and the Wake Losses analysis method with UQ. The magnitude of the differences are small compared with the magnitude of the output.
  - In general, OpenOA is now moving away from pinning the maximum dependency version, and will stick to defining minimum dependencies to ensure modern API usage is supported across the software.
- Updated documentation for users and contributors in the Getting Started section.
- `utils.filters.bin_filter` was converted from a for loop to a vectorized method
- `utils.filters.bin_filter` and `utils.timeseries.percent_nan` were converted to be nearly pure NumPy methods operating on NumPy arrays for significant speedups of the TIE analysis method.
- `analysis.TurbineLongTermGrossEnergy.filter_turbine_data` was cleaned up for a minor gain in efficiency and readability.
- Analysis class API redesign
  - Analysis classes can be created from a `PlantData` object, like the following:

    ```python
    from openoa import PlantData

    project = PlantData("""your kwargs""")
    aep = project.MonteCarloAEP("""your analysis settings""")
    ```

  - A deep copy of the original `PlantData` object is now stored in the analysis class so that the project data is stable between uses, allowing more flexibility for users running a variety of analyses.
  - All analysis inputs are able to be provided at the initialization or run level, allowing more flexibility for when analyses are designed and modified. Additionally, the analysis defaults are set at initialization, so settings are only changed between runs if the users specifies a change.
  - The only settings that cannot be modified in an analysis run are those that change the underlying data settings, which will now require a new analysis method. See the following example:

  ```python
  from openoa import PlantData

  project = PlantData()  # note: kwargs must actually be provided to create a PlantData object

  # Use and validate of the SCADA temperature data
  aep = project.MonteCarloAEP(reg_temperature=True)

  # No longer allowed because this adds a new wind direction data requirement, which may
  # not have been validated
  aep.run(reg_wind_direction=True)

  # New method for running variations on the underlying data, which do not modify the original
  # project data in any way
  aep_temp = project.MonteCarloAEP(reg_temperature=True)
  aep_wd = project.MonteCarloAEP(reg_wind_direction=True)

  # Compare your results
  ...
  ```

  - Analysis requirements and minimum schema have been provided in the `openoa/schema` library. To review a dictionary of the minimal data requirements for an anaylsis, users may view the `ANALYSIS_REQUIREMENTS` found in `openoa/schema/metadata.py`, or be importing it and viewing as a dictionary `from openoa.schema.metadata import ANALYSIS_REQUIREMENTS`. Alternatively there is a simple landing page for analysis-specific schema files available in the [schema readme](openoa/schema/README.md)

- Better `__repr__` methods for `PlantData` and `PlantMetaData`.
  - Printing a `PlantData` object now provides a high level statistical summary of each of the
    datasets in `PlantData`, alongside other key variables.
  - Printing a `PlantMetaData` object now shows the default or provided column mapping with the
    associated expected dtypes and units, alongside other key variables.
  - Creating a class will take all of the same parameters, moving all data validation parameters to the front of the arguments for each class, so check your class initializations when changing versions.
  - `AnalysisClass.run()` now takes all of the same arguments as the class initialization, except for those that modify what data will be validated. For example, `MonteCarloAEP` has arguments `reg_temperature` and `reg_wind_direction`, which flag if additional columns should be present in the reanalysis data, therefore modifying the data validation requirements. As such, they will not be able to updated in `run()`, and a new analysis class instance will need to be created.
  - `reanalysis_subset` is being replaced with `reanalysis_products` in all cases to use a consistent naming convention across classes.

### Deprecations

- The following methods have been removed from the plotting library, `utils/plot.py` given that they have either been replaced with newer methods or have been unused and unmaintained for long enough that their original intent is lost. If you still use any of these, please let us know in the Issues, and we'll be happy to bring it up to date:
  - `plot_array`
  - `subplot_powerRose_array`
  - `powerRose_array`
  - `subplot_c1_c2`
  - `subplot_c1_c2_flagged`
  - `subplot_c1_c2_raw_flagged`
  - `subplt_power_curve`
  - `turbine_polar_line`
  - `turbine_polar_4Dscatter`
  - `turbine_polar_contourf`
  - `turbine_polar_contour`


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
