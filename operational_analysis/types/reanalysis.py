import os
import importlib
from pathlib import Path

import numpy as np

import operational_analysis.toolkits.met_data_processing as met
from operational_analysis.types import timeseries_table
from operational_analysis.toolkits import reanalysis_downloading


class ReanalysisData(object):
    """
    This class houses the different reanalysis data products and their related functions
    for use in the PRUF OA code. ReanalysisData holds an array of TimeseriesTable in the _product attribute.
    The keys (names) of these attributes can be found in the _products attribute.
    """

    def __init__(self, engine="pandas"):
        self._products = ["merra2", "ncep2", "erai", "era5"]
        self._engine = engine
        self._product = {}
        for product in self._products:
            self._product[product] = timeseries_table.TimeseriesTable.factory(engine)

        if engine == "spark":
            self._sql = importlib.import_module("pyspark.sql")
            self._pyspark = importlib.import_module("pyspark")
            self._sc = self._pyspark.SparkContext.getOrCreate()
            self._sqlContext = self._sql.SQLContext.getOrCreate(self._sc)

    def load(self, path, name, format="csv", lat=None, lon=None, **kwargs):
        """
        This method loads reanalysis data into the ReanalysisData object using one of two formats. When format is
        "csv", reanalysis data will be imported from csv files in the specified path with file name "{name}_{product}.
        csv", where product is the reanalysis product name. When format is "planetos", data will be imported from csv
        files if they exist in the specified path. Otherwise, reanalysis data will be downloaded via the PlanetOS API,
        saved as csv files in the specified path, and then loaded into the ReanalysisData object. Wind speed, wind
        direction, and air density variables are also derived from the reanalysis data when using the "planetos" format
        Note that only "merra2" and "era5" reanalysis products are supported when using the "planetos" format.

        Args:
            path (:obj:`string`): Path where reanalysis files are or will be saved
            name (:obj:`string`): String used as a prefix when forming reanalysis data file names (e.g., the
                project name).
            format (:obj:`string`, optional): Format of the reanalysis data. If "csv", data will be imported from
                existing csv files. If "planetos", data will be downloaded via the PlanetOS API. Defaults to "csv".
            lat (:obj:`float`, optional): Latitude (degrees). Used when format = "planetos". Defaults to None.
            lon (:obj:`float`, optional): Longitude (degrees). Used when format = "planetos". Defaults to None.
            **kwargs: Optional keyword arguments passed to
                :py:func:`operational_analysis.toolkits.reanalysis_downloading.download_reanalysis_data_planetos`

        Raises:
            NotImplementedError: When a format other than "csv" or "planetos" is specified
            NotImplementedError: When the ReanalysisData object's engine is specified as "spark"
        """

        if self._engine == "pandas":
            if format == "csv":
                for product in self._products:
                    self._product[product].load(path, "{}_{}".format(name, product))
            elif format == "planetos":
                for product in list(set(self._products) & set(("merra2", "era5"))):
                    # Download from PlanetOS if csv file doesn't already exist
                    if not (Path(path) / f"{name}_{product}.csv").exists():
                        reanalysis_downloading.download_reanalysis_data_planetos(
                            product,
                            lat=lat,
                            lon=lon,
                            calc_derived_vars=True,
                            save_pathname=path,
                            save_filename=f"{name}_{product}",
                            **kwargs,
                        )

                    self._product[product].load(path, "{}_{}".format(name, product))
            else:
                raise NotImplementedError(
                    'Not a valid format. Allowable formats are "csv" and "planetos".'
                )

        if self._engine == "spark":
            raise NotImplementedError("Spark version of this function is not yet implemented")

    def compute_derived_variables(
        self,
        products=None,
        u_col="u_ms",
        v_col="v_ms",
        temperature_col="temperature_K",
        surf_pres_col="surf_pres_Pa",
    ):
        """
        This method computes the derived variables wind speed, wind direction, and air density from reanalysis
        variables including the u and v wind speed components, temperature, and surface pressure for each reanalysis
        data product loaded in the ReanalysisData object. The derived variables are added as columns to the reanalysis
        product dataframes.

        Args:
            products (:obj:`list`, optional): List of reanalysis products to compute derived variables for. If none are
                specified, all products for which dataframes exist will be used. Defaults to None.
            u_col (:obj:`string`, optional): Name of dataframe column containing eastward wind component in m/s.
                Defaults to "u_ms".
            v_col (:obj:`string`, optional): Name of dataframe column containing northward wind component in m/s.
                Defaults to "v_ms".
            temperature_col (:obj:`string`, optional): Name of dataframe column containing temperature in Kelvins.
                Defaults to "temperature_K".
            surf_pres_col (:obj:`string`, optional): Latitude (degrees). Name of dataframe column containing surface
                pressure in Pascals. Defaults to "surf_pres_Pa".

        Raises:
            NotImplementedError: When the ReanalysisData object's engine is specified as "spark"
        """

        if products is None:
            # By default, process reanalysis products for which a dataframe exists
            products = [p for p in self._products if self._product[p].df is not None]

        if self._engine == "pandas":
            for product in products:
                self._product[product].df["windspeed_ms"] = np.sqrt(
                    self._product[product].df[u_col] ** 2 + self._product[product].df[v_col] ** 2
                )
                self._product[product].df["winddirection_deg"] = met.compute_wind_direction(
                    self._product[product].df[u_col], self._product[product].df[v_col]
                )
                self._product[product].df["rho_kgm-3"] = met.compute_air_density(
                    self._product[product].df[temperature_col],
                    self._product[product].df[surf_pres_col],
                )

        if self._engine == "spark":
            raise NotImplementedError("Spark version of this function is not yet implemented")

    def save(self, path, name):
        if self._engine == "pandas":
            for product, table in self._product.items():
                table.save(path, "{}_{}".format(name, product))

        if self._engine == "spark":
            raise NotImplementedError("Spark version of this function is not yet implemented")

    def rename_columns(self, mapping):
        for k in list(mapping.keys()):
            if k != mapping[k]:
                self._reanalysis[k] = self._reanalysis[mapping[k]]
                self._reanalysis[mapping[k]] = None

    def head(self):
        return self._reanalysis.head()
