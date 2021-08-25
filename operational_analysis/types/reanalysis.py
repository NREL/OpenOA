import os
import importlib

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
        saved as csv files in the specified path, and then loaded into the ReanalysisData object. Note that only
        "merra2" and "era5" reanalysis products are supported when using the "planetos" format.

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
                for product in [p for p in self._products if p in ["merra2", "era5"]]:
                    # Download from PlanetOS if csv file doesn't already exist
                    if not os.path.exists(os.path.join(path, "{}_{}.csv".format(name, product))):
                        reanalysis_downloading.download_reanalysis_data_planetos(
                            product,
                            lat=lat,
                            lon=lon,
                            save_pathname=path,
                            save_filename="{}_{}".format(name, product),
                            **kwargs,
                        )

                    self._product[product].load(path, "{}_{}".format(name, product))
            else:
                raise NotImplementedError(
                    'Not a valid format. Allowable formats are "csv" and "planetos".'
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
