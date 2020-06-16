from operational_analysis.types import timeseries_table


class ReanalysisData(object):
    """
    This class houses the different reanalysis data products and their related funcitons
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
            self._sql = importlib.import_module('pyspark.sql')
            self._pyspark = importlib.import_module('pyspark')
            self._sc = self._pyspark.SparkContext.getOrCreate()
            self._sqlContext = self._sql.SQLContext.getOrCreate(self._sc)

    def load(self, path, name):
        if self._engine == "pandas":
            for product in self._products:
                self._product[product].load(path, "{}_{}".format(name, product))

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
