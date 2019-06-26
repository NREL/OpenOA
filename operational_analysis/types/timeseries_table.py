# timeseries_table.py
"""

A basic columnar timeseries datastructure whose
underlying dataframe backend can be Pandas, Dask,
or Spark.
#
The assumption here is that there is a single
column that contains the timestamp and the other columns
are various metrics/data. The time stamps may be on
a regular interval, or they may be irregularly spaced.
The dataframe may be somewhat sparse (e.g., not all metrics may
be defined at all time points).

"""

import datetime
import importlib
import pandas as pd

from operational_analysis import logged_method_call
from operational_analysis import logging
logger = logging.getLogger(__name__)


# The abstract class sets the interface for the timeseries table
class AbstractTimeseriesTable:
    df = None
    _time_field = 'time'
    _metric_fields = []

    def __init__(self):
        pass

    # save data from our dataframe into some format that load can read
    def save(self, path, name, format):
        raise NotImplementedError("Called method on abstract class")

    # load tabular data in some format into our dataframe
    def load(self, path, name, format):
        raise NotImplementedError("Called method on abstract class")

    # ensure the given columns exist, create them if they do not, drop the rest
    def ensure_columns(self, std):
        raise NotImplementedError("Called method on abstract class")

    # given a mapping of column names, rename them
    def rename_columns(self, mapping):
        raise NotImplementedError("Called method on abstract class")

    # given a mapping of column names, rename them
    def copy_column(self, to, fro):
        raise NotImplementedError("Called method on abstract class")

    # return true if the data frame hasn't been initialized or has no data
    def is_empty(self):
        raise NotImplementedError("Called method on abstract class")

    # create columns like year/month/day from the time column
    def explode_time(self, vars):
        raise NotImplementedError("Called method on abstract class")

    # use a strptime string to process a date/time like string into a datetime object
    def normalize_time_to_datetime(self, format, col=None):
        raise NotImplementedError("Called method on abstract class")

    # convert unix time in seconds to datetime
    def epoch_time_to_datetime(self, col=None):
        raise NotImplementedError("Called method on abstract class")

    # return the first 5 rows of our dataframe as a pandas dataframe
    def head(self):
        raise NotImplementedError("Called method on abstract class")

    # apply a function to each element in a given column, modifying the column in place
    def map_column(self, col, func):
        raise NotImplementedError("Called method on abstract class")

    # merge our dataframe with a pandas dataframe
    def pandas_merge(self, right, right_cols, how, on):
        raise NotImplementedError("Called method on abstract class")

    # Return list of unique values in a given column
    def unique(self, col):
        raise NotImplementedError("Called method on abstract class")

    # Combine the given timeseries table with this one, row-wise
    def rbind(self, tt):
        raise NotImplementedError("Called method on abstract class")

    def trim_timeseries(self, start, stop):
        raise NotImplementedError("Called method on abstract class")

    @property
    def time_field(self):
        return self._time_field

    @property
    def metric_fields(self):
        return self._metric_fields

    @property
    def schema(self):
        raise NotImplementedError("Called method on abstract class")


# These inherited classes implement it
class PandasTimeseriesTable(AbstractTimeseriesTable):
    """ Pandas based timeseries table
    """

    def __init__(self, *args, **kwargs):
        self._pd = __import__('pandas', globals(), locals(), [], 0)

    @logged_method_call
    def save(self, path, name, format="csv"):
        """Write data to file
        """
        logger.info("save name:{}".format(name))
        if format != "csv":
            raise NotImplementedError("Cannot save to format %s yet" % (format,))
        self.df.to_csv("%s/%s.csv" % (path, name))

    @logged_method_call
    def load(self, path, name, format="csv", nrows=None):
        """Read data from a file
        """
        logger.info("Loading name:{}".format(name))
        if format != "csv":
            raise NotImplementedError("Cannot save to format %s yet" % (format,))
        self.df = self._pd.read_csv("%s/%s.csv" % (path, name), nrows=nrows)

    def rename_columns(self, mapping):
        """Rename columns based on mapping

        Args:
            mapping (dict): new and old column names based on {"new":"old"} convention
        """
        for k in list(mapping.keys()):
            if k != mapping[k]:
                self.df[k] = self.df[mapping[k]]
                self.df[mapping[k]] = None

    def copy_column(self, to, fro):
        """Copy column data

        Args:
            fro (str): column name to copy data from
            to (str): column name to copy to
        """
        logger.debug("copying {} to {}".format(fro, to))
        self.df[to] = self.df[fro]

    def ensure_columns(self, std):
        """ @deprecated Set column types to specified type

        Args:
            std (dict):
        """
        for col in list(std.keys()):
            logging.debug("checking  {} is astype {} ".format(col, std[col]))
            if col not in self.df.columns:
                if std[col] == 'float64':
                    self.df[col] = float('nan')
                else:
                    self.df[col] = None
            self.df[col] = self.df[col].astype(std[col])
        self.df = self.df[list(std.keys())]

    @property
    def schema(self):
        """ Return schema of this dataframe as a dictionary.

        Returns:
            (dict): {column_name(str): column_type(str)}
        """
        return {col: str(t) for col, t in zip(self.df, self.df.dtypes)}

    def validate(self, schema):
        """ Validate this timeseriestable object against its schema.
        
        Returns:
            (bool): True if valid, Rasies an exception if not valid."""
        if schema["type"] != "timeseries":
            raise Exception("Incompatible schema type {} applied to TimeseriesTable".format(schema["type"]))

        df_schema = self.schema
        for field in schema["fields"]:
            if field["name"] in df_schema.keys():
                assert df_schema[field["name"]] == field["type"], \
                    "Incompatible type for field {}. Expected {} but got {}".format( \
                        field["name"], field["type"], df_schema[field["name"]])
                del df_schema[field["name"]]

        assert len(df_schema) == 0, "Extra columns are present in TimeseriesTable: \n {}".format(df_schema)

        return True

    def is_empty(self):
        """ Test if data is None

        Returs:
            (bool): True if None, False if not None
        """
        return self.df is None

    def explode_time(self, vars=["year", "month", "day"]):
        """ Create new columns for components of time

        Args:
            vars (list): list of time components
        """

        for v in vars:
            self.df[v] = self.df[self._time_field].apply(lambda x: getattr(x, v), 1)

    def normalize_time_to_datetime(self, format="%Y-%m-%d %H:%M:%S", col=None):
        """ Apply datetime format to timestamp column
        """
        if col is None:
            col = self._time_field
        logging.debug("setting {} to datetime ".format(col))
        self.df[col] = self.df[col].apply(lambda x: datetime.datetime.strptime(x, format), 1)

    def to_datetime(self, format="%Y-%m-%d %H:%M:%S", col=None):
        """ Run pd.to_datetime on timestamp column
        """
        if col is None:
            col = self._time_field
        logger.debug("Running pd.to_datetime on {} ".format(col))
        self.df[col] = pd.to_datetime(self.df[col])

    def epoch_time_to_datetime(self, col=None):
        """Format col as datetime
        """
        if col is None:
            col = self._time_field
        logger.debug("Running to_datetime on {}  ".format(col))
        self.df[col] = self.df[col].apply(lambda x: pd.to_datetime(x, unit='s'), 1)

    def head(self):
        """ Head data """
        return self.df.head()

    def map_column(self, col, func):
        """Apply a function to col
        """
        logger.debug("Mapping col:{}".format(col))
        if col not in self.df.columns:
            self.df[col] = 'unknown'

        self.df[col] = self.df[col].apply(func, 1)

    def pandas_merge(self, right, right_cols, how='left', on='id'):
        """ Run merge with data """
        logger.debug("merging right:{} right_cols:{} ".format(right, right_cols))
        self.df = self.df.merge(right.loc[:, right_cols], how=how, on=on)

    def unique(self, col):
        """Get unique values of a column  """
        logger.debug("unique col:{}".format(col))
        return self.df[col].unique()

    def rbind(self, tt):
        """Append data  """
        logger.debug("appending tt.df")
        self.df = self.df.append(tt.df)

    def to_pandas(self):
        """ Return data """
        return self.df

    def trim_timeseries(self, start, stop):
        """ Get time range

        Args:
            start (datetime):  start of time-sereies trim
            stop (datetime):  stop of time-sereies trim
        """
        logger.debug("trim_timeseries start:{} stop:{} ".format(start, stop))
        self.df = self.df.loc[(self.df[self._time_field] >= start) & (self.df[self._time_field] <= stop), :]

    def max(self):
        """ Find maximum timestamp value """
        return self.df[self._time_field].max()

    def min(self):
        """ Find minimum timestamp value """
        return self.df[self._time_field].min()


class SparkTimeseriesTable(AbstractTimeseriesTable):
    def __init__(self, *args, **kwargs):
        self._f = importlib.import_module('pyspark.sql.functions')
        self._t = importlib.import_module('pyspark.sql.types')
        self._sql = importlib.import_module('pyspark.sql')
        self._pyspark = importlib.import_module('pyspark')
        self._sc = self._pyspark.SparkContext.getOrCreate()
        self._sqlContext = self._sql.SQLContext.getOrCreate(self._sc)
        self.type_map = {"datetime64[ns]": self._t.TimestampType(),
                         "string": self._t.StringType(),
                         "object": self._t.StringType(),
                         "float64": self._t.DoubleType()}

    def save(self, path, name, format="parquet"):
        if format != "parquet":
            raise NotImplementedError("Cannot save to format %s yet" % (format,))
        self.df.write.mode('overwrite').parquet("%s/%s.parquet" % (path, name))

    def load(self, path, name, format="parquet", nrows=None):
        if format == "parquet":
            self.df = self._sqlContext.read.parquet('%s/%s.parquet' % (path, name))
        elif format == "csv":
            self.df = self._sqlContext.read.format("com.databricks.spark.csv") \
                .options(header='true', inferschema='true').load("%s/%s.csv" % (path, name))
        if nrows is not None:
            self.df = self.df.limit(nrows)

    def rename_columns(self, mapping):
        for k in list(mapping.keys()):
            if k != mapping[k]:
                self.df = self.df.withColumnRenamed(mapping[k], k)

    def copy_column(self, to, fro):
        self.df = self.df.withColumn(df[fro])

    def ensure_columns(self, std):
        for col in list(std.keys()):
            if col not in self.df.columns:
                self.df = self.df.withColumn(col, self._f.lit(None).cast(self._t.StringType()))
            else:
                cast_to = self.type_map[std[col]]
                self.df = self.df.withColumn(col, self.df[col].cast(cast_to))
        self.df = self.df.select(list(std.keys()))

    def is_empty(self):
        return self.df is None

    def explode_time(self, vars=["year", "month", "day", "hour"]):
        if "year" in vars:
            self.df = self.df.withColumn("year", self._f.year(self._time_field))
        if "month" in vars:
            self.df = self.df.withColumn("month", self._f.month(self._time_field))
        if "day" in vars:
            self.df = self.df.withColumn("day", self._f.dayofmonth(self._time_field))
        if "hour" in vars:
            self.df = self.df.withColumn("hour", self._f.hour(self._time_field))

    def normalize_time_to_datetime(self, format, col=None):
        if col is None:
            col = self._time_field
        raise NotImplementedError("TODO")

    def epoch_time_to_datetime(self, col=None):
        if col is None:
            col = self._time_field
        self.df = self.df.withColumn(col, self._f.from_unixtime(col))

    def head(self):
        return self.df.limit(5).toPandas()

    def map_column(self, col, func):
        as_udf = self._f.udf(func, self._t.StringType())
        self.df.withColumn(col, as_udf(col))

    def pandas_merge(self, right, right_cols, how, on):
        right = right.loc[:, right_cols]
        schema = [self._t.StructField(x, self.type_map[right[x].dtype.name], True) for x in right_cols]
        schema = self._t.StructType(schema)
        right = self._sqlContext.createDataFrame(right, schema)
        self.df = self.df.join(right, on, how)

    def unique(self, col):
        if self.is_empty():
            return []
        else:
            self.df.select(col).distinct().rdd.map(lambda x: x[0]).collect()

    def rbind(self, tt):
        raise NotImplementedError("TODO")

    def trim_timeseries(self, start, stop):
        raise NotImplementedError("TODO")


class DaskTimeseriesTable(AbstractTimeseriesTable):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DASK implementation is TBD")


# The timeseries table class is a factory
class TimeseriesTable:
    _classes = {
        "spark": SparkTimeseriesTable,
        "pandas": PandasTimeseriesTable,
        "dask": DaskTimeseriesTable
    }

    @staticmethod
    def factory(engine="pandas", *args, **kwargs):
        if engine not in list(TimeseriesTable._classes.keys()):
            raise NotImplementedError("Engine %s Not Implemented" % (engine,))
        else:
            return TimeseriesTable._classes[engine](*args, **kwargs)
