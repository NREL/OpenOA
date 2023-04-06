import pandas as pd

class EntrConnection:
    _conn: object

    def pandas_query(self, query_string:str) -> pd.DataFrame:
        pass

class PySparkEntrConnection(EntrConnection):

    def __init__(self):
        """
        Get PySpark-Based Connection object for ENTR Warehouse.
        """
        from pyspark.sql import SparkSession
        self._conn = SparkSession.builder\
            .appName("entr_openoa_connector")\
            .config("spark.driver.memory", "4g")\
            .config("spark.executor.memory", "4g")\
            .config("spark.sql.warehouse.dir", "/home/jovyan/warehouse")\
            .config("spark.hadoop.javax.jdo.option.ConnectionURL", "jdbc:derby:;databaseName=/home/jovyan/warehouse/metastore_db;create=true")\
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
            .enableHiveSupport()\
            .getOrCreate()
        self._conn.sql("use entr_warehouse")

    def pandas_query(self, query_string:str) -> pd.DataFrame:
        """
        Query the PySpark-Based ENTR Warehouse, returning a pandas dataframe.

        Args:
            query_string(:obj:`str`): Spark SQL Query
        
        Returns:
            :obj:`pandas.DataFrame`: Result of the query.
        """
        return self._conn.sql(query_string).toPandas()

class PyHiveEntrConnection(EntrConnection):

    def __init__(self,thrift_server_host:str="localhost",thrift_server_port:int=10000):
        """
        Get PyHive-Based Connection object for ENTR Warehouse. This connection object at self._conn is DBAPI2 compatible.

        Args:
            thrift_server_host(:obj:`str`): URL of Apache Thrift2 server
            thrift_server_port(:obj:`int`): Port of Apache Thrift2 server
        """
        from pyhive import hive
        self._conn = hive.Connection(host=thrift_server_host, port=thrift_server_port)

    def pandas_query(self, query_string:str) -> pd.DataFrame:
        """
        Query the PyHive-Based ENTR Warehouse, returning a pandas dataframe.

        Args:
            query_string(:obj:`str`): SQL Query
        
        Returns:
            :obj:`pandas.DataFrame`: Result of the query.
        """
        return pd.read_sql(query_string, self._conn)