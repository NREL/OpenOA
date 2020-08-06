import importlib
import itertools

import geopandas as gp
import numpy as np
import pandas as pd
from shapely.geometry import Point


class AssetData(object):
    """
    This class wraps around a GeoPandas dataframe that contains
    metadata about the plant assets. It provides some useful functions
    to work with this data (e.g., calculating nearest neighbors, etc.)
    """

    def __init__(self, engine="pandas"):
        self._asset = None
        self._nearest_neighbors = None
        self._nearest_towers = None
        self._engine = engine
        if engine == "spark":
            self._sql = importlib.import_module('pyspark.sql')
            self._pyspark = importlib.import_module('pyspark')
            self._sc = self._pyspark.SparkContext.getOrCreate()
            self._sqlContext = self._sql.SQLContext.getOrCreate(self._sc)

    def load(self, path, name, format="csv"):
        if self._engine == "pandas":
            self._asset = pd.read_csv("%s/%s.%s" % (path, name, format))
        elif self._engine == "spark":
            self._asset = self._sqlContext.read.format("com.databricks.spark.csv") \
                .options(header='true', inferschema='true').load("%s/%s.csv" % (path, name)).toPandas()

    def save(self, path, name, format="csv"):
        if self._engine == "pandas":
            self._asset.to_csv("%s/%s.%s" % (path, name, format))
        elif self._engine == "spark":
            self._sqlContext.createDataFrame(self._asset).write.mode("overwrite").format("com.databricks.spark.csv") \
                .options(header='true', inferschema='true').save("%s/%s.csv" % (path, name))

    def prepare(self, active_turbine_ids, active_tower_ids, srs='epsg:4326'):
        """Prepare the asset data frame for further analysis work. Currently, this function calls parse_geometry(srs)
        and calculate_nearest(active_turbine, active_tower), passing through the arguments to this function.

        Args:
            active_turbine_ids (:obj:`list`): List of IDs of turbines to consider.
            active_tower_ids (:obj:`list`): List of IDs of met towers to consider.
            srs (:obj:`str`, optional): Used to define the coordinate
                reference system (CRS). Defaults to the European
                Petroleum Survey Group (EPSG) code 4326 to be used with
                the World Geodetic System reference system, WGS 84.

        Returns: None
            Sets asset 'geometry', 'nearest_turbine_id' and 'nearest_tower_id' column.

        """
        self.parse_geometry(srs)
        self.calculate_nearest(active_turbine_ids, active_tower_ids)

    def parse_geometry(self, srs='epsg:4326', zone=None, longitude=None):
        """Calculate UTM coordinates from latitude/longitude.

        The UTM system divides the Earth into 60 zones, each 6deg of
        longitude in width. Zone 1 covers longitude 180deg to 174deg W;
        zone numbering increases eastward to zone 60, which covers
        longitude 174deg E to 180deg. The polar regions south of 80deg S
        and north of 84deg N are excluded.

        Ref: http://geopandas.org/projections.html

        Args:
            srs (:obj:`str`, optional): Used to define the coordinate
                reference system (CRS). Defaults to the European
                Petroleum Survey Group (EPSG) code 4326 to be used with
                the World Geodetic System reference system, WGS 84.
            zone (:obj:`int`, optional): UTM zone. If set to None
                (default), then calculated from the longitude.
            longitude (:obj:`float`, optional): Reference longitude for
                calculating the UTM zone. If None (default), then taken
                as the average longitude of all assets.

        Returns: None
            Sets asset 'geometry' column.
        """
        if zone is None:
            # calculate zone
            if longitude is None:
                longitude = self.df['longitude'].mean()
            zone = int(np.floor((180 + longitude) / 6.0)) + 1

        self._asset = gp.GeoDataFrame(self._asset)
        self._asset['geometry'] = self._asset.apply(lambda x: Point(x['longitude'], x['latitude']), 1)
        self._asset.set_geometry('geometry')
        self._asset.crs = {'init': srs}
        self._asset = self._asset.to_crs(
            "+proj=utm +zone=" + str(zone) + " +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    def calculate_nearest(self, active_turbine_ids, active_tower_ids):
        """Create or overwrite a column called 'nearest_turbine_id' or 'nearest_tower_id' which contains the asset id
        of the closest active turbine or tower to the closest turbine or tower. The columns are only valid for turbines
        or towers listed in the parameters of this function, and it will only calculate the value of the correct column
        for each asset. Turbines, for example, will have null 'nearest_tower_id' and vice versa.

        Args:
            active_turbine_ids (:obj:`list`): List of IDs of turbines to consider.
            active_tower_ids (:obj:`list`): List of IDs of met towers to consider.

        Returns: None
            Sets asset 'nearest_turbine_id' and 'nearest_tower_id' column.
        """
        self._asset['nearest_turbine_id'] = None
        if active_turbine_ids is not None and len(active_turbine_ids) > 0:
            nn = self.nearest_neighbors()
            for k, v in nn.items():
                v = [val for val in v if val in active_turbine_ids]
                self._asset.loc[self._asset['id'] == k, 'nearest_turbine_id'] = v[0]
        if active_tower_ids is not None and len(active_tower_ids) > 0:
            nt = self.nearest_towers()
            self._asset['nearest_tower_id'] = None
            for k, v in nt.items():
                v = [val for val in v if val in active_tower_ids]
                self._asset.loc[self._asset['id'] == k, 'nearest_tower_id'] = v[0]

    def distance_matrix(self):
        ret = np.ones((self._asset.shape[0], self._asset.shape[0])) * -1
        for i, j in itertools.permutations(self._asset.index, 2):
            point1 = self._asset.loc[i, 'geometry']
            point2 = self._asset.loc[j, 'geometry']
            distance = point1.distance(point2)
            ret[i][j] = distance
        return ret

    def asset_ids(self):
        return self._asset.loc[:, 'id'].values

    def tower_ids(self):
        return self._asset.loc[self._asset['type'] == 'tower', 'id'].values

    def turbine_ids(self):
        return self._asset.loc[self._asset['type'] == 'turbine', 'id'].values

    def remove_assets(self, to_delete):
        self._asset = self._asset.loc[~self._asset['id'].isin(to_delete), :].reset_index(drop=True)

    def nearest_neighbors(self):
        if self._nearest_neighbors is not None:
            return self._nearest_neighbors

        ret = {}
        towers = self._asset.loc[self._asset['type'] == 'tower', :].index
        turbines = self._asset.loc[self._asset['type'] == 'turbine', :].index
        m = self.distance_matrix()
        for i in turbines:
            row = m[i]
            row[row == -1] = float("inf")
            row[towers.tolist()] = float("inf")
            ret[self._asset.loc[i, "id"]] = [self._asset.loc[x, "id"] for x in row.argsort()]

        self._nearest_neighbors = ret
        return ret

    def nearest_tower_to(self, id):
        return self._asset.loc[self._asset['id'] == id, 'nearest_tower_id'].values[0]

    def nearest_turbine_to(self, id):
        return self._asset.loc[self._asset['id'] == id, 'nearest_turbine_id'].values[0]

    def nearest_towers(self):
        if self._nearest_towers is not None:
            return self._nearest_towers

        ret = {}
        turbines = self._asset.loc[self._asset['type'] == 'turbine', :].index
        m = self.distance_matrix()
        for i in turbines:
            row = m[i]
            row[row == -1] = float("inf")
            row[turbines.tolist()] = float("inf")
            ret[self._asset.loc[i, "id"]] = [self._asset.loc[x, "id"] for x in row.argsort()]

        self._nearest_towers = ret
        return ret

    def rename_columns(self, mapping):
        for k in list(mapping.keys()):
            if k != mapping[k]:
                self._asset[k] = self._asset[mapping[k]]
                self._asset[mapping[k]] = None

    def head(self):
        return self._asset.head()

    @property
    def df(self):
        return self._asset
