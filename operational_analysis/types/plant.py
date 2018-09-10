import io
import json
import os

from dateutil.parser import parse

from asset import AssetData
from operational_analysis.types import timeseries_table
from reanalysis import ReanalysisData


class PlantData(object):
    """Plant-level schema for operational wind plant data.
    
    The PlantData object can serialize all of these structures and reload them
    them from the cache as needed.
    
    The underlying datastructure is a TimeseriesTable, which is agnostic to the underlying
    engine and can be implemented with Pandas, Spark, or Dask (for instance).
    
    Individual plants will extend this object with their own
    prepare() and other methods.

    """


    def __init__(self,path,name,engine="pandas",toolkit=["pruf_analysis"]):
        """
        Create a plant data object without loading any data.
        
        Args:
            path(string): path where data should be read/written
            name(string): uniqiue name for this plant in case there's multiple plant's data in the directory
            engine(string): backend engine - pandas, spark or dask
            toolkit(list): the _tool_classes attribute defines a list of toolkit modules that can be loaded
            
        Returns:
            New object
        """
        self._scada = timeseries_table.TimeseriesTable.factory(engine)
        self._meter = timeseries_table.TimeseriesTable.factory(engine)
        self._tower = timeseries_table.TimeseriesTable.factory(engine)
        self._status = timeseries_table.TimeseriesTable.factory(engine)
        self._curtail = timeseries_table.TimeseriesTable.factory(engine)
        self._asset = AssetData(engine)
        self._reanalysis=ReanalysisData(engine)
        self._name = name
        self._path = path
        self._engine = engine

        self._version = 1

        self._status_labels = ["full","unavailable"]

        self._scada_std = {"time": "datetime64[ns]","id": "string","power_kw": "float64",
                            "windspeed_ms": "float64","winddirection_deg": "float64",
                            "status_label": "string", "pitch_deg": "float64", "temp_c": "float64"}
        self._tower_std = {"time": "datetime64[ns]","id": "string"}
        self._meter_std = {"time": "datetime64[ns]","power_kw": "float64","energy_kwh": "float64"}
        self._reanalysis_std = {"time": "datetime64[ns]","windspeed_ms": "float64",
                            "winddirection_deg": "float64","rho_kgm-3": "float64"}
        self._status_std = {"time": "datetime64[ns]","id": "string","status_id": "int64", "status_code":"int64", "status_text":"string"}
        self._curtail_std = {"time": "datetime64[ns]","curtailment_pct": "float64","availability_pct": "float64","net_energy": "float64"}
        self._asset_std = {"id": "string","latitude": "float64","longitude": "float64",
                            "rated_power_kw": "float64","type": "string"}

        self._tables = ["_scada","_meter","_status","_tower","_asset","_curtail","_reanalysis"]


    def amend_std(self,dfname,new_fields):
       """
       Amend a dataframe standard with new or changed fields. Consider running ensure_columns afterward to automatically
       create the new required columns if they don't exist.

       Args:
           dfname (string): one of scada, status, curtail, etc.
           new_fields (dict): set of new fields and types in the same format as _scada_std to be added/changed in the std
       
       Returns:
           New data field standard
       """

       k = "_%s_std" % (dfname,)
       setattr(self,k,dict(itertools.chain(getattr(self,k).iteritems(), new_fields.iteritems())))

    def get_time_range(self):
        """Get time range as tuple
        
        Returns:
            (tuple):
                start_time(datetime): start time
                stop_time(datetime): stop time
        """
        return (self._start_time,self._stop_time)

    def set_time_range(self,start_time,stop_time):
        """Set time range given two unparsed timestamp strings
        
        Args:
            start_time(string): start time
            stop_time(string): stop time
            
        Returns:
            Nothing
        
        """
        self._start_time = parse(start_time)
        self._stop_time = parse(stop_time)

    def save(self, path=None):
        """Save out the project and all JSON serializeable attributes to a file path.

            Args:
                path(string): Location of new directory into which plant will be saved. The directory should not already
                    exist. Defaults to self._path

            Returns:
                Nothing
        """
        if path == None:
            raise RuntimeError("Path not specified.")

        os.mkdir(path)

        meta_dict = {}
        for ca, ci in self.__dict__.iteritems():
            if ca in self._tables:
                ci.save(path, ca)
            elif ca in ["_start_time", "_stop_time"]:
                meta_dict[ca] = str(ci)
            else:
                meta_dict[ca] = ci

        with io.open(os.path.join(path, "metadata.json"), 'w', encoding="utf-8") as outfile:
            outfile.write(unicode(json.dumps(meta_dict, ensure_ascii=False)))

    def load(self, path=None):
        """Load this project and all associated data from a file path

            Args:
                path(string): Location of plant data directory. Defaults to self._path

            Returns:
                Nothing
        """
        if not path:
            path = self._path

        for df in self._tables:
            getattr(self, df).load(path, df)

        meta_path = os.path.join(path, "metadata.json")
        if(os.path.exists(meta_path)):
            with io.open(os.path.join(path, "metadata.json"), 'r') as infile:
                meta_dict = json.load(infile)
                for ca, ci in meta_dict.iteritems():
                    if ca in ["_start_time", "_stop_time"]:
                        ci = parse(ci)
                    setattr(self, ca, ci)

    def ensure_columns(self):
        """Ensure all dataframes contain necessary columns and format as needed"""
        for df in ["_scada","_meter","_status","_tower","_curtail"]:
            if not getattr(self,df).is_empty():
                getattr(self,df).ensure_columns(getattr(self,"%s_std" % (df,)))

    def merge_asset_metadata(self):
        """Merge metadata from the asset table into the scada and tower tables"""
        if (not (self._scada.is_empty()) and (len(self._asset.turbine_ids()) > 0)):
            self._scada.pandas_merge(self._asset.df,["latitude","longitude","rated_power_kw","id",
                                                     "nearest_turbine_id","nearest_tower_id"],"left",on="id")
        if (not (self._tower.is_empty()) and (len(self._asset.tower_ids()) > 0)):
            self._tower.pandas_merge(self._asset.df,["latitude","longitude","rated_power_kw","id",
                                                     "nearest_turbine_id","nearest_tower_id"],"left",on="id")

    def prepare(self):
        """Prepare this object for use by loading data and doing essential preprocessing."""
        self.ensure_columns()
        if not ((self._scada.is_empty()) or (self._tower.is_empty())):
            self._asset.prepare(self._scada.unique("id"),self._tower.unique('id'))
        self.merge_asset_metadata()

    @property
    def scada(self):
        return self._scada

    @property
    def meter(self):
        return self._meter

    @property
    def tower(self):
        return self._tower

    @property
    def reanalysis(self):
        return self._reanalysis
            
    @property
    def status(self):
        return self._status

    @property
    def asset(self):
        return self._asset

    @property
    def curtail(self):
        return self._curtail
