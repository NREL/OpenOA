import pandas as pd
from operational_analysis.types import PlantData


class TurbineExampleProject(PlantData):
    """This class loads data for NREL's GE Turbine into a PlantData object"""

    def __init__(self, path="data", name="turbine_example", engine="pandas"):

        super(TurbineExampleProject, self).__init__(path, name, engine, toolkit=[])

    def prepare(self):
        self.scada.load(self._path, "scada_10min_4cols", "csv")
        self.scada.rename_columns({"time": "dttm",
                                   "power_kw": "kw",
                                   "winddirection_deg": "nacelle_position",
                                   "windspeed_ms": "wind_speed"})
        self.scada.df.set_index('time', inplace=True, drop=False)
        self.scada.df.drop(['dttm', 'kw', 'nacelle_position', 'wind_speed'], axis=1, inplace=True)
        self.scada.normalize_time_to_datetime("%Y-%m-%d %H:%M:%S")
