import pandas as pd
from operational_analysis.types import PlantData
import numpy as np
from operational_analysis.toolkits.unit_conversion import convert_power_to_energy
from operational_analysis.toolkits.met_data_processing import compute_wind_direction
from dateutil.parser import parse
import urllib

class TurbineExampleProject(PlantData):
    """This class loads data for NREL's GE Turbine into a PlantData object"""

    def __init__(self, path="data", name="turbine_example", engine="pandas"):

        self._scada_freq = '10T'

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

        self.fix_data_issues()

    def fix_data_issues(self):
        """
        TODO: Remove this method!
        This is a temporary method to remedy the data issues in this example project so it can be used for testing.
        """
        p = self
        # Project doesn't have turbine ids, but analysis requires it.
        p.scada.df["id"] = "T0"
        p.scada.df.index = pd.to_datetime(p.scada.df.index)

        # Project is missing reanalysis data, but analysis requires it.
        def generate_reanal(index):
            d = pd.DataFrame(index=index)
            d["u_ms"] = np.random.random(d.shape[0]) * 15
            d["v_ms"] = np.random.random(d.shape[0]) * 15
            d["windspeed_ms"] = np.sqrt(d["u_ms"]**2 + d["v_ms"]**2)
            d["winddirection_deg"] = compute_wind_direction(d["u_ms"], d["v_ms"])
            d["rho_kgm-3"] = np.random.random(d.shape[0]) * 20
            return d
        
        index_reanal = pd.date_range(start = '1990-01-01', end = '2017-12-31')
        p._reanalysis._product['merra2'].df = generate_reanal(index_reanal)
        p._reanalysis._product['erai'].df = generate_reanal(index_reanal)
        p._reanalysis._product['ncep2'].df = generate_reanal(index_reanal)

        # Project is missing energy column
        p.scada.df['energy_kwh'] = convert_power_to_energy(p.scada.df['power_kw'], sample_rate_min=10.0)

class TurbineEngieOpenData(PlantData):
    """This class loads wind turbine data from the engie open data platform  https://opendata-renewables.engie.com"""

    def __init__(self, name, start_date ,end_date , engine="pandas"):
        """
        Create a turbine based on data loaded from the engie open data platform.

        Args:
            name(string): uniqiue name (wind_turbine_name) of the wind turbine in the engie open data platform
            start_date(string): start date of the data to be loaded into the object (%d.%m.%y %H:%M)
            end_date(string): end date of the data to be loaded into the object (%d.%m.%y %H:%M)

        Returns:
            New object
        """
        path = ""
        self._start_date = start_date
        self._end_date = end_date
        super(TurbineEngieOpenData, self).__init__(path, name, engine, toolkit=[])

    def prepare(self):
        #The Engie Open Data set consists of two data sets, 2013 - 2016 and 2017 - 2020. The data is collected from 
        #the two sources depending on the selected period
        self._urls = []
        if parse(self._start_date,dayfirst=True).year <= 2016:
             self._urls.append('https://opendata-renewables.engie.com/api/v2/catalog/datasets/la-haute-borne-data'
                               +'-2013-2016/exports/json?where=wind_turbine_name%20%3D%20%22')
        elif parse(self._start_date,dayfirst=True).year > 2016 or parse(self._end_date.year,dayfirst=True).year > 2016:
             self._urls.append('https://opendata-renewables.engie.com/api/v2/catalog/datasets/la-haute-borne-data'
                               +'-2017-2020/exports/json?where=wind_turbine_name%20%3D%20%22')
        for url in self._urls:
            url = (url + self._name+'%22%20and%20date_time%20%3E%3D%20%22'
                 +urllib.parse.quote(self._start_date)+'%22%20and%20date_time%20%3C%3D%20%22'
                 +urllib.parse.quote(self._end_date)+'%22&rows=-1&pretty=false&timezone=UTC')
            if self.scada.df is None:
                 self.scada.df = pd.read_json(url,orient='columns')
            else:
                 self.scada.df.append(pd.read_json(url,orient='columns'), ignore_index=True)
        self.scada.rename_columns({"time": "date_time",
                                   "power_kw": "p_avg",
                                   "winddirection_deg": "wa_avg",
                                   "windspeed_ms": "ws_avg"})
        self.scada.df.set_index('time', inplace=True, drop=False)
        self.scada.df.drop(['date_time', 'p_avg', 'wa_avg', 'ws_avg'], axis=1, inplace=True)