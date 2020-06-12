import pandas as pd
from dateutil.parser import parse
import urllib
from operational_analysis.types import PlantData


class TurbineEngieOpenData(PlantData):
    """This class loads wind turbine data from the engie open data platform  https://opendata-renewables.engie.com"""

    def __init__(self, name, start_date ,end_date , engine="pandas"):
        """
        Create a turbine based on data loaded from the engie open data platform.

        Args:
            name(string): uniqiue name (wind_turbine_name) of the wind turbine in the engie open data platform
            start_date(string): start date of the data to be loaded into the object (%m.%d.%y %H:%M)
            end_date(string): end date of the data to be loaded into the object (%m.%d.%y %H:%M)

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
                                   "wtur_W_avg": "p_avg",
                                   "wmet_wDir_avg": "wa_avg",
                                   "wmet_wdspd_avg": "ws_avg"})
        self.scada.df.set_index('time', inplace=True, drop=False)
        self.scada.df.drop(['date_time', 'p_avg', 'wa_avg', 'ws_avg'], axis=1, inplace=True)
