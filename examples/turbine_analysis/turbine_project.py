import pandas as pd
from operational_analysis.types import PlantData
import numpy as np
from operational_analysis.toolkits.unit_conversion import convert_power_to_energy
from operational_analysis.toolkits.met_data_processing import compute_wind_direction

class TurbineExampleProject(PlantData):
    """This class loads data for NREL's GE Turbine into a PlantData object"""

    def __init__(self, path="data", name="turbine_example", engine="pandas"):

        self._scada_freq = '10T'

        super(TurbineExampleProject, self).__init__(path, name, engine, toolkit=[])

    def prepare(self):
        self.scada.load(self._path, "scada_10min_4cols", "csv")
        self.scada.rename_columns({"time": "dttm",
                                   "wtur_W_avg": "kw",
                                   "wmet_wDir_avg": "nacelle_position",
                                   "wmet_wdspd_avg": "wind_speed"})
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
        p.scada.df['energy_kwh'] = convert_power_to_energy(p.scada.df['wtur_W_avg'], sample_rate_min='10T')

