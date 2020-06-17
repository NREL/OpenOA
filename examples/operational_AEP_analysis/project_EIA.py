##############################################
# Data import script for Example EIA project #
##############################################

"""

This script imports the EIA energy data and reanalysis data needed to perform an operational AEP estimate.

Reported net energy data are real but reported availablity and curtailment losses were synthesized

"""

import pandas as pd
from operational_analysis.types import PlantData
from operational_analysis.toolkits import unit_conversion as un


class Project_EIA(PlantData):
    """This class loads data for the EIA site into a PlantData object"""

    def __init__(self, path="data",
                 name="EIA", engine="pandas", toolkit=["plant_analysis"]):
        super(Project_EIA, self).__init__(path, name, engine, toolkit)

    def prepare(self):
        """
        Do all loading and preparation of the data for this plant.
        """

        # Set time frequencies of data
        self._meter_freq = '1MS'  # Monthly meter data
        self._curtail_freq = '1MS'  # Monthly curtailment data

        ######################
        # MONTHLY METER DATA #
        ######################

        self._meter.load(self._path, 'plant_data', 'csv')  # Load monthly data

        # Get 'time' field in datetime format
        self._meter.df['time'] = pd.to_datetime(self._meter.df['year_month'], format='%Y %m ')
        self._meter.df.set_index('time', inplace=True, drop=False)  # Set datetime as index

        # Rename variables
        self._meter.df['energy_kwh'] = self._meter.df['net_energy_mwh'] * 1000.  # convert MWh to kWh

        # Remove the fields we are not yet interested in
        self._meter.df.drop(['year_month', 'net_energy_mwh', 'availability_pct', 'curtailment_pct'], axis=1,
                            inplace=True)

        #############################################
        # MONTHLY CURTAILMENT AND AVAILABILITY DATA #
        #############################################

        self._curtail.load(self._path, 'plant_data', 'csv')  # Load monthly data

        # Get 'time' field in datetime format
        self._curtail.df['time'] = pd.to_datetime(self._curtail.df['year_month'], format='%Y %m')
        self._curtail.df.set_index('time', inplace=True, drop=False)  # Set datetime as index

        # Get losses in energy units
        gross_energy = un.compute_gross_energy(self._curtail.df['net_energy_mwh'],
                                               self._curtail.df['availability_pct'],
                                               self._curtail.df['curtailment_pct'], 'frac', 'frac')
        self._curtail.df['curtailment_kwh'] = self._curtail.df['curtailment_pct'] * gross_energy * 1000
        self._curtail.df['availability_kwh'] = self._curtail.df['availability_pct'] * gross_energy * 1000

        # Remove the fields we are not yet interested in
        self._curtail.df.drop(['net_energy_mwh', 'year_month'], axis=1, inplace=True)

        ###################
        # REANALYSIS DATA #
        ###################

        # merra2
        self._reanalysis._product['merra2'].load(self._path, "merra2_data", "csv")
        self._reanalysis._product['merra2'].rename_columns({"time": "datetime",
                                                            "windspeed_ms": "ws_50m",
                                                            "rho_kgm-3": "dens_50m",
                                                            "winddirection_deg": "wd_50m"})
        self._reanalysis._product['merra2'].normalize_time_to_datetime("%Y-%m-%d %H:%M:%S")
        self._reanalysis._product['merra2'].df.set_index('time', inplace=True, drop=False)

        # ncep2
        self._reanalysis._product['ncep2'].load(self._path, "ncep2_data", "csv")
        self._reanalysis._product['ncep2'].rename_columns({"time": "datetime",
                                                           "windspeed_ms": "ws_10m",
                                                           "rho_kgm-3": "dens_10m",
                                                           "winddirection_deg": "wd_10m"})
        self._reanalysis._product['ncep2'].normalize_time_to_datetime("%Y%m%d %H%M")
        self._reanalysis._product['ncep2'].df.set_index('time', inplace=True, drop=False)

        # erai
        self._reanalysis._product['erai'].load(self._path, "erai_data", "csv")
        self._reanalysis._product['erai'].rename_columns({"time": "datetime",
                                                          "windspeed_ms": "ws_58",
                                                          "rho_kgm-3": "dens_58",
                                                          "winddirection_deg": "wd_58"})
        self._reanalysis._product['erai'].normalize_time_to_datetime("%Y-%m-%d %H:%M:%S")
        self._reanalysis._product['erai'].df.set_index('time', inplace=True, drop=False)
