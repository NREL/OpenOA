# This class defines key analytical procedures in a quality check process for turbine data.
# After analyzing the data for missing and duplicate timestamps, timezones, Daylight Savings Time corrections, and extrema values,
# a report is produced to allow the user to make informed decisions about how to handle the data.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5pyd
import dateutil
from pyproj import Proj
from operational_analysis.toolkits import timeseries
from operational_analysis.toolkits import filters
from operational_analysis.toolkits import power_curve

from operational_analysis import logged_method_call
from operational_analysis import logging

logger = logging.getLogger(__name__)

class QCAuto(object):

    @logged_method_call
    def __init__(self,scada_df, ws_field='wmet_wdspd_avg', power_field= 'wtur_W_avg', time_field= 'datetime', id_field= 'Unit', freq = '10T', lat_lon= (0,0)):
        """
        Initialize QCAuto object with data and parameters.

        Args:
         scada_df(:obj:`DataFrame object`): DataFrame object that contains turbine data
         ws_field(:obj: 'String'): String name of the windspeed field to scada_df
         power_field(:obj: 'String'): String name of the power field to scada_df
         time_field(:obj: 'String'): String name of the time field to scada_df
         id_field(:obj: 'String'): String name of the id field to scada_df
         freq(:obj: 'String'): String representation of the resolution for the time field to scada_df
         lat_lon(:obj: 'tuple'): latitude and longitude of farm represented as a tuple
        """
        logger.info("Initializing QC_Automation Object")
        
        self._scada_df = scada_df
        self._ws = ws_field
        self._w = power_field
        self._t = time_field
        self._id = id_field
        self._freq = freq
        self._lat_lon = lat_lon

        self._scada_sum = self._scada_df.groupby(self._scada_df[self._t])[self._w].sum().to_frame()

    @logged_method_call
    def run(self):
        """
        Run the EYA Gap analysis functions in order by calling this function.
        
        Args:
            (None)
            
        Returns:
            (None)
        """

        logger.info("Identifying Time Duplications")
        self.dup_time_identification()
        logger.info("Identifying Time Gaps")
        #self.gap_time_identification()
        logger.info("Plotting Daylight Savings Time plots")
        self.daylight_savings_plot(hour_window=3)
        logger.info("Evaluating timezone deviation from UTC")
        ws_diurnal = self.ws_diurnal_prep()
        self.wtk_diurnal_plot(ws_diurnal)
        self.corr_df_calc(ws_diurnal)
        logger.info("Isolating Extrema Values")
        self.print_max_min()
        logger.info("Plotting Turbine Power Curves")
        #self.turb_plots()
        
        
        logger.info("QC Diagnostic Complete")

    def dup_time_identification(self):
        return self._scada_df.duplicated(subset= [self._id, self._t])

    def gap_time_identification(self):
        for i in self._scada_df[self._id].unique():
            print(timeseries.find_time_gaps(self._scada_df[self._t], self._freq))

    def indicesForCoord(self,f):

        """
        This function finds the nearest x/y indices for a given lat/lon. 
        Rather than fetching the entire coordinates database, which is 500+ MB, this
        uses the Proj4 library to find a nearby point and then converts to x/y indices
        Args:
        f (h5 file): file to be read in
        """

        dset_coords = f['coordinates']
        projstring = """+proj=lcc +lat_1=30 +lat_2=60 
                    +lat_0=38.47240422490422 +lon_0=-96.0 
                    +x_0=0 +y_0=0 +ellps=sphere 
                    +units=m +no_defs """
        projectLcc = Proj(projstring)
        origin_ll = reversed(dset_coords[0][0])  # Grab origin directly from database
        origin = projectLcc(*origin_ll)

        lat, lon = self._lat_lon
        coords = (lon, lat)
        coords = projectLcc(*coords)
        delta = np.subtract(coords, origin)
        ij = [int(round(x/2000)) for x in delta]
        return tuple(reversed(ij))

    def ws_diurnal_prep(self, start_date = '2007-01-01', end_date = '2013-12-31'):

        """
        This method returns a Pandas Series corresponding to hourly average windspeeds

        Args:
        start_date(:obj:'String'): start date to diurnal analysis (optional)
        end_date(:obj:'String'): end date to diurnal analysis (optional)


        Returns:
        ws_diurnal (Pandas Series): Series where each index corresponds to a different hour of the day and each value corresponds to the average windspeed
        """

        f = h5pyd.File("/nrel/wtk-us.h5", 'r')

        # Setup date and time
        dt = f["datetime"]
        dt = pd.DataFrame({"datetime": dt[:]},index=range(0,dt.shape[0]))
        dt['datetime'] = dt['datetime'].apply(dateutil.parser.parse)

        project_idx = self.indicesForCoord(f)

        print("y,x indices for project: \t\t {}".format(project_idx))
        print("Coordinates of project: \t {}".format(self._lat_lon))
        print("Coordinates of project: \t {}".format(f["coordinates"][project_idx[0]][project_idx[1]]))

        # Get wind speed at 80m from the specified lat/lon
        ws = f['windspeed_80m']
        t_range = dt.loc[(dt.datetime >= start_date) & (dt.datetime < end_date)].index
    
        # Convert to dataframe
        ws_tseries = ws[min(t_range):max(t_range)+1, project_idx[0], project_idx[1]]
        ws_df=pd.DataFrame(index=dt.loc[t_range,'datetime'],data={'ws':ws_tseries})
    
        # Calculate diurnal profile of wind speed
        ws_diurnal=ws_df.groupby(ws_df.index.hour).mean()

        return ws_diurnal

    def wtk_diurnal_plot(self, ws_diurnal):

        """
        This method plots the WTK diurnal plot alongisde the hourly power averages of the scada_df across all turbines
        Args:
        ws_diurnal: Pandas series that contains windspeed data from the wind toolkit with hourly resolution
        Returns: None
        """

        scada_df_temp = self._scada_sum.copy()
        scada_diurnal = scada_df_temp.groupby(scada_df_temp.index.hour)[self._w].mean()

        ws_norm = ws_diurnal/ws_diurnal.mean()
        scada_norm = scada_diurnal/scada_diurnal.mean()
    
        plt.figure(figsize=(8,5))
        plt.plot(ws_norm, label = 'WTK wind speed (UTC)')
        plt.plot(scada_norm, label = 'SCADA power')
        plt.grid()
        plt.xlabel('Hour of day')
        plt.ylabel('Normalized values')
        plt.title('WTK and SCADA Timezone Comparison')
        plt.legend()
        plt.show()

    def corr_df_calc(self, ws_diurnal):
        """
        This method returns a correlation matrix that compares the current power data (with different shift thresholds) to wind speed data from the WTK with hourly resolution.

        Args:
        ws_diurnal: Pandas series that contains windspeed data from the wind toolkit with hourly resolution

        Returns:
        A correlation matrix with the correlation between the ws_diurnal data and the power data of scada_df with different degrrees of hourly shifts
        """
        scada_diurnal = self._scada_sum.groupby(self._scada_sum.index.hour)[self._w].mean()
    
        return_corr = np.empty((24))
        for i in np.arange(24):
            scada_temp = scada_diurnal.shift(i)
        
            if i != 0:
                scada_temp[np.arange(i)] = scada_diurnal[-i:]

            return_corr[i] = np.corrcoef(ws_diurnal['ws'], scada_temp)[0,1]

        s = pd.DataFrame(index = np.arange(24), data = {'Hour Shift Correlation': return_corr})
        print(s)
        return s


    def daylight_savings_plot(self, hour_window = 3):
        """
        Produce a timeseries plot showing daylight savings events for each year using the passed data.

        Args:
        hour_window(:obj: 'int'): number of hours outside of the Daylight Savings Time transitions to view in the plot

        Returns:
        None
        """
    
        # List of daylight savings days back to 2010
        dst_df = pd.read_csv('./daylight_savings.csv') 
    
        self._scada_sum['time'] = self._scada_sum.index
        df_full = timeseries.gap_fill_data_frame(self._scada_sum, 'time', self._freq) # Gap fill so spring ahead is visible
        df_full.set_index('time', inplace = True) # Have to reset index to datetime
    
        years = df_full.index.year.unique() # Years in data record
        num_years = len(years)
    
        plt.figure(figsize = (12,20))

        for y in np.arange(num_years):
            dst_data = dst_df.loc[dst_df['year'] == years[y]] 
        
            # Set spring ahead window to plot
            spring_start = pd.to_datetime(dst_data['start']) - pd.Timedelta(hours = hour_window)
            spring_end = pd.to_datetime(dst_data['start']) + pd.Timedelta(hours = hour_window)
        
            # Set fall back window to plot
            fall_start = pd.to_datetime(dst_data['end']) - pd.Timedelta(hours = hour_window)
            fall_end = pd.to_datetime(dst_data['end']) + pd.Timedelta(hours = hour_window)
        
            # Get data corresponding to each
            data_spring = df_full.loc[(df_full.index > spring_start.values[0]) & (df_full.index < spring_end.values[0])]
            data_fall = df_full.loc[(df_full.index > fall_start.values[0]) & (df_full.index < fall_end.values[0])]

            # Plot each as side-by-side subplots
            plt.subplot(num_years, 2, 2*y + 1)
            if np.sum(~np.isnan(data_spring[self._w])) > 0:
                plt.plot(data_spring[self._w])
            plt.title(years[y])
        
            plt.subplot(num_years, 2, 2*y + 2)
            if np.sum(~np.isnan(data_fall[self._w])) > 0:
                plt.plot(data_fall[self._w])
            plt.title(years[y])
    
        plt.tight_layout()
        plt.show()

    def print_max_min(self):

        """
        Prints the maximum and minimum value for each column in the DataFrame

        Args:
        None

        Returns:
        Plot of different values for int/float columns of DataFrame along with producing time-based plots for columns that are numeric-typed
        """
        
        for c in self._scada_df.columns:
            print('min', c, self._scada_df[c].min())
            print('max', c, self._scada_df[c].max())
        
        plt.figure(figsize=(12,8))

        n = 1
        for c in self._scada_df.columns:
            if (self._scada_df[c].dtype==float) | (self._scada_df[c].dtype==int):
                plt.subplot(2,2,n)
                plt.hist(self._scada_df[c].dropna(), 40)
                n = n + 1
                plt.title(c)
            plt.show()

    def turb_plots(self):

        """
        Produces plots of each individual turbine

        Args:
        None

        Returns:
        None
        """
        turbs = self._scada_df[self._id].unique()
        num_turbs = len(turbs)
        num_rows = np.ceil(num_turbs/4.)

        plt.figure(figsize = (15,num_rows * 5))
        n = 1
        for t in turbs:
            plt.subplot(num_rows, 4, n)
            scada_sub = self._scada_df.loc[self._scada_df[self._id] == t, :]
            plt.scatter(scada_sub[self._ws], scada_sub[self._ws], s = 5)
            n = n + 1
            plt.title(t)
        plt.tight_layout()
        plt.show()


    



