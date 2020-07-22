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

class WindToolKitQualityControlDiagnosticSuite(object):

    """This class defines key analytical procedures in a quality check process for turbine data.
    After analyzing the data for missing and duplicate timestamps, timezones, Daylight Savings Time corrections, and extrema values,
    the user can make informed decisions about how to handle the data.
    """

    @logged_method_call
    def __init__(self,df, ws_field='wmet_wdspd_avg', power_field= 'wtur_W_avg', time_field= 'datetime',
                 id_field= None, freq = '10T', lat_lon= (0,0), dst_subset = 'American',
                 check_tz = False):
        """
        Initialize QCAuto object with data and parameters.

        Args:
         df(:obj:`DataFrame object`): DataFrame object that contains data
         ws_field(:obj: 'String'): String name of the windspeed field to df
         power_field(:obj: 'String'): String name of the power field to df
         time_field(:obj: 'String'): String name of the time field to df
         id_field(:obj: 'String'): String name of the id field to df
         freq(:obj: 'String'): String representation of the resolution for the time field to df
         lat_lon(:obj: 'tuple'): latitude and longitude of farm represented as a tuple
         dst_subset(:obj: 'String'): Set of Daylight Savings Time transitions to use (currently American or France)
         check_tz(:obj: 'boolean'): Boolean on whether to use WIND Toolkit data to assess timezone of data
        """
        logger.info("Initializing QC_Automation Object")

        self._df = df
        self._ws = ws_field
        self._w = power_field
        self._t = time_field
        self._id = id_field
        self._freq = freq
        self._lat_lon = lat_lon
        self._dst_subset = dst_subset
        self._check_tz = check_tz

        if self._id==None:
            self._id = 'ID'
            self._df['ID'] = 'Data'

    @logged_method_call
    def run(self):
        """
        Run the QC analysis functions in order by calling this function.

        Args:
            (None)

        Returns:
            (None)
        """

        logger.info("Identifying Time Duplications")
        self.dup_time_identification()
        logger.info("Identifying Time Gaps")
        self.gap_time_identification()
        logger.info('Grabbing DST Transition Times')
        self.create_dst_df()

        if self._check_tz:
            logger.info("Evaluating timezone deviation from UTC")
            self.ws_diurnal_prep()
            self.corr_df_calc()

        logger.info("Isolating Extrema Values")
        self.max_min()
        logger.info("QC Diagnostic Complete")


    def dup_time_identification(self):
        """
        This function identifies any time duplications in the dataset.

        Args:
        (None)

        Returns:
        (None)
        """
        self._time_duplications = self._df.loc[self._df.duplicated(subset= [self._id, self._t]), self._t]

    def gap_time_identification(self):
        """
        This function identifies any time gaps in the dataset.

        Args:
        (None)

        Returns:
        (None)
        """
        self._time_gaps = timeseries.find_time_gaps(self._df[self._t], freq=self._freq)

    def indicesForCoord(self,f):

        """
        This function finds the nearest x/y indices for a given lat/lon.
        Rather than fetching the entire coordinates database, which is 500+ MB, this
        uses the Proj4 library to find a nearby point and then converts to x/y indices.
        This function relies on the Wind Toolkit HSDS API.
        
        Args:
            f (h5 file): file to be read in
        
        Returns:
            x and y coordinates corresponding to a given lat/lon as a tuple
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
        This method links into Wind Toolkit data on AWS as a data source, grabs wind speed data, and calculates diurnal hourly averages.
        These diurnal hourly averages are returned as a Pandas series.

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

        self._wtk_ws_diurnal= ws_diurnal

    def wtk_diurnal_plot(self):

        """
        This method plots the WTK diurnal plot alongisde the hourly power averages of the df across all turbines
        
        Args:
            (None)
        
        Returns:
            (None)
        """

        sum_df = self._df.groupby(self._df[self._t])[self._w].sum().to_frame()
        #df_temp = sum_df.copy()
        #df_temp[self._t] = df_temp.index

        #df_diurnal = df_temp.groupby(df_temp[self._t].dt.hour)[self._w].mean()
        df_diurnal = sum_df.groupby(sum_df.index.hour)[self._w].mean()

        ws_norm = self._wtk_ws_diurnal/self._wtk_ws_diurnal.mean()
        df_norm = df_diurnal/df_diurnal.mean()

        plt.figure(figsize=(8,5))
        plt.plot(ws_norm, label = 'WTK wind speed (UTC)')
        plt.plot(df_norm, label = 'QC power')
        plt.grid()
        plt.xlabel('Hour of day')
        plt.ylabel('Normalized values')
        plt.title('WTK and QC Timezone Comparison')
        plt.legend()
        plt.show()

    def corr_df_calc(self):
        """
        This method creates a correlation series that compares the current power data (with different shift thresholds) to wind speed data from the WTK with hourly resolution.

        Args:
            (None)

        Returns:
            (None)
        """

        self._df_diurnal = self._df.groupby(self._df[self._t].dt.hour)[self._w].mean()
        return_corr = np.empty((24))

        for i in np.arange(24):
            return_corr[i] = np.corrcoef(self._wtk_ws_diurnal['ws'], np.roll(self._df_diurnal,i))[0,1]

        self._hour_shift = pd.DataFrame(index = np.arange(24), data = {'corr_by_hour': return_corr})


    def create_dst_df(self):
        if self._dst_subset == 'American':
        # American DST Transition Dates (Local Time)
            self._dst_dates = pd.DataFrame()
            self._dst_dates['year'] =  [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
            self._dst_dates['start'] = ['3/9/08 2:00', '3/8/09 2:00', '3/14/10 2:00', '3/13/11 2:00', '3/11/12 2:00', '3/10/13 2:00', '3/9/14 2:00', '3/8/15 2:00', '3/13/16 2:00', '3/12/17 2:00', '3/11/18 2:00' , '3/10/19 2:00']
            self._dst_dates['end'] = ['11/2/08 2:00', '11/1/09 2:00', '11/7/10 2:00', '11/6/11 2:00', '11/4/12 2:00', '11/3/13 2:00', '11/2/14 2:00', '11/1/15 2:00', '11/6/16 2:00', '11/5/17 2:00', '11/4/18 2:00', '11/3/19 2:00']
        else:
            # European DST Transition Dates (Local Time)
            self._dst_dates = pd.DataFrame()
            self._dst_dates['year'] = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
            self._dst_dates['start'] = ['3/30/08 2:00', '3/29/09 2:00', '3/28/10 2:00', '3/27/11 2:00', '3/25/12 2:00', '3/31/13 2:00', '3/30/14 2:00', '3/29/15 2:00', '3/27/16 2:00', '3/26/17 2:00', '3/25/18 2:00' , '3/31/19 2:00']
            self._dst_dates['end'] = ['10/26/08 3:00', '10/25/09 3:00', '10/31/10 3:00', '10/30/11 3:00', '10/28/12 3:00', '10/27/13 3:00', '10/26/14 3:00', '10/25/15 3:00', '10/30/16 3:00', '10/29/17 3:00', '10/28/18 3:00', '10/27/19 3:00']

    def daylight_savings_plot(self, hour_window = 3):

        """
        Produce a timeseries plot showing daylight savings events for each year using the passed data.

        Args:
            hour_window(:obj: 'int'): number of hours outside of the Daylight Savings Time transitions to view in the plot (optional)

        Returns:
            (None)
        """

        self._df_dst =  self._df.loc[self._df[self._id]==self._df[self._id].unique()[0], :]

        df_full = timeseries.gap_fill_data_frame(self._df_dst, self._t, self._freq) # Gap fill so spring ahead is visible
        df_full.set_index(self._t, inplace=True)
        self._df_full = df_full

        years = df_full.index.year.unique() # Years in data record
        num_years = len(years)

        plt.figure(figsize = (12,20))

        for y in np.arange(num_years):
            dst_data = self._dst_dates.loc[self._dst_dates['year'] == years[y]]

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
            plt.title(str(years[y]) + ', Spring')
            plt.ylabel('Power')
            plt.xlabel('Date')

            plt.subplot(num_years, 2, 2*y + 2)
            if np.sum(~np.isnan(data_fall[self._w])) > 0:
                plt.plot(data_fall[self._w])
            plt.title(str(years[y]) + ', Fall')
            plt.ylabel('Power')
            plt.xlabel('Date')

        plt.tight_layout()
        plt.show()

    def max_min(self):

        """
        This function creates a DataFrame that contains the max and min values for each column

        Args:
        (None)

        Returns:
        (None)
        """

        self._max_min = pd.DataFrame(index = self._df.columns, columns = {'max', 'min'})
        self._max_min['max'] = self._df.max()
        self._max_min['min'] = self._df.min()

    def plot_by_id(self, x_axis = None, y_axis = None):

        """
        This is generalized function that allows the user to plot any two fields against each other with unique plots for each unique ID.
        For scada data, this function produces turbine plots and for meter data, this will return a single plot.

        Args:
            x_axis(:obj:'String'): Independent variable to plot (default is windspeed field)
            y_axis(:obj:'String'): Dependent variable to plot (default is power field)

        Returns:
            (None)
        """
        if x_axis is None:
            x_axis = self._ws

        if y_axis is None:
            y_axis = self._w

        turbs = self._df[self._id].unique()
        num_turbs = len(turbs)
        num_rows = np.ceil(num_turbs/4.)

        plt.figure(figsize = (15,num_rows * 5))
        n = 1
        for t in turbs:
            plt.subplot(num_rows, 4, n)
            scada_sub = self._df.loc[self._df[self._id] == t, :]
            plt.scatter(scada_sub[x_axis], scada_sub[y_axis], s = 5)
            n = n + 1
            plt.title(t)
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
        plt.tight_layout()
        plt.show()

    def column_histograms(self):
        """
        Produces histogram plot for each numeric column.
        
        Args:
            (None)

        Returns:
            (None)
        """

        for c in self._df.columns:
            if (self._df[c].dtype==float) | (self._df[c].dtype==int):
                #plt.subplot(2,2,n)
                plt.figure(figsize=(8,6))
                plt.hist(self._df[c].dropna(), 40)
                #n = n + 1
                plt.title(c)
                plt.ylabel('Count')
                plt.show()
