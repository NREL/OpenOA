
Import the Data
---------------

This notebook will explore the usage of different toolkit functions to
filter data and compute power curves from a single wind turbine. We
start the notebook by importing a project file (turbine\_project.py, in
this case), instantiating a project, and calling its prepare method.
This process provides us with a PlantData object (turbine, in this case)
which can be used to query data.

.. code:: ipython2

    from turbine_project import TurbineExampleProject
    
    turbine = TurbineExampleProject()
    turbine.prepare()
    
    display(turbine.scada.head())


.. parsed-literal::

    INFO:operational_analysis.types.timeseries_table:Loading name:scada_10min_4cols



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>wmet_wdspd_avg</th>
          <th>wmet_wDir_avg</th>
          <th>wtur_W_avg</th>
          <th>time</th>
          <th>id</th>
          <th>energy_kwh</th>
        </tr>
        <tr>
          <th>time</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2010-02-17 19:30:00</th>
          <td>0.63</td>
          <td>0.0</td>
          <td>-2.07</td>
          <td>2010-02-17 19:30:00</td>
          <td>T0</td>
          <td>-0.345000</td>
        </tr>
        <tr>
          <th>2010-02-17 19:40:00</th>
          <td>1.81</td>
          <td>0.0</td>
          <td>-2.24</td>
          <td>2010-02-17 19:40:00</td>
          <td>T0</td>
          <td>-0.373333</td>
        </tr>
        <tr>
          <th>2010-02-17 19:50:00</th>
          <td>2.13</td>
          <td>0.0</td>
          <td>-3.10</td>
          <td>2010-02-17 19:50:00</td>
          <td>T0</td>
          <td>-0.516667</td>
        </tr>
        <tr>
          <th>2010-02-17 20:00:00</th>
          <td>1.33</td>
          <td>0.0</td>
          <td>-1.91</td>
          <td>2010-02-17 20:00:00</td>
          <td>T0</td>
          <td>-0.318333</td>
        </tr>
        <tr>
          <th>2010-02-17 20:10:00</th>
          <td>0.63</td>
          <td>0.0</td>
          <td>-1.89</td>
          <td>2010-02-17 20:10:00</td>
          <td>T0</td>
          <td>-0.315000</td>
        </tr>
      </tbody>
    </table>
    </div>


PlantData objects contain references to several TimeseriesTable objects.
The only one which is useful for this demo is "scada." The "df" member
of the scada TimeseriesTable exposes a Pandas dataframe which contains
cleaned SCADA data. Let's use turbine.scada to define python variables
windspeed, power\_kw, and df for convenience in later examples.

.. code:: ipython2

    windspeed = turbine.scada.df["wmet_wdspd_avg"]
    power_kw = turbine.scada.df["wtur_W_avg"]
    df = turbine.scada.df

Filter the Data
---------------

The filtering toolkit implements commonly used filters and operated by
returning a boolean series called "flag" which can be used to index the
original dataframe. We start by importing the filtering toolkit:

.. code:: ipython2

    from operational_analysis.toolkits import filters

First let's define a plotting function for the turbine power curve which
we can use to examine the effect of different filtering functions on the
data

.. code:: ipython2

    import numpy as np
    import matplotlib.pyplot as plt

.. code:: ipython2

    def plot_flagged_pc(ws, p, flag_bool, alpha):
        plt.scatter(ws, p, s = 1, alpha = alpha)
        plt.scatter(ws[flag_bool], p[flag_bool], s = 1, c = 'red')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('Power (kW)')
        plt.show()

Now first let's take a look at the unprocessed data:

.. code:: ipython2

    plot_flagged_pc(windspeed, power_kw, np.repeat('True', df.shape[0]), 1)



.. image:: Turbine_Toolkit_Examples_files/Turbine_Toolkit_Examples_10_0.png


Range Filter
~~~~~~~~~~~~

We immediately see two high wind speed outliers likely due to sensor
malfunction. We can flag and filter these outliers from the dataset
using the 'range\_flag' function:

.. code:: ipython2

    # Show outliers
    out_of_range = filters.range_flag(windspeed, below=0, above=70)
    windspeed[out_of_range].head()




.. parsed-literal::

    time
    2017-02-03 20:00:00    89.79
    2017-02-03 20:10:00    86.67
    Name: wmet_wdspd_avg, dtype: float64



Let's remove the outliers and plot the result:

.. code:: ipython2

    # Remove outliers
    windspeed = windspeed[~out_of_range]
    power_kw = power_kw[~out_of_range]

.. code:: ipython2

    # Show updated power curve
    plot_flagged_pc(windspeed, power_kw, np.repeat('True', df.shape[0]), 0.2)



.. image:: Turbine_Toolkit_Examples_files/Turbine_Toolkit_Examples_15_0.png


Window Range Filter
~~~~~~~~~~~~~~~~~~~

We may want to remove turbine downtime events, which we can identify as
near zero power at high wind speeds. We can do this using the
'window\_range function' and removing data greater than 6 m/s but with
power less than 20 kW:

.. code:: ipython2

    out_of_window = filters.window_range_flag(windspeed, 6., 40, power_kw, 20., 2000.)
    plot_flagged_pc(windspeed, power_kw, out_of_window, 0.2)



.. image:: Turbine_Toolkit_Examples_files/Turbine_Toolkit_Examples_17_0.png


Again, let's remove these flagged data from consideration:

.. code:: ipython2

    windspeed = windspeed[~out_of_window]
    power_kw = power_kw[~out_of_window]

Bin Filter
~~~~~~~~~~

We may be interested in fitting a power curve to data representing
'normal' turbine operation. In other words, we want to flag all
anomalous data or data represenatative of underperformance. To do this,
the 'bin\_filter' function is useful. It works by binning the data by a
specified variable, bin width, and start and end points. The criteria
for flagging is based on some measure (scalar or standard deviation)
from the mean or median of the bin center.

As an example, let's bin on power in 100 kW increments, starting from
25.0 kW but stopping at 90% of peak power (i.e. we don't want to flag
all the data at peak power and high wind speed. Let's use a scalar
threshold of 1.5 m/s from the median for each bin. Let's also consider
data on both sides of the curve by setting the 'direction' parameter to
'all'

.. code:: ipython2

    max_bin = 0.90*power_kw.max()
    bin_outliers = filters.bin_filter(power_kw, windspeed, 100, 1.5, 'median', 20., max_bin, 'scalar', 'all')
    plot_flagged_pc(windspeed, power_kw, bin_outliers, 0.5)



.. image:: Turbine_Toolkit_Examples_files/Turbine_Toolkit_Examples_21_0.png


As seen above, one call for the bin filter has done a decent job of
cleaning up the power curve to represent 'normal' operation, without
excessive removal of data points. There are a few points at peak power
but low wind speed that weren't flagged, however. Let catch those, and
then remove those as well as the flagged data above, and plot our
'clean' power curve

.. code:: ipython2

    out_of_window = filters.window_range_flag(windspeed, 4., 8., power_kw, 0., 1250.)
    windspeed = windspeed[(~out_of_window) & (~bin_outliers)]
    power_kw = power_kw[(~out_of_window) & (~bin_outliers)]
    plot_flagged_pc(windspeed, power_kw, np.repeat('True', len(windspeed)), 0.1)



.. image:: Turbine_Toolkit_Examples_files/Turbine_Toolkit_Examples_23_0.png


The bin filter gives a bit of a 'choppy' look to the power curve at
lower wind speeds, which we could clean up by using a smaller bin size.

Unresponsive Filter
~~~~~~~~~~~~~~~~~~~

As a final filtering demonstration, we can look for an unrespsonsive
sensor (i.e. repeating measurements). In this case, let's look for 3 or
more repeating wind speed measurements:

.. code:: ipython2

    frozen = filters.unresponsive_flag(windspeed, 3)
    windspeed[frozen].head()




.. parsed-literal::

    time
    2010-02-18 02:10:00    0.63
    2010-02-18 02:20:00    0.63
    2010-02-18 02:30:00    0.63
    2010-02-18 02:40:00    0.63
    2010-02-18 02:50:00    0.63
    Name: wmet_wdspd_avg, dtype: float64



We actually found a lot, so let's remove these data as well before
moving on to power curve fitting.

.. code:: ipython2

    windspeed = windspeed[~frozen]
    power_kw = power_kw[~frozen]

Fit the Power Curves
--------------------

Now that our data has been reasonably processed, we can fit a power
curve model to the data. Here we illustrate two types of power curves:
the standard IEC binned power curve model and a Logistic 5 parameter
model (L5P):

.. code:: ipython2

    from operational_analysis.toolkits import power_curve

.. code:: ipython2

    # Fit the power curves
    iec_curve = power_curve.IEC(windspeed, power_kw)
    l5p_curve = power_curve.logistic_5_parametric(windspeed, power_kw)

.. code:: ipython2

    # Plot the results
    x = np.linspace(0,20,100)
    plt.figure(figsize = (10,6))
    plt.scatter(windspeed, power_kw, alpha=0.1, s = 1, c = 'gray')
    plt.plot(x, iec_curve(x), color="red", label = 'IEC', linewidth = 3)
    plt.plot(x, l5p_curve(x), color="C2", label = 'L5P', linewidth = 3)
    plt.xlabel('Wind speed (m/s)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.show()


.. parsed-literal::

    /Users/jperrsau/src/pruf/openoa-public/OpenOA/operational_analysis/toolkits/power_curve/parametric_forms.py:35: RuntimeWarning: divide by zero encountered in power
      return d + (a - d) / (1 + (x / c) ** b) ** g



.. image:: Turbine_Toolkit_Examples_files/Turbine_Toolkit_Examples_32_1.png


The above plot shows that the IEC method accurately captures the power
curve, although it results in a 'choppy' fit, while the L5P model
(constrained by its parametric form) deviates from the knee of the power
curve through peak production.

Relative Speed of Power Curve Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also note the speed of the computations: The IEC method is by far the
fastest.

.. code:: ipython2

    import time

.. code:: ipython2

    start = time.time()
    power_curve.IEC(windspeed, power_kw)
    end = time.time()
    print 'IEC: ' + str(np.round(end - start, 3)) + ' seconds'


.. parsed-literal::

    IEC: 0.123 seconds


.. code:: ipython2

    start = time.time()
    power_curve.logistic_5_parametric(windspeed, power_kw)
    end = time.time()
    print 'L5P: ' + str(np.round(end - start, 3)) + ' seconds'


.. parsed-literal::

    L5P: 49.02 seconds


.. code:: ipython2

    l5p_curve(x)




.. parsed-literal::

    array([1.00000000e-03, 1.03123632e-03, 1.79422804e-03, 6.27210929e-03,
           2.11943700e-02, 5.82308826e-02, 1.35050059e-01, 2.76288292e-01,
           5.14453280e-01, 8.90773009e-01, 1.45599541e+00, 2.27113815e+00,
           3.40818239e+00, 4.95069820e+00, 6.99438183e+00, 9.64747646e+00,
           1.30310380e+01, 1.72789955e+01, 2.25379439e+01, 2.89665923e+01,
           3.67347782e+01, 4.60219483e+01, 5.70149966e+01, 6.99053523e+01,
           8.48852179e+01, 1.02142884e+02, 1.21857085e+02, 1.44190431e+02,
           1.69282026e+02, 1.97239502e+02, 2.28130815e+02, 2.61976280e+02,
           2.98741419e+02, 3.38331310e+02, 3.80587077e+02, 4.25285177e+02,
           4.72139894e+02, 5.20809269e+02, 5.70904356e+02, 6.22001336e+02,
           6.73655735e+02, 7.25417749e+02, 7.76847546e+02, 8.27529466e+02,
           8.77084163e+02, 9.25178039e+02, 9.71529587e+02, 1.01591259e+03,
           1.05815644e+03, 1.09814396e+03, 1.13580732e+03, 1.17112258e+03,
           1.20410358e+03, 1.23479541e+03, 1.26326807e+03, 1.28961059e+03,
           1.31392560e+03, 1.33632475e+03, 1.35692474e+03, 1.37584413e+03,
           1.39320080e+03, 1.40910994e+03, 1.42368265e+03, 1.43702492e+03,
           1.44923691e+03, 1.46041265e+03, 1.47063980e+03, 1.47999972e+03,
           1.48856758e+03, 1.49641256e+03, 1.50359820e+03, 1.51018267e+03,
           1.51621912e+03, 1.52175606e+03, 1.52683769e+03, 1.53150425e+03,
           1.53579231e+03, 1.53973516e+03, 1.54336301e+03, 1.54670331e+03,
           1.54978102e+03, 1.55261878e+03, 1.55523717e+03, 1.55765489e+03,
           1.55988892e+03, 1.56195472e+03, 1.56386633e+03, 1.56563653e+03,
           1.56727696e+03, 1.56879821e+03, 1.57020994e+03, 1.57152095e+03,
           1.57273928e+03, 1.57387225e+03, 1.57492657e+03, 1.57590835e+03,
           1.57682320e+03, 1.57767623e+03, 1.57847214e+03, 1.57921524e+03])



