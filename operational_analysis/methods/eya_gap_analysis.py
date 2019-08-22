# This class defines key analytical routines for performing a 'gap-analysis'
# on EYA-estimated annual energy production (AEP) and that from operational data.
# Categories considered are availability, electrical losses, and long-term
# gross energy. The main output is a 'waterfall' plot linking the EYA-
# estimated and operational-estimated AEP values. 

import pandas as pd
import numpy as np

from operational_analysis import logged_method_call
from operational_analysis import logging

logger = logging.getLogger(__name__)

class EYAGapAnalysis(object):
    """
    A serial (Pandas-driven) implementation of performing a gap analysis between the estimated
    annual energy production (AEP) from an energy yield estimate (EYA) and the actual AEP as 
    measured from an operational assessment (OA)

    The gap analysis is based on comparing the following three key metrics

        1. Availability loss
        2. Electrical loss
        3. Sum of turbine ideal energy
        
    Here turbine ideal energy is defined as the energy produced during 'normal' or 'ideal' turbine operation, 
    i.e., no downtime or considerable underperformance events. This value encompasses several different aspects 
    of an EYA (wind resource estimate, wake losses,turbine performance, and blade degradation) and in most cases
    should have the largest impact in a gap analysis relative to the first two metrics.
    
    This gap analysis method is fairly straighforward. Relevant EYA and OA metrics are passed in when defining
    the class, differences in EYA estimates and OA results are calculated, and then a 'waterfall' plot is created
    showing the differences between the EYA and OA-estimated AEP values and how they are linked from differences in 
    the three key metrics.
    
    Waterfall plot code was taken and modified from the following post: https://pbpython.com/waterfall-chart.html
    
    """ 

    @logged_method_call
    def __init__(self, plant, eya_estimates, oa_results, make_fig = True, save_fig_path = False):
        """
        Initialize EYA gap analysis class with data and parameters.

        Args:
         plant(:obj:`PlantData object`): PlantData object from which EYAGapAnalysis should draw data.
         eya_estimates(:obj:`numpy array`): Numpy array with EYA estimates listed in required order
         oa_results(:obj:`numpy array`): Numpy array with OA results listed in required order.
         make_fig(:obj:`boolean`): Indicate whether to produce the waterfall plot
         save_fig_path(:obj:`boolean` or `string'): Provide path to save waterfall plot, or set to 
                                                    False to not save plot

        """
        logger.info("Initializing EYA Gap Analysis Object")
        
        # Store EYA inputs into dictionary
        self._eya_estimates = {'aep': eya_estimates[0], # GWh/yr
                               'gross_energy': eya_estimates[1], # GWh/yr
                               'availability_losses': eya_estimates[2], # Fraction
                               'electrical_losses': eya_estimates[3], # Fraction
                               'turbine_losses': eya_estimates[4], # Fraction
                               'blade_degradation_losses': eya_estimates[5], # Fraction
                               'wake_losses': eya_estimates[6]} # Fraction
        
        # Store OA results into dictionary
        self._oa_results = {'aep': oa_results[0], # GWh/yr
                            'availability_losses': oa_results[1], # Fraction
                            'electrical_losses': oa_results[2], # Fraction
                            'turbine_ideal_energy': oa_results[3]} # Fraction

        # Axis labels for waterfall plot
        self._plot_index = ['eya_aep', 'ideal_energy', 'avail_loss', 'elec_loss', 'unexplained/uncertain']
        self._makefig = make_fig
        self._savefigpath = save_fig_path

        #Plant variable to use for plotting
        self._plant = plant
        self._data = [] # Array to hold index values for each plant
        
    @logged_method_call
    def run(self):
        """
        Run the EYA Gap analysis functions in order by calling this function.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        self._compiled_data = self.compile_data() # Compile EYA and OA data
        
        if self._makefig:
            self.waterfall_plot(self._compiled_data, self._plot_index, self._savefigpath) # Produce waterfall plot
        
        logger.info("Gap analysis complete")
        
    def compile_data(self):
        """
        Compile EYA and OA metrics, compute differences, and return data needed for waterfall plot.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        # Calculate EYA ideal turbine energy
        eya_turbine_ideal_energy = self._eya_estimates['gross_energy'] * \
                                         (1 - self._eya_estimates['turbine_losses']) * \
                                         (1 - self._eya_estimates['wake_losses']) * \
                                         (1 - self._eya_estimates['blade_degradation_losses'])
        
        # Get required gap analysis values from EYA
        eya_aep = self._eya_estimates['aep']
        eya_avail = self._eya_estimates['availability_losses']
        eya_elec = self._eya_estimates['electrical_losses']

        # Get required gap analysis values from OA
        oa_turb_ideal = self._oa_results['turbine_ideal_energy']
        oa_aep = self._oa_results['aep']
        oa_avail = self._oa_results['availability_losses']
        oa_elec = self._oa_results['electrical_losses']

        # Calculate EYA-OA differences, determine the residual or unaccounted value
        turb_gross_diff = oa_turb_ideal - eya_turbine_ideal_energy
        avail_diff = (eya_avail - oa_avail) * eya_turbine_ideal_energy
        elec_diff = (eya_elec - oa_elec) * eya_turbine_ideal_energy
        unaccounted = - (eya_aep + turb_gross_diff + avail_diff + elec_diff) + oa_aep
        
        # Combine calculations into array and return
        data = [eya_aep, turb_gross_diff, avail_diff, elec_diff, unaccounted]
        self._data = data
        return data
        
    def waterfall_plot(self, data, index, save_fig_path):
        """
        Produce a waterfall plot showing the progression from the EYA to OA estimates of AEP. 
        
        Args:
            data(:obj:`numpy array`): data to be used to create waterfall plot
            index(:obj:`list`): List of string values to be used for x-axis labels
            path(:obj:`string`): Location to save waterfall plot
            
        Returns:
            (None)
        """
        
        # Store data and create a blank series to use for the waterfall
        trans = pd.DataFrame(data = {'amount': data}, index = index) # Assign gaps to data frame
        blank = trans.amount.cumsum().shift(1).fillna(0) # Perform cumulative sum on gap values
    
        # Get the net total number for the final element in the waterfall
        total = trans.sum().amount
        trans.loc["oa_aep"]= total # Add new field to gaps data frame
        blank.loc["oa_aep"] = total # Add new field to cumulative sum data frame
    
        # The steps graphically show the levels as well as used for label placement
        step = blank.reset_index(drop=True).repeat(3).shift(-1)
        step[1::3] = np.nan
    
        # When plotting the last element, we want to show the full bar,
        # Set the blank to 0
        blank.loc["oa_aep"] = 0
    
        #Plot and label
        my_plot = trans.plot(kind='bar', stacked=True, bottom=blank, legend=None, 
                             figsize=(12, 6))
        my_plot.plot(step.index, step.values,'k')
        my_plot.set_ylabel("Energy (GWh/yr)")
        my_plot.set_title(self._plant)
    
        #Get the y-axis position for the labels
        y_height = trans.amount.cumsum().shift(1).fillna(0)
    
        # Get an offset so labels don't sit right on top of the bar
        mx = trans.max() # Max value in gap analysis values
        neg_offset = mx / 25
        pos_offset = mx / 50
        
        # Add labels to each bar
        loop = 0
        for index, row in trans.iterrows():
            # For the last item in the list, we don't want to double count
            if row['amount'] == total:
                y = y_height[loop]
            else:
                y = y_height[loop] + row['amount']
                
            # Determine if we want a neg or pos offset
            if row['amount'] > 0:
                y += pos_offset
            else:
                y -= neg_offset
            my_plot.annotate("{:,.0f}".format(row['amount']),(loop,y),ha="center")
            loop += 1
    
        # Adjust y-axis to focus on region of interest
        plt_min = blank[1:-1].min() # Min value in cumulative sum values
        plt_max = blank[1:].max() # Min value in cumulative sum values
        my_plot.set_ylim(0.9 * plt_min, 1.1 * plt_max)#blank.max()+int(plot_offset))
        
        #Rotate the labels
        my_plot.set_xticklabels(trans.index,rotation=0)
 
        # Save figure
        if save_fig_path != False:
            my_plot.get_figure().savefig(save_fig_path + "/waterfall.png", dpi=200, bbox_inches='tight')
            
        return my_plot
    