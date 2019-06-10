# This class defines key analytical routines for performing a 'gap-analysis'
# on EYA-estimated annual energy production (AEP) and that from operational data.
# Categories considered are availability, electrical losses, and long-term
# gross energy. The main output is a 'waterfall' plot linking the EYA-
# estimated and operational-estiamted AEP values. 

import pandas as pd
import numpy as np

#from operational_analysis.toolkits import met_data_processing
#from operational_analysis.toolkits import filters
#from operational_analysis.toolkits.power_curve import functions

from operational_analysis import logged_method_call
from operational_analysis import logging

logger = logging.getLogger(__name__)

class EYAGapAnalysis(object):
    """
    A serial (Pandas-driven) implementation of performing a gap analysis between the estimated
    annual energy production (AEP) and the actual AEP as measured using operational data.

    The method proceeds as follows:

        1. 


    """

    @logged_method_call
    def __init__(self, plant, eya_estimates, oa_results, plant_name, save_path):
        """
        Initialize turbine long-term gross energy analysis with data and parameters.

        Args:
         plant(:obj:`PlantData object`): PlantData object from which EYAGapAnalysis should draw data.

        """
        logger.info("Initializing TurbineLongTermGrossEnergy Object")
        
        # Store EYA inputs into dictionary
        self._eya_estimates = {'aep': eya_estimates[0],
                               'gross_energy': eya_estimates[1],
                               'availability_losses': eya_estimates[2],
                               'electrical_losses': eya_estimates[3],
                               'turbine_losses': eya_estimates[4],
                               'blade_degradation_losses': eya_estimates[5],
                               'wake_losses': eya_estimates[6]}
        
        # Store OA results into dictionary
        self._oa_results = {'aep': oa_results[0],
                            'availability_losses': oa_results[1],
                            'electrical_losses': oa_results[2],
                            'turbine_ideal_energy': oa_results[3]}

        # Axis labels for waterfall plot
        self._plot_index = ['eya_aep', 'ideal_energy', 'avail_loss', 'elec_loss', 'unexplained/uncertain']
        
        self._name = plant_name # Name of plant
        self._path = save_path # Where to save waterfall plot
        
        
    @logged_method_call
    def run(self):
        """
        Perform pre-processing of data into an internal representation for which the analysis can run more quickly.
        
        Args:
            (None)
            
        Returns:
            (None)
        """
        
        data = self.compile_data()
        self.waterfall_plot(data, self._plot_index, self._name, self._path )
        
        logger.info("Gap analysis complete")
        
    def compile_data(self):
        
        # Log the completion of the run
        logger.info("Compiling data")
        
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
        data = [eya_aep, turb_gross_diff, avail_diff, elec_diff, unaccounted]#, oa_plant_p50
        return data
        
    def waterfall_plot(self, data, index, plant_name, path):
        """
        Document waterfall plot
        """
        
        index = index
        
        # Store data and create a blank series to use for the waterfall
        trans = pd.DataFrame(data = {'amount': data}, index = index) # Assign gaps to data frame
        blank = trans.amount.cumsum().shift(1).fillna(0) # Perform cumulative sum on gap values
    
        # Get the net total number for the final element in the waterfall
        total = trans.sum().amount
        trans.loc["oa_aep"]= total # Add new field to gaps data frame
        blank.loc["oa_aep"] = total # Add new fiekd to cumulative sum data frame
    
        # The steps graphically show the levels as well as used for label placement
        step = blank.reset_index(drop=True).repeat(3).shift(-1)
        step[1::3] = np.nan
    
        # When plotting the last element, we want to show the full bar,
        # Set the blank to 0
        blank.loc["oa_aep"] = 0
    
        #Plot and label
        my_plot = trans.plot(kind='bar', stacked=True, bottom=blank, legend=None, 
                             figsize=(12, 6), title = plant_name + " gap analysis")
        my_plot.plot(step.index, step.values,'k')
        my_plot.set_ylabel("Energy (GWh/yr)")
    
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
        my_plot.get_figure().savefig(path + plant_name + "_waterfall.png", dpi=200, bbox_inches='tight')
    