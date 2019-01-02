#from .parametric_forms import *
#from .parametric_optimize import *
#from scipy.optimize import differential_evolution
#from pygam import LinearGAM
#import pandas as pd
import numpy as np

from scipy.stats import randint as sp_randint

"""
This module is a library of hyperparameters available for different 
learning algorithms. This library would be employed during a machine-learning pipeline
where an analyst would optimize hyperparmeters during cross-validation
"""

extra_trees_hyp = {"max_depth": [4, 8, 12, 16, 20, None],
                   "min_samples_split": sp_randint(2, 11),
                   "min_samples_leaf": sp_randint(1, 11),
                   "n_estimators": np.arange(10,801,40)}

gam_hyp = {'n_splines': np.arange(5,40)}

