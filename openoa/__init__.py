__version__ = "3.0rc1"
"""
When bumping version, please be sure to also update parameters in sphinx/conf.py
"""


# API Shortcuts

import openoa.analysis
from openoa.plant import PlantData


PlantData.MonteCarloAEP = openoa.analysis.aep.MonteCarloAEP
