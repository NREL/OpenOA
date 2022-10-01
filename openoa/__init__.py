__version__ = "3.0.0"
"""
When bumping version, please be sure to also update parameters in sphinx/conf.py
"""


## API Shortcuts

from openoa.plant import PlantData
import openoa.analysis

PlantData.MonteCarloAEP = openoa.analysis.aep.MonteCarloAEP
