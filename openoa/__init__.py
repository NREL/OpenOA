__version__ = "3.0rc1"
"""
When bumping version, please be sure to also update parameters in sphinx/conf.py
"""

from types import MethodType

from openoa.plant import PlantData
from openoa.analysis import (
    WakeLosses,
    MonteCarloAEP,
    EYAGapAnalysis,
    ElectricalLosses,
    TurbineLongTermGrossEnergy,
)


# API Shortcuts


# Attach analysis classes to PlantData
setattr(PlantData, "WakeLosses", classmethod(WakeLosses))
setattr(PlantData, "MonteCarloAEP", classmethod(MonteCarloAEP))
setattr(PlantData, "EYAGapAnalysis", classmethod(EYAGapAnalysis))
setattr(PlantData, "ElectricalLosses", classmethod(ElectricalLosses))
setattr(PlantData, "TurbineLongTermGrossEnergy", classmethod(TurbineLongTermGrossEnergy))


# TODO: Add analysis results computation methods to PlantData
def gap_analysis(project: PlantData, eya_estimates: dict, oa_results: dict):
    gap = EYAGapAnalysis(project, eya_estimates, oa_results)
    gap.run()
    return gap


setattr(PlantData, "gap_analysis", classmethod(gap_analysis))
