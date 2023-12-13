"""
Management model class: Includes differential equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field

@define
class ManagementModule:
    """
    Calculator of management
    """

    # Class parameters
    plantingDay: float = field(default=30)  ## day that planting (sowing) occurs, in units of ordinal day of year (DOY)
    harvestDay: float = field(default=235)  ## day that harvest occurs, in units of ordinal day of year (DOY)
    frequPlanting: float = field(default=0)  ## frequency of planting (days-1)
    propPhPlanting: float = field(default=0)  ## fraction of planted biomass that is photosynthetic (almost always equal to 0, as it is seeds that are planted, which have no photosynthetic biomass. Although, this parameter allows planting of seedlings which have some photosynthetic biomass as soon as they're planted). Modification: This variable was previously defined using "frequPlanting", which didn't match with the units or its definition.
    maxDensity: float = field(default=40)  ## number of individual plants per m2
    plantingRate: float = field(default=40)  ## number of individual plants per m2 per day (# plants m-2 d-1)
    plantWeight: float = field(default=0.0009)  ## mass of individual plant at sowing (Question: units? kg? g?)
    propPhHarvesting: float = field(default=0.3)  ## proportion of Photosynthetic_Biomass harvested
    propNPhHarvest: float = field(default=0.4)  ## proportion of Non_Photosynthetic_Biomass harvested
    PhHarvestTurnoverTime: float = field(default=1)  ## Turnover time (days). Modification: This is a new parameter required to run in this framework. It does not exist in the Stella code, but it is needed as a replacement for "DT"
    NPhHarvestTurnoverTime: float = field(default=1)  ## Turnover time (days). Modification: This is a new parameter required to run in this framework. It does not exist in the Stella code, but it is needed as a replacement for "DT"
    propTillage: float = field(default=0.5)  ## Management module: propTillage=intensityTillage/10 (ErrorCheck: in the Stella code propTillage=intensityTillage/10 but in the documentation propTillage=(intensityTillage*9)/(5*10), why?)
    propHarvPhLeft: float = field(default=0.1)  ## the percentage of photosynthetic biomass left after harvesting
    propHarvNPhLeft: float = field(default=0.8)  ## the percentage of non-photosynthetic biomass left after harvesting