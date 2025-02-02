"""
Canopy gas exchange model class: Includes equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from daesim_optimised.canopylayers import CanopyLayers
from daesim_optimised.canopyradiation import CanopyRadiation
from daesim_optimised.leafgasexchange2 import LeafGasExchangeModule2

@define
class CanopyGasExchange:

    ## Module dependencies
    Leaf: Callable = field(default=LeafGasExchangeModule2())    ## It is optional to define PlantDev for this method. If no argument is passed in here, then default setting for Leaf is the default LeafGasExchangeModule()
    Canopy: Callable = field(default=CanopyLayers())    ## It is optional to define PlantDev for this method. If no argument is passed in here, then default setting for Canopy is the default CanopyLayers()
    CanopyRad: Callable = field(default=CanopyRadiation())    ## It is optional to define CanopyRad for this method. If no argument is passed in here, then default setting for CanopyRad is the default CanopyRadiation()

    def calculate(
        self,
        leafTempC,    ## Leaf temperature, degrees Celsius
        airCO2,  ## leaf surface CO2 partial pressure, bar, (corrected for boundary layer effects)
        airO2,   ## leaf surface O2 partial pressure, bar, (corrected for boundary layer effects)
        airRH,   ## Relative humidity, %
        fgsw, ## Leaf water potential limitation factor on stomatal conductance, unitless
        LAI,    ## Leaf area index, m2/m2
        SAI,    ## Stem area index, m2/m2
        clumping_factor,  ## Foliage clumping index (-)
        z,      ## Canopy height, m
        sza,    ## Solar zenith angle, degrees
        swskyb, ## Atmospheric direct beam solar radiation, W/m2
        swskyd, ## Atmospheric diffuse solar radiation, W/m2
    ) -> Tuple[float]:

        swleaf = self.CanopyRad.calculate(LAI,SAI,clumping_factor,z,sza,swskyb,swskyd)

        # Calculate gas exchange per canopy multi-layer element (per leaf area basis)
        An_mle_l = np.zeros((self.Canopy.nlevmlcan, self.Canopy.nleaf))
        gs_mle_l = np.zeros((self.Canopy.nlevmlcan, self.Canopy.nleaf))
        Rd_mle_l = np.zeros((self.Canopy.nlevmlcan, self.Canopy.nleaf))
        
        # Vectorized calculation of Q (absorbed PPFD, mol PAR m-2 s-1) for all layers and leaves
        Q = 1e-6 * swleaf[self.Canopy.nbot:self.Canopy.ntop+1, :] * self.CanopyRad.J_to_umol  # absorbed PPFD, mol PAR m-2 s-1
        
        # Vectorized calculation of An, gs, Rd for all layers and leaves
        An, gs, Ci, Vc, Ve, Vs, Rd = self.Leaf.calculate(Q, leafTempC, airCO2, airO2, airRH, fgsw)
        
        # Assign results to the respective arrays
        An_mle_l[self.Canopy.nbot:self.Canopy.ntop+1, :] = An
        gs_mle_l[self.Canopy.nbot:self.Canopy.ntop+1, :] = gs
        Rd_mle_l[self.Canopy.nbot:self.Canopy.ntop+1, :] = Rd

        # Determine leaf area index (LAI) per canopy layer based on prescribed distribution
        dlai = self.Canopy.cast_parameter_over_layers_betacdf(LAI,self.Canopy.beta_lai_a,self.Canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)

        # Determine sunlit and shaded fractions for canopy layers
        (fracsun, kb, omega, avmu, betab, betad, tbi) = self.CanopyRad.calculateRTProperties(LAI,SAI,clumping_factor,z,sza)

        ## Calculate gas exchange per canopy layer (per ground area basis)
        ## - multiply the sunlit rates (per leaf area) by the sunlit lai, and the shaded rates (per leaf area) by the shaded lai
        An_ml = dlai*fracsun*An_mle_l[:,self.Canopy.isun] + dlai*(1-fracsun)*An_mle_l[:,self.Canopy.isha]
        gs_ml = dlai*fracsun*gs_mle_l[:,self.Canopy.isun] + dlai*(1-fracsun)*gs_mle_l[:,self.Canopy.isha]
        Rd_ml = dlai*fracsun*Rd_mle_l[:,self.Canopy.isun] + dlai*(1-fracsun)*Rd_mle_l[:,self.Canopy.isha]

        return An_ml, gs_ml, Rd_ml, An_mle_l, gs_mle_l, Rd_mle_l, fracsun
