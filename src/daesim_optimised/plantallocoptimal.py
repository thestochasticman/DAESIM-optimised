"""
Plant optimal trajectory carbon allocation model class: Includes equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from daesim_optimised.plantcarbonwater import PlantModel as PlantCH2O


@define 
class PlantOptimalAllocation:
    """
    Calculator of plant allocation based on optimal trajectory principle
    """

    ## Module dependencies
    Plant: Callable = field(default=PlantCH2O())    ## It is optional to define Plant for this method. If no argument is passed in here, then default setting for Plant is the default PlantModel().

    ## Class parameters
    
    ## Biomass pool step size (defined as a factor or 'multiplier') for evaluating marginal gain and marginal cost with finite difference method
    dWL_factor: float = field(default=1.01)    ## Step size for leaf biomass pool
    dWR_factor: float = field(default=1.01)    ## Step size for leaf biomass pool

    ## Allocation coefficients that are not optimal
    u_Stem: float = field(default=0.0)    ## Carbon allocation coefficient to stem
    u_Seed: float = field(default=0.0)    ## Carbon allocation coefficient to seed

    ## Turnover rates (carbon pool lifespan)
    tr_L: float = field(default=0.01)    ## Turnover rate of leaf biomass (days-1)
    tr_R: float = field(default=0.01)    ## Turnover rate of root biomass (days-1)

    def calculate(
        self,
        W_L,         ## leaf structural dry biomass (g d.wt m-2)
        W_R,         ## root structural dry biomass (g d.wt m-2)
        soilTheta,   ## volumetric soil water content (m3 m-3)
        leafTempC,   ## leaf temperature (deg C)
        airTempC,    ## air temperature (deg C), outside leaf boundary layer 
        airRH,      ## relative humidity of air (%), outside leaf boundary layer
        airCO2,  ## leaf surface CO2 partial pressure, bar, (corrected for boundary layer effects)
        airO2,   ## leaf surface O2 partial pressure, bar, (corrected for boundary layer effects)
        airP,    ## air pressure, Pa, (in leaf boundary layer)
        airUhc,  ## wind speed at top-of-canopy, m s-1
        swskyb, ## Atmospheric direct beam solar radiation, W/m2
        swskyd, ## Atmospheric diffuse solar radiation, W/m2
        sza,    ## Solar zenith angle, degrees
        SAI,    ## Stem area index, m2 m-2
        CI,     ## Foliage clumping index, -
        hc,     ## Canopy height, m
        d_r,    ## Root depth, m
    ) -> Tuple[float]:

        ## Calculate control run
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = self.Plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        
        ## Calculate sensitivity run for leaf biomass
        GPP_L, Rml_L, Rmr_L, E_L, f_Psil_L, Psil_L, Psir_L, Psis_L, K_s_L, K_sr_L, k_srl_L = self.Plant.calculate(W_L*self.dWL_factor,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        
        ## Calculate sensitivity run for root biomass
        GPP_R, Rml_R, Rmr_R, E_R, f_Psil_R, Psil_R, Psir_R, Psis_R, K_s_R, K_sr_R, k_srl_R = self.Plant.calculate(W_L,W_R*self.dWR_factor,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        
        
        ## Calculate change in GPP per unit change in biomass pool
        dGPPdWleaf = (GPP_L-GPP_0)/(W_L*self.dWL_factor - W_L)
        dGPPdWroot = (GPP_R-GPP_0)/(W_R*self.dWR_factor - W_R)
        
        ## Calculate marginal change in GPP-Rm per unit change in biomass pool
        dGPPRmdWleaf = ((GPP_L - (Rml_L + Rmr_L))-(GPP_0 - (Rml_0 + Rmr_0)))/(W_L*self.dWL_factor - W_L)
        dGPPRmdWroot = ((GPP_R - (Rml_R + Rmr_R))-(GPP_0 - (Rml_0 + Rmr_0)))/(W_R*self.dWR_factor - W_R)
        
        ## Calculate marginal change in cost per unit change in biomass pool - proportional to pool instantaneous turnover rate (inverse of mean lifespan)
        dSdWleaf = self.tr_L
        dSdWroot = self.tr_R

        ## Calculate allocation coefficients
        if dGPPRmdWleaf <= 0:
            # If marginal gain is >= 0, set coefficient to zero to avoid division with zeros
            u_L_prime = 0
        else:
            u_L_prime = (dGPPRmdWleaf/dSdWleaf)/((dGPPRmdWleaf/dSdWleaf)+np.maximum(0,dGPPRmdWroot/dSdWroot))
        if dGPPRmdWroot <= 0:
            # If marginal gain is >= 0, set coefficient to zero to avoid division with zeros
            u_R_prime = 0
        else:
            u_R_prime = (dGPPRmdWroot/dSdWroot)/(np.maximum(0,dGPPRmdWleaf/dSdWleaf)+(dGPPRmdWroot/dSdWroot))

        ## Scale optimal allocation coefficients so that the sum of all allocation coefficients is equal to 1
        u_L = (1 - (self.u_Stem+self.u_Seed))*u_L_prime
        u_R = (1 - (self.u_Stem+self.u_Seed))*u_R_prime

        return u_L, u_R, dGPPRmdWleaf, dGPPRmdWroot, dGPPdWleaf, dGPPdWroot, dSdWleaf, dSdWroot