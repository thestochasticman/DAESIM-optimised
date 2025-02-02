"""
Plant model class: Includes differential equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp
from daesim_optimised.biophysics_funcs import func_TempCoeff, growing_degree_days_DTT_nonlinear, growing_degree_days_DTT_linear1, growing_degree_days_DTT_linear2, growing_degree_days_DTT_linear3
from daesim_optimised.management import ManagementModule
from daesim_optimised.climate import ClimateModule
from daesim_optimised.climate_funcs import solar_day_calcs

@define
class PlantModuleCalculator:
    """
    Calculator of plant biophysics
    """

    # Class parameters
    halfSaturCoeffP: float = field(
        default=0.01
    )  ## half-saturation coefficient for P; 0.000037 [Question: why do the Stella docs give a value (0.000037) that is different to the actual value used (0.01)?]
    halfSaturCoeffN: float = field(
        default=0.02
    )  ## half-saturation coefficient for Nitrogen; 0.00265 [Question: why do the Stella docs give a value (0.00265) that is different to the actual value used (0.002)?]
    optTemperature: float = field(default=20)  ## optimal temperature
    NPPCalibration: float = field(
        default=0.45
    )  ## NPP 1/day(The rate at which an ecosystem accumulates energy); 0.12; 0.25
    saturatingLightIntensity: float = field(
        default=600
    )  ## Saturating light intensity (langleys/d) for the selected crop

    maxAboveBM: float = field(default=0.6)  ## Max above ground biomass kg/m2. 0.9
    maxPropPhAboveBM: float = field(
        default=0.75
    )  ## proportion of photosynthetic biomass in the above ground biomass (0,1)(alfa*). 0.65

    mortality_constant: float = field(default=0.01)  ## temporary variable

    propPhMortality: float = field(
        default=0.015
    )  ## Proportion of photosynthetic biomass that dies in fall time
    propPhtoNPhMortality: float = field(
        default=0
    )  ## A proportion that decides how much of the Ph biomass will die or go to the roots at the fall time
    propPhBLeafRate: float = field(
        default=0
    )  ## Leaf fall rate [Question: This parameter has the prefix "prop" but it says it is a "rate". What does this parameter mean? What are its units? Why is the default value = 0?]
    dayLengRequire: float = field(
        default=13
    )  ## [Question: Need some information on this.]
    propPhBEverGreen: float = field(
        default=0.3
    )  ## Proportion of evergreen photo biomass

    FallLitterTurnover: float = field(
        default=0.001
    )  ## Modificatoin: This is a new parameter required to run in this framework.

    Max_Photosynthetic_Biomass: float = field(
        default=0.6
    )  ## A dummy, stella enforced variable that is needed for the sole purpose of tracking the maximum attained value of non-photosynthetic biomass; Finding the max of Ph in the whole period of model run; maximal biomass reached during the season

    iniNPhAboveBM: float = field(
        default=0.04
    )  ## initially available above ground non-photosynthetic tissue. kg/m2
    propAboveBelowNPhBM: float = field(
        default=0.85
    )  ## Ratio of above to below ground non-photosynthetic biomas (beta)
    heightMaxBM: float = field(default=1.2)  ## Height at maximum biomass
    estimateHeight: float = field(default=1.2)  ## Height of the crop (m). Modification: Changed the units from cm (in Stella) to m (used here). ErrorCheck: Not sure if the units conversions were correct in Stella DAESsim, as this was in cm yet everything else was in m.
    iniRootDensity: float = field(default=0.05)
    propNPhRoot: float = field(
        default=0.002
    )  ## [Question: If "NPh" means non-photosynthetic, then why isn't propNPhRoot = 1? 100% of root biomass should be non-photosynthetic]

    BioRepro: float = field(
        default=1800
    )  ## bio time when reproductive organs start to grow; 1900
    propPhtoNphReproduction: float = field(
        default=0.005
    )  ## fraction of photo biomass that may be transferred to non-photo when reproduction occurs

    propNPhMortality: float = field(
        default=0.04
    )  # non-photo mortality rate [Question: Units?? ]

    bioStart: float = field(
        default=20
    )  # start of sprouting [Question: What does this mean physiologically?]
    bioEnd: float = field(
        default=80
    )  # start of sprouting [Question: What does this mean physiologically?]
    sproutRate: float = field(
        default=0.01
    )  # Sprouting rate. Rate of translocation of assimilates from non-photo to photo bimass during early growth period

    rhizodepositReleaseRate: float = field(
        default=0.025
    )  ## Question: What does this parameter mean physiologically? No documentation available in Stella
    GDD_method: str = field(
        default="nonlinear"
        ) ## method used to calculate daily thermal time and hence growing degree days. Options are: "nonlinear", "linear1", "linear2", "linear3"
    GDD_Tbase: float = field(
        default=5.0
    )  ## Base temperature (minimum threshold) used for calculating the growing degree days
    GDD_Topt: float = field(
        default=25.0
    )  ## Optimum temperature used for calculating the growing degree days (non-linear method only)
    GDD_Tupp: float = field(
        default=40.0
    )  ## Upper temperature (maximum threshold) used for calculating the growing degree days

    def calculate(
        self,
        Photosynthetic_Biomass,
        Non_Photosynthetic_Biomass,
        Bio_time,
        solRadGrd,
        airTempCMin,
        airTempCMax,
        _doy,
        _year,
        Site=ClimateModule(),   ## It is optional to define Site for this method. If no argument is passed in here, then default setting for Site is the default ClimateModule(). Note that this may be important as it defines many site-specific variables used in the calculations.
        Management=ManagementModule(),   ## It is optional to define Management for this method. If no argument is passed in here, then default setting for Management is the default ManagementModule()
    ) -> Tuple[float]:
        PhBioPlanting = self.calculate_BioPlanting(_doy,Management.plantingDay,Management.propPhPlanting,Management.plantingRate,Management.plantWeight) ## Modification: using a newly defined parameter in this function instead of "frequPlanting" as used in Stella, considering frequPlanting was being used incorrectly, as its use didn't match with the units or definition.
        NPhBioPlanting = self.calculate_BioPlanting(_doy,Management.plantingDay,1-Management.propPhPlanting,Management.plantingRate,Management.plantWeight)  ## Modification: using a newly defined parameter in this function instead of "frequPlanting" as used in Stella, considering frequPlanting was being used incorrectly, as its use didn't match with the units or definition.

        WatStressHigh = 1
        WatStressLow = 0.99

        # Call the initialisation method
        PlantConditions = self._initialise(self.iniNPhAboveBM)

        PhBioHarvest = self.calculate_PhBioHarvest(Photosynthetic_Biomass,Non_Photosynthetic_Biomass,PlantConditions["maxBM"],_doy,Management.harvestDay,Management.propPhHarvesting,Management.PhHarvestTurnoverTime)
        NPhBioHarvest = self.calculate_NPhBioHarvest(Non_Photosynthetic_Biomass,_doy,Management.harvestDay,Management.propNPhHarvest,Management.NPhHarvestTurnoverTime)

        # Call "conditions for plant" methods (following Stella code naming convention for this)
        rootBM = self.calculate_rootBM(Non_Photosynthetic_Biomass)
        NPhAboveBM = self.calculate_NPhAboveBM(rootBM)
        calSoilDepth = 0.09  ## TODO: Add soil module variable for calSoilDepth
        rootDensity = self.calculate_rootDensity(
            Non_Photosynthetic_Biomass, calSoilDepth
        )  ## TODO: Add soil module variable for calSoilDepth
        Elevation = 70.74206543  ## TODO: Add elevation to climate module
        RootDepth = self.calculate_RootDepth(
            rootBM, rootDensity, Elevation
        )  ## TODO: Add elevation to climate module

        propPhAboveBM = self.calculate_propPhAboveBM(Photosynthetic_Biomass, NPhAboveBM)

        # Call the calculate_PhBioNPP method
        airTempC = Site.compute_mean_daily_air_temp(airTempCMin,airTempCMax)
        PhNPP = self.calculate_PhBioNPP(
            Photosynthetic_Biomass, solRadGrd, airTempC, WatStressHigh, WatStressLow
        )

        # Call the calculate_PhBioMort method
        # PhBioMort = self.calculate_PhBioMort(Photosynthetic_Biomass)
        dayLength = Site.sunlight_duration(_year,_doy+1)   ## TODO: Need to modify this; at some point we want to remove the dependency on dayLengthPrev altogether
        dayLengthPrev = Site.sunlight_duration(_year,_doy)   ## TODO: Need to modify this; at some point we want to remove the dependency on dayLengthPrev altogether
        PhBioMort, Fall_litter = self.calculate_PhBioMortality(
            Photosynthetic_Biomass,
            dayLength,
            dayLengthPrev,
            WatStressHigh,
            WatStressLow,
        )

        # Call the calculated_NPhBioMort method
        NPhBioMort = self.calculate_NPhBioMort(Non_Photosynthetic_Biomass)

        # Call the calculate_Transdown method
        Transdown = self.calculate_Transdown(
            Photosynthetic_Biomass,
            Bio_time,
            PhNPP,
            Fall_litter,
            propPhAboveBM,
        )

        Transup = self.calculate_Transup(
            Non_Photosynthetic_Biomass, Bio_time, propPhAboveBM
        )

        exudation = self.calculate_exudation(rootBM)

        # ODE for photosynthetic biomass
        dPhBMdt = PhNPP + PhBioPlanting + Transup - PhBioHarvest - Transdown - PhBioMort
        # ODE for non-photosynthetic biomass
        dNPhBMdt = (
            NPhBioPlanting
            + Transdown
            - Transup
            - NPhBioHarvest
            - NPhBioMort
            - exudation
        )

        sunrise, solarnoon, sunset = Site.solar_day_calcs(_year,_doy)
        DTT = self.calculate_dailythermaltime(airTempCMin,airTempCMax,sunrise,sunset)
        GDD_reset = self.calculate_growingdegreedays_reset(Bio_time,_doy,Management.plantingDay)
        dGDDdt = DTT - GDD_reset

        return (dPhBMdt, dNPhBMdt, dGDDdt)

    def _initialise(self, iniNPhAboveBM):
        ## TODO: Need to handle this initialisation better.
        ## Ideally, we want initial values like "iniNPhAboveBM" (although that's not exactly tied to an ODE state) to only be defined in the "Model", not here in the calculator.

        maxPhAboveBM = (
            self.maxAboveBM * self.maxPropPhAboveBM
        )  ## The maximum general photosynthetic biomass above ground (not site specific). kg/m2*dimless=kg/m2
        maxNPhBM = (self.maxAboveBM - maxPhAboveBM) * (
            1 + 1 / self.propAboveBelowNPhBM
        )  ## maximum above ground non phtosynthetic+(max above ground non pythosynthetic/proportion of above to below ground non photosynthetic)kg/m2
        maxNPhAboveBM = (
            iniNPhAboveBM * (self.propAboveBelowNPhBM + 1) / self.propAboveBelowNPhBM
        )  ## initial non-photo above ground biomass
        maxBM = maxNPhBM + maxPhAboveBM  ## kg/m2
        iniPhBM = (
            self.propPhBEverGreen
            * iniNPhAboveBM
            * self.maxPropPhAboveBM
            / (1 - self.maxPropPhAboveBM)
        )  ## initial biomass of photosynthetic tissue  (kgC/m^2).[]. calculated based on the available above ground non-photosynthetic tissue. PH/(PH+Ab_BM)=Max_Ph_to_Ab_BM. The introduction of the PhBio_evgrn_prop in this equation eliminates all leaves from deciduous trees.  Only coniferous trees are photosynthesizing!

        return {
            "iniPhBM": iniPhBM,
            "maxBM": maxBM,
            "maxNPhBM": maxNPhBM,
            "maxPhAboveBM": maxPhAboveBM,
            "maxNPhAboveBM": maxNPhAboveBM,
        }

    def calculate_PhBioNPP(
        self, Photosynthetic_Biomass, solRadGrd, airTempC, WatStressHigh, WatStressLow
    ):
        PO4Aval = 0.2  # TODO: Change name to "PO4Avail"; ErrorCheck: What is this? Why is it set to 0 in Stella? This makes the NutrCoeff=0, NPPControlCoeff=0, then calculatedPhBioNPP=0
        DINAvail = 0.2  # [Question: No documentation. Check paper or ask Firouzeh/Justin to provide.] ErrorCheck: What is this? Why is it set to 0 in Stella? This makes the NutrCoeff=0, NPPControlCoeff=0, then calculatedPhBioNPP=0
        NutrCoeff = min(
            (DINAvail / (DINAvail + self.halfSaturCoeffN)),
            (PO4Aval / (PO4Aval + self.halfSaturCoeffP)),
        )

        LightCoeff = (
            solRadGrd
            * 10
            / self.saturatingLightIntensity
            * np.exp(1 - solRadGrd / self.saturatingLightIntensity)
        )

        WaterCoeff = min(WatStressHigh, WatStressLow)

        TempCoeff = func_TempCoeff(airTempC,optTemperature=self.optTemperature)

        NPPControlCoeff = (
            np.minimum(LightCoeff, TempCoeff) * WaterCoeff * NutrCoeff
        )  ## Total control function for primary production,  using minimum of physical control functions and multiplicative nutrient and water controls.  Units=dimensionless.

        maxPhAboveBM = (
            self.maxAboveBM * self.maxPropPhAboveBM
        )  ## The maximum general photosynthetic biomass above ground (not site specific). kg/m2*dimless=kg/m2

        # calculatedPhBioNPP ## Estimated net primary productivity
        # This function essentially says: if Photosynthetic_Biomass < maxPhAboveBM: calculate NPP; else: NPP=0
        calculatedPhBioNPP = np.maximum((Photosynthetic_Biomass < maxPhAboveBM) * (NPPControlCoeff * self.NPPCalibration * Photosynthetic_Biomass * (1-Photosynthetic_Biomass/maxPhAboveBM)),0)

        return calculatedPhBioNPP

    def calculate_PhBioMort(self, Photosynthetic_Biomass):
        PhBioMort = self.mortality_constant * Photosynthetic_Biomass

        return PhBioMort

    def calculate_PhBioMortality(
        self,
        Photosynthetic_Biomass,
        dayLength,
        dayLengthPrev,
        WatStressHigh,
        WatStressLow,
    ):
        WaterCoeff = min(
            WatStressHigh, WatStressLow
        )  ## TODO: Modify the structure here as it is used a couple of times in the Plant module

        PropPhMortDrought = 0.1 * max(0, (1 - WaterCoeff))

        # FallLitterCalc
        ## Modification: Had to vectorize the if statements to support array-based calculations
        _vfunc = np.vectorize(self.calculate_FallLitter)
        FallLitterCalc = _vfunc(Photosynthetic_Biomass, dayLength, dayLengthPrev)

        ## [Question: What does this function represent?]
        Fall_litter = np.minimum(
            FallLitterCalc,
            np.maximum(
                0,
                Photosynthetic_Biomass
                - self.propPhBEverGreen
                * self.Max_Photosynthetic_Biomass
                / (1 + self.propPhBEverGreen),
            ),
        )

        ## PhBioMort: mortality of photosynthetic biomass as influenced by seasonal cues plus mortality due to current
        ## (not historical) water stress.  Use maximum specific rate of mortality and constraints due to unweighted
        ## combination of seasonal litterfall and water stress feedbacks (both range 0,1). units = 1/d * kg * (dimless +dimless)  = kg/d
        PhBioMort = Fall_litter * (
            1 - self.propPhtoNPhMortality
        ) + Photosynthetic_Biomass * (PropPhMortDrought + self.propPhMortality)

        return (PhBioMort, Fall_litter)

    def calculate_FallLitter(
        self, Photosynthetic_Biomass, dayLength, dayLengthPrev
    ):  ## TODO: change the naming convention for these sub-calculators to something like calculate_conditional_x(), as these are just convenient coding hacks that allow us to vectorize the if/elif/else statements.
        if (dayLength > self.dayLengRequire) or (
            dayLength >= dayLengthPrev
        ):  ## Question: These two options define very different phenological periods. Why use this approach?
            ## Question: This formulation means that during the growing season (when dayLength < dayLengRequire) there is no litter production! Even a canopy that is growing will turnover leaves/branches and some litter will be produced. This is especially true for trees like Eucalypts.
            ## TODO: Modify this formulation to something more similar to my Knorr implementation in CARDAMOM, where there is a background turnover rate.
            return 0
        elif Photosynthetic_Biomass < 0.01 * (1 - self.propPhBEverGreen):
            ## [Question: What is the 0.01 there for?]
            ## [Question: So, this if statement option essentially says convert ALL Photosynthetic_Biomass into litter in this time-step, yes? Why? ]
            ## Modification: I had to remove the time-step (DT) dependency in the equation below. Instead, I have implemented a turnover rate parameter. This may change the results compared to Stella.
            return (1 - self.propPhBEverGreen) * np.minimum(
                Photosynthetic_Biomass * self.FallLitterTurnover, Photosynthetic_Biomass
            )
        else:
            ## [Question: What is this if statement option doing? Why this formulation? Why is it to the power of 3?]
            return (1 - self.propPhBEverGreen) * (
                self.Max_Photosynthetic_Biomass
                * self.propPhBLeafRate
                / Photosynthetic_Biomass
            ) ** 3

    def calculate_Transdown(
        self, Photosynthetic_Biomass, Bio_time, PhNPP, Fall_litter, propPhAboveBM
    ):
        # TransdownRate:
        # The plant attempts to obtain the optimum photobiomass to total above ground biomass ratio.  Once this is reached,
        # NPP is used to grow more Nonphotosythethic biomass decreasing the optimum ratio.  This in turn allows new Photobiomass
        # to compensate for this loss; IF Ph_to_Ab_BM[Habitat,Soil]>•Max_Ph_to_Ab_BM[Habitat,Soil] THEN 1; ELSE Ph_to_Ab_BM[Habitat,Soil]/•Max_Ph_to_Ab_BM[Habitat,Soil]

        #         if Bio_time > self.BioRepro + 1:
        #             TransdownRate = 1 - 1 / (Bio_time - self.BioRepro) ** 0.5
        #         elif propPhAboveBM < self.maxPropPhAboveBM:
        #             TransdownRate = 0
        #         else:
        #             TransdownRate = (
        #                 np.cos((self.maxPropPhAboveBM / propPhAboveBM) * np.pi / 2) ** 0.1
        #             )
        ## Modification: Had to vectorize the above if statements to support array-based calculations
        _vfunc = np.vectorize(self.calculate_TransdownRate,otypes=[float])
        TransdownRate = _vfunc(Bio_time, propPhAboveBM)

        ## TODO: Include HarvestTime info
        # if HarvestTime > 0:
        #     Transdown = 0
        # else:
        #     Transdown = TransdownRate*(PhNPP+propPhtoNphReproduction*Photosynthetic_Biomass)+(propPhtoNPhMortality*Fall_litter)
        Transdown = TransdownRate * (
            PhNPP + self.propPhtoNphReproduction * Photosynthetic_Biomass
        ) + (self.propPhtoNPhMortality * Fall_litter)

        return Transdown

    def calculate_TransdownRate(
        self, Bio_time, propPhAboveBM
    ):  ## TODO: change the naming convention for these sub-calculators to something like calculate_conditional_x(), as these are just convenient coding hacks that allow us to vectorize the if/elif/else statements.
        if Bio_time > self.BioRepro + 1:
            return 1 - 1 / (Bio_time - self.BioRepro) ** 0.5
        elif propPhAboveBM < self.maxPropPhAboveBM:
            return 0
        else:
            return np.cos((self.maxPropPhAboveBM / propPhAboveBM) * np.pi / 2) ** 0.1

    def calculate_Sprouting(self, Bio_time, propPhAboveBM):
        # Stella code: IF  propPhAboveBM < maxPropPhAboveBM  AND Bio_time >bioStart AND Bio_time <  bioEnd THEN 1 ELSE 0
        _vfunc = np.vectorize(self.calculate_Sprouting_conditional)
        Sprouting = _vfunc(Bio_time, propPhAboveBM)
        return Sprouting

    def calculate_Sprouting_conditional(self, Bio_time, propPhAboveBM):
        if (
            (propPhAboveBM < self.maxPropPhAboveBM)
            and (Bio_time > self.bioStart)
            and (Bio_time < self.bioEnd)
        ):
            return 1
        else:
            return 0

    def calculate_Transup(self, Non_Photosynthetic_Biomass, Bio_time, propPhAboveBM):
        Sprouting = self.calculate_Sprouting(Bio_time, propPhAboveBM)

        return Sprouting * self.sproutRate * Non_Photosynthetic_Biomass

    def calculate_NPhBioMort(self, Non_Photosynthetic_Biomass):
        return self.propNPhMortality * Non_Photosynthetic_Biomass

    def calculate_rootBM(self, Non_Photosynthetic_Biomass):
        return Non_Photosynthetic_Biomass / (self.propAboveBelowNPhBM + 1)

    def calculate_NPhAboveBM(self, rootBM):
        return self.propAboveBelowNPhBM * rootBM

    def calculate_rootDensity(self, Non_Photosynthetic_Biomass, calSoilDepth):
        rootDensity = np.maximum(
            self.iniRootDensity,
            Non_Photosynthetic_Biomass * self.propNPhRoot * 1 / calSoilDepth,
        )
        return rootDensity

    def calculate_RootDepth(self, rootBM, rootDensity, Elevation):
        RootDepth = np.maximum((Elevation / 100) - 1, rootBM / rootDensity)
        return RootDepth

    def calculate_propPhAboveBM(self, Photosynthetic_Biomass, NPhAboveBM):
        _vfunc = np.vectorize(self.calculate_propPhAboveBM_conditional)
        propPhAboveBM = _vfunc(Photosynthetic_Biomass, NPhAboveBM)
        return propPhAboveBM

    def calculate_propPhAboveBM_conditional(self, Photosynthetic_Biomass, NPhAboveBM):
        if NPhAboveBM == 0:
            return 0
        else:
            return Photosynthetic_Biomass / (Photosynthetic_Biomass + NPhAboveBM)

    def calculate_exudation(self, rootBM):
        exudation = rootBM * self.rhizodepositReleaseRate
        return exudation

    def calculate_BioPlanting(self,_doy,plantingDay,propBMPlanting,plantingRate,plantWeight):
        """
        _doy = ordinal day of year
        propBMPlanting = the proportion of planting that applies to this live biomass pool (e.g. if sowing seeds, calculation of the the non-photosynthetic planting flux will require propBMPlanting=1). Modification: The Stella code uses a parameter "frequPlanting" which isn't the correct use, given its definition. 

        returns:
        BioPlanting = the flux of carbon planted
        """
        _vfunc = np.vectorize(self.calculate_BioPlanting_conditional,otypes=[float])
        BioPlanting = _vfunc(_doy,plantingDay,propBMPlanting,plantingRate,plantWeight)
        return BioPlanting

    def calculate_BioPlanting_conditional(self,_doy,plantingDay,propBMPlanting,plantingRate,plantWeight):
        # Modification: I have modified the variables/parameters used in this function as the definitions and units in the Stella code didn't match up (see previous parameters maxDensity and frequPlanting vs new parameters plantingRate and propPhPlanting).
        PlantingTime = self.calculate_plantingtime_conditional(_doy,plantingDay)
        BioPlanting = PlantingTime * plantingRate * plantWeight * propBMPlanting
        return BioPlanting

    def calculate_PhBioHarvest(self,Photosynthetic_Biomass,Non_Photosynthetic_Biomass,maxBM,_doy,harvestDay,propPhHarvesting,PhHarvestTurnoverTime):
        _vfunc = np.vectorize(self.calculate_harvesttime_conditional,otypes=[float])
        HarvestTime = _vfunc(_doy,harvestDay)
        _vfunc = np.vectorize(self.calculate_CalcuHeight_conditional,otypes=[float])
        CalcuHeight = _vfunc(Photosynthetic_Biomass,Non_Photosynthetic_Biomass,maxBM)
        _vfunc = np.vectorize(self.calculate_removaltime_conditional,otypes=[float])
        RemovalTime = _vfunc(CalcuHeight,_doy,propPhHarvesting)
        #PhBioHarvest = (HarvestTime+RemovalTime)*propPhHarvesting*Photosynthetic_Biomass/PhHarvestTurnoverTime  ## Question: Why is it HarvestTime+RemovalTime? That could produce a factor of 2 for this flux. Why not max(0,HarvestTime,RemovalTime)?
        PhBioHarvest = (HarvestTime)*propPhHarvesting*Photosynthetic_Biomass/PhHarvestTurnoverTime  ## Modification: Not considering "Removal" as a harvest flux, so RemovalTime is not being used. It is unclear what this represents, it is not triggered in the default Stella DAESim run, and it seems to cause real headaches for the scipy solver, so I'm leaving it out for now.
        return PhBioHarvest

    def calculate_harvesttime_conditional(self,_doy,harvestDay):
        if harvestDay is None:
            return 0
        elif (harvestDay <= _doy < harvestDay+3):  ## Question: Why is the harvest period fixed to three days? Why not one considering this model technically applies to one management unit (or rather one m2)?
            return 1
        else:
            return 0

    def calculate_plantingtime_conditional(self,_doy,plantingDay):
        if plantingDay is None:
            return 0
        elif (plantingDay <= _doy < plantingDay+1):
            return 1
        else:
            return 0

    def calculate_removaltime_conditional(self,CalcuHeight,_doy,propPhHarvesting):
        "Regular maintenance that does not cause death of plants. Kicks in whenever the plant height exceeds a certain value"
        ## Question: What does this represent in terms of management practice? Is it appropriate or necessary? Or just required because the plant growth model would continue to grow beyond the max height (or biomass) if this removal flux wasn't present.
        if CalcuHeight*propPhHarvesting > self.estimateHeight:
            return 1
        else:
            return 0

    def calculate_CalcuHeight_conditional(self,Photosynthetic_Biomass,Non_Photosynthetic_Biomass,maxBM):
        "Height of the plants relative to current biomass. It is assumed that max height occurs at max biomass."
        if maxBM > 0:
            TotalBM = Photosynthetic_Biomass + Non_Photosynthetic_Biomass
            return self.heightMaxBM * TotalBM/maxBM
        else:
            return 0

    def calculate_NPhBioHarvest(self,Non_Photosynthetic_Biomass,_doy,harvestDay,propNPhHarvest,NPhHarvestTurnoverTime):
        _vfunc = np.vectorize(self.calculate_harvesttime_conditional,otypes=[float])
        HarvestTime = _vfunc(_doy,harvestDay)
        NPhBioHarvest = HarvestTime*propNPhHarvest*Non_Photosynthetic_Biomass/NPhHarvestTurnoverTime
        return NPhBioHarvest

    def calculate_dailythermaltime(self,Tmin,Tmax,sunrise,sunset):
        if self.GDD_method == "nonlinear":
            _vfunc = np.vectorize(growing_degree_days_DTT_nonlinear)
            DTT = _vfunc(Tmin,Tmax,sunrise,sunset,self.GDD_Tbase,self.GDD_Tupp,self.GDD_Topt)
        elif self.GDD_method == "linear1":
            _vfunc = np.vectorize(growing_degree_days_DTT_linear1)
            DTT = _vfunc(Tmin,Tmax,self.GDD_Tbase,self.GDD_Tupp)
        elif self.GDD_method == "linear2":
            _vfunc = np.vectorize(growing_degree_days_DTT_linear2)
            DTT = _vfunc(Tmin,Tmax,self.GDD_Tbase,self.GDD_Tupp)
        elif self.GDD_method == "linear3":
            _vfunc = np.vectorize(growing_degree_days_DTT_linear3)
            DTT = _vfunc(Tmin,Tmax,self.GDD_Tbase,self.GDD_Tupp)
        return DTT

    def calculate_growingdegreedays_reset(self,GDD,_doy,plantingDay):
        _vfunc = np.vectorize(self.calculate_growingdegreedays_reset_conditional)
        GDD_reset_flux = _vfunc(GDD,_doy,plantingDay)
        return GDD_reset_flux

    def calculate_growingdegreedays_reset_conditional(self,GDD,_doy,plantingDay):
        """
        When it is planting/sowing time (plantingTime == 1), subtract some arbitrarily large 
        number from the growing degree days (GDD) state. It just has to be a large enough 
        number to trigger a zero-crossing "event" for Scipy solve_ivp to recognise. 
        """
        PlantingTime = self.calculate_plantingtime_conditional(_doy,plantingDay)
        GDD_reset_flux = PlantingTime * 1e9
        return GDD_reset_flux




"""
Differential equation solver implementation for plant model
"""

@define
class PlantModelSolver:

    """
    Plant model solver implementation
    """

    calculator: PlantModuleCalculator
    """Calculator of plant model"""

    site: ClimateModule
    """Site and climate details"""

    management: ManagementModule
    """Management details"""

    state1_init: float
    """
    Initial value for state 1
    """

    state2_init: float
    """
    Initial value for state 2
    """

    state3_init: float
    """
    Initial value for state 3
    """

    time_start: float
    """
    Time at which the initialisation values apply.
    """

    def run(
        self,
        airTempCMin: Callable[[float], float],
        airTempCMax: Callable[[float], float],
        solRadGrd: Callable[[float], float],
        _doy: Callable[[float], float],
        _year: Callable[[float], float],
        time_axis: float,
    ) -> Tuple[float]:
        func_to_solve = self._get_func_to_solve(
            self.site,
            self.management,
            airTempCMin,
            airTempCMax,
            solRadGrd,
            _doy,
            _year,
        )

        t_eval = time_axis
        t_span = (self.time_start, t_eval[-1])
        start_state = (
            self.state1_init,
            self.state2_init,
            self.state3_init,
        )

        ## When the state y[2] goes below zero we trigger the solver to terminate and restart at the next time step
        def zero_crossing_event(t, y):
            return y[2]
        zero_crossing_event.terminal = True
        zero_crossing_event.direction = -1

        solve_kwargs = {
            "t_span": t_span,
            "t_eval": t_eval,
            "y0": start_state,
            "events": zero_crossing_event,
        }

        res_raw = self._solve_ivp(
            func_to_solve,
            **solve_kwargs,
        )

        ## if a termination event occurs, we restart the solver at the next time step 
        ## with new initial values and continue until the status changes
        if res_raw.status == 1:
            res_next = res_raw
        while res_raw.status == 1:
            t_restart = np.ceil(res_next.t_events[0][0])
            t_eval = time_axis[time_axis >= t_restart]
            t_span = (t_restart, t_eval[-1])
            start_state_restart = res_next.y_events[0][0]
            start_state_restart[2] = 0
            solve_kwargs = {
                "t_span": t_span,
                "t_eval": t_eval,
                "y0": tuple(start_state_restart),
                "events": zero_crossing_event,
            }
            res_next = self._solve_ivp(
                func_to_solve,
                **solve_kwargs,
                )
            res_raw.t = np.append(res_raw.t,res_next.t)
            res_raw.y = np.append(res_raw.y,res_next.y,axis=1)
            if res_next.status == 1:
                res_raw.t_events = res_raw.t_events + res_next.t_events
                res_raw.y_events = res_raw.y_events + res_next.y_events
            res_raw.nfev += res_next.nfev
            res_raw.njev += res_next.njev
            res_raw.nlu += res_next.nlu
            res_raw.sol = res_next.sol
            res_raw.message = res_next.message
            res_raw.success = res_next.success
            res_raw.status = res_next.status

        return res_raw

    def _get_func_to_solve(
        self,
        Site,
        Management,
        airTempCMin: Callable[float, float],
        airTempCMax: Callable[float, float],
        solRadGrd: Callable[float, float],
        _doy: Callable[float, float],
        _year: Callable[float, float],
    ) -> Callable[float, float]:
        def func_to_solve(t: float, y: np.ndarray) -> np.ndarray:
            """
            Function to solve i.e. f(t, y) that goes on the RHS of dy/dt = f(t, y)

            Parameters
            ----------
            t
                time

            y
                State vector

            Returns
            -------
                dy / dt (also as a vector)
            """
            airTempCMinh = airTempCMin(t).squeeze()
            airTempCMaxh = airTempCMax(t).squeeze()
            solRadGrdh = solRadGrd(t).squeeze()
            _doyh = _doy(t).squeeze()
            _yearh = _year(t).squeeze()

            dydt = self.calculator.calculate(
                Photosynthetic_Biomass=y[0],
                Non_Photosynthetic_Biomass=y[1],
                Bio_time=y[2],
                solRadGrd=solRadGrdh,
                airTempCMin=airTempCMinh,
                airTempCMax=airTempCMaxh,
                _doy=_doyh,
                _year=_yearh,
                Site=Site,
                Management=Management,
            )

            # TODO: Use this python magic when we have more than one state variable in dydt
            # dydt = [v for v in dydt]

            return dydt

        return func_to_solve

    def _solve_ivp(
        self, func_to_solve, t_span, t_eval, y0, rtol=1e-6, atol=1e-6, **kwargs
    ) -> OptimizeResult:
        raw = solve_ivp(
            func_to_solve,
            t_span=t_span,
            t_eval=t_eval,
            y0=y0,
            atol=atol,
            rtol=rtol,
            **kwargs,
        )
        if not raw.success:
            info = "Your model failed to solve, perhaps there was a runaway feedback?"
            error_msg = f"{info}\n{raw}"
            raise SolveError(error_msg)

        return raw