import numpy as np
import numba as nb
from math import exp

@nb.vectorize(["float64(float64, float64)"], nopython=True)
def func_TempCoeff(airTempC: float, optTemperature: float=20):
    """
    Vectorized function to calculate the temperature coefficient.
    
    Parameters
    ----------
    airTempC : float
        Air temperature in degrees Celsius.
    
    optTemperature : float
        Optimal temperature at which the coefficient is 1.
    
    Returns
    -------
    TempCoeff : float
        Temperature coefficient.
    """
    factor1 = 0.20 * airTempC - 0.20 * optTemperature
    factor2 = abs(40 - airTempC)
    constant = 1.0 / (40 - optTemperature)  # Precompute division
    exponent = 8.0 - 0.2 * optTemperature
    return np.exp(factor1) * np.exp(exponent * np.log(factor2 * constant))


## Precompiling for faster run time
func_TempCoeff(np.array([25.0]), 20.0) 
func_TempCoeff(25.0, 20.0)

def growing_degree_days_HTT(Th, Tb, Tu, Topt, normalise=False):
    Th = np.asarray(Th)  # Ensure input is a NumPy array
    HTT = np.zeros_like(Th)  # Initialize an array of zeros with the same shape as Th

    # Identify valid ranges where Tb <= Th <= Tu
    valid_range = (Th >= Tb) & (Th <= Tu)

    # Apply the formula only to the valid range
    if normalise:
        HTT[valid_range] = ((Tu - Th[valid_range]) / (Tu - Topt)) * \
                           ((Th[valid_range] - Tb) / (Topt - Tb))**((Topt - Tb) / (Tu - Topt))
    else:
        HTT[valid_range] = ((Tu - Th[valid_range]) / (Tu - Topt)) * \
                           ((Th[valid_range] - Tb) / (Topt - Tb))**((Topt - Tb) / (Tu - Topt)) * (Topt - Tb)
    
    return HTT

