import numpy as np
import numba as nb

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
    delta_temp = airTempC - optTemperature
    factor1 = 0.20 * delta_temp
    factor2 = abs((40 - airTempC) / (40 - optTemperature))
    exponent = 8.0 - 0.2 * optTemperature
    return np.exp(factor1) * (factor2 ** exponent)