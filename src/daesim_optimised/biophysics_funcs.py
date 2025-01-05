"""
Biophysics helper functions used across more than one DAESim module
"""
from typing_extensions import Callable
import numpy as np
import numba as nb

@nb.vectorize(['float64(float64, float64)'])
def func_TempCoeff_numba(airTempC: float, optTemperature: float=20):
    """
    Vectorized function to calculate the temperature coefficient.
    Errorcheck: This function seems okay for temperatures below 40 degC but it goes whacky above 40 degC. This is a problem that we'll have to correct.
    TODO: Correct the whacky values from the calculate_TempCoeff functiono when airTempC > 40 degC.
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

    # d = airTempC - optTemperature
    # a = np.exp(0.2 * d)
    # b = 40 - airTempC
    # c = 40 - optTemperature
    # g = 0.2 * c
    # h = abs(b/c)
    # i = np.exp(g * np.log(h)) # a * (h ** g)
    # z = a * i
    delta = airTempC - optTemperature
    ratio = abs((40-airTempC)/(40-optTemperature))
    return np.exp(0.2 * delta) * np.exp((0.2 * optTemperature) * np.log(ratio))

## Precompiling for faster run time
func_TempCoeff_numba(np.array([25.0]), 20.0) 
func_TempCoeff_numba(25.0, 20.0)

def func_TempCoeff(airTempC: float, optTemperature: float=20):
  return func_TempCoeff_numba(airTempC, optTemperature)

# Precomputed constants
R_fT_arrheniuspeaked = 8.314  # universal gas constant J mol-1 K-1
T_ref_fT_arrheniuspeaked = 298.15  # Reference temperature in Kelvin
T_ref_alt_fT_arrheniuspeaked = 289.15  # Alternative reference temperature in Kelvin
@nb.njit(cache=True)
def fT_arrheniuspeaked_numba(k_25, T_k, E_a=70.0, H_d=200, DeltaS=0.650):
    """
    Optimized function to apply a peaked Arrhenius-type temperature scaling.
    """

    # Convert units from kJ to J
    E_a *= 1e3
    H_d *= 1e3
    DeltaS *= 1e3
    # Convert temperature to Kelvin
    T_k += 273.15
    
    # Precompute repeated terms
    reciprocal_T = 1.0 / T_k
    exp_term1 = (E_a * (T_k - T_ref_fT_arrheniuspeaked)) / (T_ref_fT_arrheniuspeaked * R_fT_arrheniuspeaked * T_k)
    exp_term2 = (T_ref_fT_arrheniuspeaked * DeltaS - H_d) / (T_ref_alt_fT_arrheniuspeaked * R_fT_arrheniuspeaked)
    exp_term3 = (T_k * DeltaS - H_d) * reciprocal_T / R_fT_arrheniuspeaked
    
    # Compute scaling factor
    k_scaling = np.exp(exp_term1) * ((1.0 + np.exp(exp_term2)) / (1.0 + np.exp(exp_term3)))
    return k_25 * k_scaling

fT_arrheniuspeaked_numba(0.1, 0.1, 70.0, 200, 0.650)
def fT_arrheniuspeaked(k_25, T_k, E_a=70.0, H_d=200, DeltaS=0.650):
  return fT_arrheniuspeaked_numba(k_25, T_k, E_a, H_d, DeltaS)

R_fT_arrhenius = 8.314
@nb.njit(cache=True)
def fT_arrhenius_numba(k_25, T_k, E_a, T_opt):
    """
    Applies an Arrhenius-type temperature scaling function to the given parameter.
    
    Parameters
    ----------
    k_25: float
        Rate constant at 25oC

    T_k: float
        Temperature, degrees Celsius

    E_a: float
        Activation energy, kJ mol-1, gives the rate of exponential increase of the function

    T_opt: float
        Optimum temperature for rate constant, K

    Returns
    -------
    Temperature adjusted rate constant at the given temperature.

    References
    ----------
    Medlyn et al. (2002) Equation 16
    """
    T_k = T_k + 273.15
    E_a = E_a * 1e3  # convert kJ mol-1 to J mol-1

    k_scaling = np.exp( (E_a * (T_k - T_opt))/(T_opt*R_fT_arrhenius*T_k)) 

    return k_25*k_scaling

fT_arrhenius_numba(0.25, 0.25, 70.0, 298.15)

def fT_arrhenius(k_25, T_k, E_a=70.0, T_opt=298.15):
    return fT_arrhenius_numba(k_25, T_k, E_a, T_opt)
fT_arrhenius(0.25, 0.25, 70.0, 298.15)
fT_arrhenius(0.25, 0.25, 70.0, 298.15)

@nb.njit(cache=True)
def fT_Q10_numba(k_25, T_k, Q10=2.0):
    """
    Numba-optimized Q10 temperature scaling function for scalar inputs.
    """
    exponent = (T_k - 25.0) * 0.1
    # k_scaling = Q10 ** exponent
    k_scaling = np.exp(exponent * np.log(Q10))
    return k_25 * k_scaling
  
fT_Q10_numba(0.1, 0.1, 2.0)

def fT_Q10(k_25, T_k, Q10:float=2.0):
    return fT_Q10_numba(k_25, T_k, Q10)
  
fT_Q10(0.1, 0.1, 2.0)
fT_Q10(0.1, 0.1, 2.0)

import numba as nb
import numpy as np

@nb.njit(cache=True)
def arange(start, stop, step):
    """
    Numba-optimized version of np.arange.
    
    Parameters
    ----------
    start: float
        Start of the interval.
        
    stop: float
        End of the interval.
        
    step: float
        Spacing between values.
        
    Returns
    -------
    result: ndarray
        Array of evenly spaced values from start to stop (exclusive).
    """
    n = int((stop - start) / step)  # Compute the number of elements
    result = np.empty(n, dtype=np.float64)  # Preallocate array

    value = start
    for i in range(n):
        result[i] = value
        value += step
    
    return result

arange(0, 24, 0.5)

@nb.njit(cache=True)
def _diurnal_temperature(Tmin, Tmax, t_sunrise, t_sunset, t_step=1):
    """
    Numba-optimized function for computing diurnal temperature profile.
    """
    t = arange(0, 24, t_step)
    # Precompute average and amplitude
    T_average = (Tmin + Tmax) / 2.0
    T_amplitude = (Tmax - Tmin) / 2.0
    
    # Precompute constants for cosine terms
    coeff1 = np.pi / (14.0 - t_sunrise)
    coeff2 = np.pi / (10.0 + t_sunrise)
    
    # Initialize temperature profile
    T_H = np.empty_like(t)
    
    # Compute temperature using Equation 1 (for daytime)
    for i in range(t.size):
        if t[i] >= t_sunrise and t[i] <= 14.0:
            T_H[i] = T_average - T_amplitude * np.cos(coeff1 * (t[i] - t_sunrise))
        else:
            H_prime = t[i] - 14.0 if t[i] > 14.0 else t[i] + 10.0
            T_H[i] = T_average + T_amplitude * np.cos(coeff2 * H_prime)
    
    return T_H

_diurnal_temperature(10.0, 30.0, 6.5, 20.25, 0.5)

# @nb.njit
# def diurnal_temperature(Tmin, Tmax, t_sunrise, t_sunset, tstep=1):
#   # Tmin_is_scalar = np.isscalar(Tmin)
#   # Tmax_is_scalar = np.isscalar(Tmax)
#   # print(Tmin_is_scalar)
#   # if Tmax_is_scalar and Tmin_is_scalar:
#   #   t = np.arange(0, 24, tstep)
#   #   return _diurnal_temperature(Tmin, Tmax, t_sunrise, t_sunset, t)

#   # if not Tmax_is_scalar and not Tmax_is_scalar:
#   if Tmin.size != Tmax.size:
#     raise ValueError('Size of Tmin and Tmax inputs must be the same')
#   else:
#     n_days = Tmin.size
#     t = np.arange(0, 24, tstep)
#     n_t_steps = t.size
#     T_hr = np.empty((n_days, n_t_steps))
    
#     for i in nb.prange(n_days):  # Parallel loop over days
#       T_hr[i, :] = _diurnal_temperature(Tmin[i], Tmax[i], t_sunrise, t_sunset, t)
#     return T_hr
#   # else:
#   #   raise ValueError('One is array, one is float')
  
# diurnal_temperature(np.array([5.0]), np.array(6.0), 6.5, 7.5, 1.0)
    

def growing_degree_days_HTT(Th,Tb,Tu,Topt,normalise):
    """
    Calculates the hourly thermal time (HTT) or 'heat units' according to a peaked temperature response model.
    The temperature response model is based on that of Yan and Hunt (1999, doi:10.1006/anbo.1999.0955).
    Also see description in Zhou and Wang (2018, doi:10.1038/s41598-018-28392-z).

    Parameters
    ----------
    Th : float
        Hourly temperature (degrees Celsius)
    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celsius)
    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)
    Topt : float
        Thermal optimum temperature (degrees Celsius)
    normalise : str
        Normalize the thermal time function to range between 0-1. 

    Returns
    -------
    Hourly thermal time (HTT): float
    """
    if Th < Tb:
        return 0
    elif (Th >= Tb) and (Th <= Tu):
        if normalise:
            return ((Tu-Th)/(Tu-Topt)) * ((Th-Tb)/(Topt-Tb))**((Topt-Tb)/(Tu-Topt))
        else:
            return ((Tu-Th)/(Tu-Topt)) * ((Th-Tb)/(Topt-Tb))**((Topt-Tb)/(Tu-Topt)) * (Topt-Tb)
    elif Tu < Th:
        return 0

def growing_degree_days_DTT_from_HTT(HTT,tstep=1):
    """
    Calculates the daily thermal time (DTT) from the hourly thermal time (HTT) by
    taking the average of the HTT values.
    """
    t = np.arange(0,24,tstep)
    n = t.size
    return np.sum(HTT)/n

def growing_degree_days_DTT_nonlinear(Tmin,Tmax,t_sunrise,t_sunset,Tb,Tu,Topt,normalise=False):
    """
    Calculates the daily thermal time from the minimum daily temperature, maximum daily
    temperature, sunrise time, sunset time, and the cardinal temperatures that describe
    the hourly thermal time temperature response model.

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

    t_sunrise: float or ndarray
        Time of sunrise (24 hour time, e.g. at 6:30 am, t = 6.5)

    t_sunrise: float or ndarray
        Time of sunset (24 hour time, e.g. at 8:15 pm, t = 20.25)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celcsius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)

    Topt : float
        Thermal optimum temperature (degrees Celsius)

    normalise : str
        Normalize the thermal time function to range between 0-1. 

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celsius)
    """
    T_diurnal_profile = _diurnal_temperature(Tmin,Tmax,t_sunrise,t_sunset)
    _vfunc = np.vectorize(growing_degree_days_HTT,otypes=[float])
    HTT_ = _vfunc(T_diurnal_profile,Tb,Tu,Topt,normalise=normalise)
    DTT = growing_degree_days_DTT_from_HTT(HTT_)
    return DTT

def growing_degree_days_DTT_linear1(Tmin,Tmax,Tb,Tu):
    """
    Calculates the daily thermal time (DTT) using the linear "Method 1" in McMaster and 
    Wilhelm (1997, doi:10.1016/S0168-1923(97)00027-0). This function requires the 
    minimum daily temperature, maximum daily temperature, the base and upper threshold 
    temperatures that describe the hourly thermal time temperature response model.

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celsius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celsius)
    """
    Tavg = (Tmin+Tmax)/2
    if Tavg < Tb:
        return 0
    elif (Tavg > Tb) and (Tavg < Tu):
        return Tavg - Tb
    elif Tavg >= Tu:
        return Tu - Tb

def growing_degree_days_DTT_linear2(Tmin,Tmax,Tb,Tu):
    """
    Calculates the daily thermal time (DTT) using the linear "Method 2" in McMaster and 
    Wilhelm (1997, doi:10.1016/S0168-1923(97)00027-0). This function requires the 
    minimum daily temperature, maximum daily temperature, the base and upper threshold 
    temperatures that describe the hourly thermal time temperature response model.

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celsius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celsius)
    """
    if Tmax < Tb:
        Tmax = Tb
    elif Tmax > Tu:
        Tmax = Tu
    if Tmin < Tb:
        Tmin = Tb
    elif Tmin > Tu:
        Tmin = Tu
    Tavg = (Tmin+Tmax)/2
    if Tavg < Tb:
        return 0
    elif (Tavg > Tb) and (Tavg < Tu):
        return Tavg - Tb
    elif Tavg >= Tu:
        return Tu - Tb

def growing_degree_days_DTT_linear3(Tmin,Tmax,Tb,Tu):
    """
    Calculates the daily thermal time (DTT) using the linear "Method 2" in Zhou and 
    Wang (2018, doi:10.1038/s41598-018-28392-z). This function requires the minimum 
    daily temperature, maximum daily temperature, the base and upper threshold 
    temperatures that describe the hourly thermal time temperature response model.

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celsius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celsius)
    """
    Tavg = (Tmin+Tmax)/2
    if Tavg <= Tb:
        return 0
    elif (Tavg > Tb) and (Tavg < Tu):
        Tm = min(Tmax,Tu)
        Tn = max(Tm,Tb)
        Tavg_prime = (Tm+Tn)/2
        return Tavg_prime - Tb
    elif Tu < Tavg:
        return Tu - Tb

def growing_degree_days_DTT_linearpeaked(Tmin,Tmax,Tb,Tu,Topt):
    """
    Calculates the daily thermal time from the minimum daily temperature, maximum daily
    temperature, and the cardinal temperatures that describe the daily thermal time 
    temperature response model. The thermal time model assumes a linear increase from 
    Tb to Topt, and a linear decrease from Topt to Tu (triangle shaped response). 

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celcsius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)

    Topt : float
        Thermal optimum temperature (degrees Celsius)

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celsius)
    """
    Tavg = (Tmin+Tmax)/2
    if Tavg < Tb:
        return 0
    elif (Tavg > Tb) and (Tavg < Topt):
        return Tavg - Tb
    elif (Tavg >= Topt) and (Tavg < Tu):
        slope = Topt/(Tu-Topt)
        return Topt - slope*(Tavg - Topt)
    elif (Tavg >= Tu):
        return 0

def growing_degree_days_DTT_linear4(Tmin,Tmax,Tb,Tu,Topt):
    """
    Calculates the daily thermal time (DTT) using a modified version of the linear
    "Method 1" in McMaster and Wilhelm (1997, doi:10.1016/S0168-1923(97)00027-0)
    that includes an upper threshold limit, at which point where DTT=0. Between Topt
    and Tu the DTT is constant. This function requires the minimum daily temperature,
    maximum daily temperature, and the cardinal temperatures that describe the daily
    thermal time temperature response model.

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celsius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)

    Topt : float
        Thermal optimum temperature (degrees Celsius)

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celsius)
    """
    Tavg = (Tmin+Tmax)/2
    if Tavg < Tb:
        return 0
    elif (Tavg > Tb) and (Tavg < Topt):
        return Tavg - Tb
    elif (Tavg >= Topt) and (Tavg < Tu):
        return Topt - Tb
    elif (Tavg >= Tu):
        return 0

def MinQuadraticSmooth(x, y, eta=0.99):
    # Ensuring x, y, and eta can be numpy arrays for vector operations
    x = np.asarray(x)
    y = np.asarray(y)
    eta = np.asarray(eta)
    
    z = np.power(x + y, 2) - 4.0 * eta * x * y
    z = np.maximum(z, 1e-18)  # Ensure z doesn't go below 1e-18
    mins = (x + y - np.sqrt(z)) / (2.0 * eta)
    return mins
