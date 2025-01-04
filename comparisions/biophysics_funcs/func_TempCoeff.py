from time import time

PATH_DF_FORCING = '/home/y/DAESIM2-ANALYSIS-DATA/DAESim_forcing_Milgadara_2018.csv'

def prepare_inputs_for_function():
  from daesim_optimised.plant import PlantModuleCalculator
  from pandas import read_csv
  import numpy as np

  df = read_csv(PATH_DF_FORCING)
  _airTempC = np.linspace(-25, 60, 1000)
  Plant1 = PlantModuleCalculator(mortality_constant=0.002, dayLengRequire=12)
  return _airTempC, Plant1.optTemperature

def get_time_and_output_previous(_airTempC, opt_temperature):
  from daesim.biophysics_funcs import func_TempCoeff as f
  start = time()
  output = f(_airTempC, opt_temperature)
  end = time()
  return (end - start), output

def get_time_and_output_new(_air_TempC, opt_temperature):
  from daesim_optimised.biophysics_funcs import func_TempCoeff as f
  start = time()
  output = f(_air_TempC, opt_temperature)
  end = time()
  return (end - start), output

def t():
  from numpy import allclose
  _airTempC, opt_temperature = prepare_inputs_for_function()
  run_time_old, output_old = get_time_and_output_previous(_airTempC, opt_temperature)
  run_time_new, output_new = get_time_and_output_new(_airTempC, opt_temperature)

  if run_time_new < run_time_old:
    print('optimsiation_succeeded')
    print('run time old', run_time_old)
    print('run time new', run_time_new)
    print( run_time_old/run_time_new, 'times faster')
  else:
    print('optimisation_failed')
    return False

  if allclose(output_new, output_old):
    print('outputs are same')
    return True
  else:
    print('ouputs are not same')
    return False
  
if __name__ == '__main__':
  t()
