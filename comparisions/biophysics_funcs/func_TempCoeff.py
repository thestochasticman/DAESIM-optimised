from time import time

PATH_DF_FORCING = '/g/data/xe2/ya6227/DAESIM/example_dfs/DAESim_forcing_Milgadara_2018.csv'

def prepare_inputs_for_function():
  from daesim_optimised.plant import PlantModuleCalculator
  from pandas import read_csv
  import numpy as np

  df = read_csv(PATH_DF_FORCING)
  _airTempC = np.linspace(-25, 60, 1000)
  Plant1 = PlantModuleCalculator(mortality_constant=0.002, dayLengRequire=12)
  return _airTempC, Plant1.optTemperature

def get_time_funcTempCoeff(_airTempC, opt_temperature):
    
  from daesim_optimised.biophysics_funcs import func_TempCoeff
  start = time()
  output = func_TempCoeff(_airTempC, opt_temperature)
  end = time()
  return (end - start), output

def get_time_funcTempCoeff_optimised(_air_TempC, opt_temperature):
  import numpy as np
  from daesim_optimised.biophysics_funcs_optimised import func_TempCoeff as func_TempCoeff_optimised

  start = time()
  output = func_TempCoeff_optimised(_air_TempC, opt_temperature)
  end = time()
  return (end - start), output


def main():
  _airTempC, opt_temperature = prepare_inputs_for_function()
  time_old, output_old = get_time_funcTempCoeff(_airTempC, opt_temperature)
  time_new, output_new = get_time_funcTempCoeff_optimised(_airTempC, opt_temperature)
  print(time_old, time_new)
  if time_new < time_old:
    print('optimsiation_succeeded')
    print(time_old/time_new)
  else:
    print('optimisation_failed')

  print((output_new-output_old))

  
if __name__ == '__main__':
  main()
