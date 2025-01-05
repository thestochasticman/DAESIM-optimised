from time import time
import numpy as np

def get_time_and_output_previous():
  from daesim.biophysics_funcs import diurnal_temperature as f
  start = time()
  # output = f(np.array([5.0]), np.array([6.0]), 6.5, 7.5, 1.0)
  end = time()
  return (end - start), 1

def get_time_and_output_new():
  from daesim_optimised.biophysics_funcs import diurnal_temperature as f
  from numpy import arange
  start = time()
  # output = f(10.0, 30.0, 6.5, 20.25, arange(0, 24, 0.5))
  output = f(np.array([5.0]), np.array(6.0), 6.5, 7.5, 1.0)
  end = time()
  return (end - start), output

def t():
  from numpy import allclose
  run_time_old, output_old = get_time_and_output_previous()
  run_time_new, output_new = get_time_and_output_new()

  if allclose(output_new, output_old):
    print('outputs are same')
  else:
    print('ouputs are not same')
    return False

  if run_time_new < run_time_old:
    print('optimsiation_succeeded')
    print('run time old', run_time_old)
    print('run time new', run_time_new)
    print(run_time_old/run_time_new, 'times faster')
  else:
    print('optimisation_failed')
    print('run time old', run_time_old)
    print('run time new', run_time_new)
    print(run_time_old/run_time_new, 'times faster')
    return False

  return True

  
if __name__ == '__main__':
  print(t())
