from time import time

def get_time_and_output_previous():
  from numpy import arange as f
  start = time()
  output = f(0, 24, 0.5)
  end = time()
  return (end - start), output

def get_time_and_output_new():
  from daesim_optimised.biophysics_funcs import arange
  start = time()
  output = arange(0, 24, 0.5)
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
  t()
