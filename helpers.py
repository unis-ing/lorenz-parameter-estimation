"""
helper functions for the LPE class.

"""
import warnings 
import dedalus.public as de
import h5py as h5
import numpy as np
import os.path
from shutil import copyfile

de.logging_setup.rootlogger.setLevel('ERROR')

# suppress warning from a deprecated h5py function being used in dedalus source code
warnings.filterwarnings("ignore", category=h5.h5py_warnings.H5pyDeprecationWarning)

scheme = de.timesteppers.RK443

result_data_folder = 'result_data/'
initial_data_folder = 'initial_data/'
test_data_folder = 'test_data/'
threshold_folder = 'thresholds/'
analysis_name = 'analysis'
analysis_temp_path = 'analysis/analysis_s1/analysis_s1_p0.h5'

lorenz_equations = ['dt(x) = -PR*x + PR*y', 
					'dt(y) + y = -PR*x - x*z', 
					'dt(z) = -B*z + x*y - B*(RA+PR)']

xt_equations = ['dt(x) - xt = 0',
				'dt(y) - yt = 0',
				'dt(z) - zt = 0']

ut_equations = ['dt(u) - ut = 0',
 				'dt(v) - vt = 0',
 				'dt(w) - wt = 0']

def get(solver, s):
	"""
		solver : solver object
		s : (str) name of a solver state
	"""
	return val(solver.state[s])

def val(state):
	return state['g'][0]

def log10mean(arr):
	return np.log10(np.mean(arr))

def nudged_lorenz_equations(nudge):
	eqns = ['dt(u) = -pr*u + pr*v', 
			'dt(v) + v = -pr*u - u*w', 
			'dt(w) = -B*w + u*v - B*(RA+pr)']
	if 'x' in nudge:
		eqns[0] += '- mu*(u-x)'
	if 'y' in nudge:
		eqns[1] += '- mu*(v-y)'
	if 'z' in nudge:
		eqns[2] += '- mu*(w-z)'
	return eqns

def get_ic_path(PR, RA):
	ic_name = '_'.join(map(str, ['PR', PR, 'RA', RA]))
	return initial_data_folder + ic_name + '.h5'

def get_th_path(PR, RA, mu):
	th_name = '_'.join(map(str, ['PR', PR, 'RA', RA, 'mu', mu]))
	return threshold_folder + th_name + '.h5'

def make_initial_data(PR, RA, B, NS):
	"""
	Make initial data by solving Lorenz equations for 5 units of time and write to
	the initial_data folder.
	"""
	dt = 0.001
	basis = de.Chebyshev('s', NS, interval=(0,1), dealias=3/2)
	domain = de.Domain([basis], grid_dtype=np.float64)

	problem = de.IVP(domain, variables=['x', 'y', 'z'])
	problem.parameters['PR'] = PR
	problem.parameters['RA'] = RA
	problem.parameters['B'] = B

	# add lorenz equations
	[problem.add_equation(eqn) for eqn in lorenz_equations]

	solver = problem.build_solver(scheme)

	x, y, z = solver.state['x'], solver.state['y'], solver.state['z']
	x['g'] = np.full(NS, fill_value=10)
	y['g'] = np.full(NS, fill_value=10)
	z['g'] = np.full(NS, fill_value=10)	

	# run for 5 units of time
	stop_it = 5 // dt + 1
	solver.stop_iteration = stop_it

	while solver.ok:
		solver.step(dt)

	f = h5.File(get_ic_path(PR, RA), 'a')
	f.create_dataset('x', data=np.array([val(x)]))
	f.create_dataset('y', data=np.array([val(y)]))
	f.create_dataset('z', data=np.array([val(z)]))
	f.close()

def make_threshold_data(PR, RA, pr, mu, B, NS, dt, nudge):
	"""
	Make data for calculating thresholds by solving nudged Lorenz equations with constant
	pr for 10 units of time. Assumes ic data exists.
		"""	
	basis = de.Chebyshev('s', NS, interval=(0,1), dealias=3/2)
	domain = de.Domain([basis], grid_dtype=np.float64)
	problem = de.IVP(domain, variables=['x', 'y', 'z',
										'xt', 'yt', 'zt',
										'u', 'v', 'w',
										'ut', 'vt', 'wt'])
	problem.parameters['PR'] = PR
	problem.parameters['RA'] = RA
	problem.parameters['B'] = B
	problem.parameters['mu'] = mu
	problem.parameters['pr'] = pr

	# add lorenz equations
	[problem.add_equation(eqn) for eqn in lorenz_equations]

	# add nudged lorenz equations
	[problem.add_equation(eqn) for eqn in nudged_lorenz_equations(nudge)]

	# add dt equations
	[problem.add_equation(eqn) for eqn in xt_equations]
	[problem.add_equation(eqn) for eqn in ut_equations]

	# build solver
	solver = problem.build_solver(scheme)

	# set ics
	f = h5.File(get_ic_path(PR, RA), 'r')
	x0 = np.array(f['x'])
	y0 = np.array(f['y'])
	z0 = np.array(f['z'])
	f.close()

	x, y, z = solver.state['x'], solver.state['y'], solver.state['z']
	x['g'] = np.full(NS, fill_value=x0)
	y['g'] = np.full(NS, fill_value=y0)
	z['g'] = np.full(NS, fill_value=z0)

	# set up analysis
	analysis = solver.evaluator.add_file_handler(filename=analysis_name, iter=1, max_size=2**(64))
	analysis.add_system(solver.state, layout='g')

	# run for 5 units of time
	stop_it = 10 // dt + 1
	solver.stop_iteration = stop_it

	while solver.ok:
		solver.step(dt)

"""
 for parameter tuning
"""

def running_mean(x, N): # https://stackoverflow.com/a/27681394
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def M_data(A, N):
	"""
		Use A = uerr and N = Pc in order to determine M.
	"""
	rm = running_mean(A, N)
	logrm = np.log10(rm)
	return logrm[1:] - logrm[:-1]
