"""
helper functions for the LPE class.

"""
import dedalus.public as de
import h5py as h5
import numpy as np
import os.path
from shutil import copyfile

de.logging_setup.rootlogger.setLevel('ERROR')

def get(solver, s):
	"""
	Input
		s : string which is the name of one of the solver states
	Output
		value at 0th place of state
	"""
	return val(solver.state[s])

def val(state):
	"""
	Input
		state : solver.state attribute
	"""
	return state['g'][0]

def log10mean(arr):
	return np.log10(np.mean(arr))

def nudged_eqns(nudge):
	"""
	return nudged lorenz equations.
	"""
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
	return 'initial_data/PR_' + str(PR) + '_RA_' + str(RA) + '.h5'

def get_th_path(PR, RA, mu):
	return 'thresholds/PR_' + str(PR) + '_RA_' + str(RA) + '_mu_' + str(mu) + '.h5'

def make_initial_data(PR, RA, B, NS, dt):
	"""
	Make initial data by solving Lorenz equations for 5 units of time and write to
	the initial_data folder.
	"""
	basis = de.Chebyshev('s', NS, interval=(0,1), dealias=3/2)
	domain = de.Domain([basis], grid_dtype=np.float64)

	problem = de.IVP(domain, variables=['x', 'y', 'z'])
	problem.parameters['PR'] = PR
	problem.parameters['RA'] = RA
	problem.parameters['B'] = B

	# lorenz equations
	problem.add_equation('dt(x) = -PR*x + PR*y')
	problem.add_equation('dt(y) + y = -PR*x - x*z')
	problem.add_equation('dt(z) = -B*z + x*y - B*(RA+PR)')

	# build solver
	solver = problem.build_solver(de.timesteppers.RK443)
	
	x, y, z = solver.state['x'], solver.state['y'], solver.state['z']
	x['g'] = np.full(NS, fill_value=10)
	y['g'] = np.full(NS, fill_value=10)
	z['g'] = np.full(NS, fill_value=10)	

	# run for 5 units of time
	stop_it = 5 // dt + 1
	solver.stop_iteration = stop_it

	while solver.ok:
		solver.step(dt)

	folder = 'initial_data'
	fname = 'PR_' + str(PR) + '_RA_' + str(RA)
	path = folder + '/' + fname + '.h5'
	f = h5.File(path, 'a')
	f.create_dataset('x', data=np.array([val(x)]))
	f.create_dataset('y', data=np.array([val(y)]))
	f.create_dataset('z', data=np.array([val(z)]))
	f.close()
