import numpy as np
import dedalus.public as de
from dedalus.core.operators import GeneralFunction
import h5py as h5
import time

de.logging_setup.rootlogger.setLevel('ERROR')

NS = 2

def get(solver, s):
	"""
	Input
		s : string which is the name of one of the solver states
	Output
		value at 0th place of state
	"""
	return solver.state[s]['g'][0]

def val(state):
	"""
	Input
		state : solver.state attribute
	"""
	return state['g'][0]

def norm(*args):
	squared = np.apply_along_axis(np.power, 0, args, 2)
	summed = np.sum(squared)
	return summed**0.5

class LPE:
	def __init__(self, PR=10, RA=28, B=8/3):

		self.PR = PR
		self.RA = RA
		self.B = B

		basis = de.Chebyshev('s', NS, interval=(0,1), dealias=3/2)
		domain = de.Domain([basis], grid_dtype=np.float64)

		problem = de.IVP(domain, variables=['x', 'y', 'z', 'u', 'v', 'w', 'xt', 'yt', 'zt', 'ut', 'vt', 'wt'])
		problem.parameters['PR'] = PR
		problem.parameters['ra'] = RA
		problem.parameters['b'] = B

		self.domain = domain
		self.problem = problem

	def rule0(self, pr, mu, x, y, u, v):
		return (pr*u - pr*v + mu*(u-x)) / (x-y)

	def rule0_c1(self, *args):
		pr = args[0]
		solver = args[1]
		mu = args[2]

		x = get(solver, 'x')
		xt = get(solver, 'xt')
		u = get(solver, 'u')
		ut = get(solver, 'ut')

		uerr = abs(x-u)
		uterr = abs(xt-ut)

		if (uerr<=self.theta) and (uterr<=self.rho) and (uerr>0) and (uterr>0):
			# decrease threshold
			self.theta *= self.d
			self.rho *= self.d
			y = get(solver, 'y')
			v = get(solver, 'v')
			return self.rule0(pr, mu, x, y, u, v)
		else:
			return pr

	def nudge_data(self, *args):
		data = args[0]
		# print(data)
		return data

	def simulate(self, window=3/2048, pr=2, mu=30, dt=1/2048, stop_iteration=1e3, pr_rule='no_update', nudge='x',
				datafile='initial_data/lorenz_data.h5', savefile='analysis', **kwargs):
		"""
			window : time between observations. Must be >= dt.
			pr : initial Prandtl number for assimilating system
			mu : nudging parameter
			dt : timestep
			stop_iteration : length of simulation
			pr_rule : (str) name of function that returns next pr
			nudge : (str) coordinate(s) to nudge. Must be some combination of 'x', 'y', 'z'. Ex) 'x', 'xz', etc.
			datafile : (str) h5 file. The last triple (x,y,z) in the time series will be taken as IC for lorenz equations.
			savefile : (str) name of analysis file
			kwargs : extra arguments for pr_rule
		"""
		domain = self.domain
		problem = self.problem
		problem.parameters['mu'] = mu

		W = np.ceil(window/dt).astype('int')

		# get data from datafile
		file = h5.File(datafile, 'r')
		tasks = file['tasks']
		x0 = np.array(tasks['x'])[-1]
		y0 = np.array(tasks['y'])[-1]
		z0 = np.array(tasks['z'])[-1]
		file.close()

		#############################
		# set pr_rule
		#############################
		if pr_rule == 'rule0_c1':
			F = self.rule0_c1
			self.theta = kwargs['theta']
			self.rho = kwargs['rho']
			self.d = kwargs['d']
		elif pr_rule == 'no_update':
			pass
		else:
			print('Update rule has not been set. Exiting.')
			return

		if pr_rule == 'no_update':
			problem.parameters['pr'] = pr
		else:
			problem.parameters['pr'] = de.operators.GeneralFunction(domain, layout='g', func=F, args=[])

		# nudged lorenz equations
		eqns = ['dt(u) = -pr*u + pr*v', 'dt(v) + v = -pr*u - u*w', 'dt(w) = -b*w + u*v - b*(ra+pr)']
		for coord in nudge:
			problem.parameters['F' + coord] = de.operators.GeneralFunction(domain, layout='g', func=self.nudge_data, args=[])
			if coord == 'x':
				eqns[0] += '- mu*(u-Fx)'
			elif coord == 'y':
				eqns[1] += '- mu*(v-Fy)'
			elif coord == 'z':
				eqns[2] += '- mu*(w-Fz)'
			else:
				print('Invalid coordinate. Exiting.')
				return

		for eqn in eqns:
			problem.add_equation(eqn)

		# original lorenz equations
		problem.add_equation('dt(x) = -PR*x + PR*y')
		problem.add_equation('dt(y) + y = -PR*x - x*z')
		problem.add_equation('dt(z) = -b*z + x*y - b*(ra+PR)')

		# d/dt equations
		problem.add_equation('dt(x) - xt = 0')
		problem.add_equation('dt(y) - yt = 0')
		problem.add_equation('dt(z) - zt = 0')
		problem.add_equation('dt(u) - ut = 0')
		problem.add_equation('dt(v) - vt = 0')
		problem.add_equation('dt(w) - wt = 0')

		solver = self.problem.build_solver(de.timesteppers.RK443)
		self.solver = solver
		x, y, z = solver.state['x'], solver.state['y'], solver.state['z']
		u, v, w = solver.state['u'], solver.state['v'], solver.state['w']
		xt, ut = solver.state['xt'], solver.state['ut']

		# initialize lorenz system
		x['g'] = np.full(NS, fill_value=x0)
		y['g'] = np.full(NS, fill_value=y0)
		z['g'] = np.full(NS, fill_value=z0)


		u['g'] = np.zeros(NS)
		v['g'] = np.zeros(NS)
		w['g'] = np.zeros(NS)

		#############################
		# set arguments for pr_rule
		#############################
		if pr_rule == 'rule0_c1':
			args = [pr, solver, mu]
			thetalist = []
			rholist = []
		if pr_rule != 'no_update':
			problem.parameters['pr'].args = args
			problem.parameters['pr'].original_args = args

		####################################
		fxargs = [0]
		problem.parameters['Fx'].args = fxargs
		problem.parameters['Fx'].original_args = fxargs

		# set up analysis
		analysis = solver.evaluator.add_file_handler(filename=savefile, iter=1)
		analysis.add_system(solver.state, layout='g')

		prlist = []

		solver.stop_sim_time = np.inf
		solver.stop_iteration = stop_iteration
		start = time.time()

		while solver.ok:
			solver.step(dt)
			# store pr & anything else
			if pr_rule != 'no_update':
				current_pr = val(problem.parameters['pr'])
			else:
				current_pr = pr
			prlist.append(current_pr)

			#############################
			# update arg_rule arguments
			#############################
			if pr_rule != 'no_update': # applies to all
				args[0] = current_pr
			if pr_rule == 'rule0_c1':
				thetalist.append(self.theta)
				rholist.append(self.rho)

			#############################
			# update observations
			#############################
			if solver.iteration % W == 0:
				for coord in nudge:
					if coord == 'x':
						data = val(x)
					elif coord == 'y':
						data = val(y)
					elif coord == 'z':
						data = val(z)
					fxargs[0] = data

			percent = (solver.iteration/solver.stop_iteration)*100
			if percent % 10 == 0:
				prerr = abs(self.PR-current_pr)
				print("{0:.0f}% complete. {1:.2f} sec elapsed. Abs pr error: {2:.8e}.".format(percent, time.time()-start, prerr))

		print('Total runtime: {:.4f} sec.'.format(time.time()-start))

		# write pr to .h5
		p = '/'.join([savefile, savefile + '_s1', savefile + '_s1_p0.h5'])
		f = h5.File(p, 'a')
		f.create_dataset('tasks/pr', data=prlist)
		f.close()

		if pr_rule == 'rule0_c1':
			return np.array(thetalist), np.array(rholist)