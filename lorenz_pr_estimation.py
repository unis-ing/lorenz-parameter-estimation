import numpy as np
import dedalus.public as de
from dedalus.core.operators import GeneralFunction
import h5py as h5
from rules import *
import time

de.logging_setup.rootlogger.setLevel('ERROR')

NS = 2

class LPE:
	def __init__(self, PR=10, RA=28, B=8/3):
		"""
			PR : true Prandtl
			RA : true Rayleigh
			B  : true physical parameter
			pr : initial Prandtl guess
			mu : nudging parameter
			domain, problem, solver : dedalus objects
			theta, rho, d : (for rule0_c1)  thresholds for update rule
		"""

		self.PR = PR
		self.RA = RA
		self.B = B

		basis = de.Chebyshev('s', NS, interval=(0,1), dealias=3/2)
		domain = de.Domain([basis], grid_dtype=np.float64)

		self.basis = basis

		problem = de.IVP(domain, variables=['x', 'y', 'z', 'u', 'v', 'w', 'xt', 'yt', 'zt', 'ut', 'vt', 'wt'])
		problem.parameters['PR'] = PR
		problem.parameters['ra'] = RA
		problem.parameters['b'] = B

		self.domain = domain
		self.problem = problem

	def simulate(self, pr=2, mu=30, dt=1/2048, stop_iteration=1e3, pr_rule='no_update', nudge='x',
				datafile='initial_data/lorenz_data.h5', filename='analysis', **kwargs):
		"""
			pr : initial Prandtl number for assimilating system
			mu : nudging parameter
			dt : timestep
			stop_iteration : length of simulation
			pr_rule : (str) name of function that returns next pr
			nudge : (str) coordinate(s) to nudge. Must be some combination of 'x', 'y', 'z'. Ex) 'x', 'xz', etc.
			datafile : (str) h5 file. The last triple (x,y,z) in the time series will be taken as IC for lorenz equations.
			filename : (str) name of analysis file
			kwargs : extra arguments for pr_rule
		"""
		self.pr = pr
		self.mu = mu

		domain = self.domain
		problem = self.problem
		problem.parameters['mu'] = mu

		# get data from datafile
		file = h5.File(datafile, 'r')
		tasks = file['tasks']
		x0 = np.array(tasks['x'])[-1][-1]
		y0 = np.array(tasks['y'])[-1][-1]
		z0 = np.array(tasks['z'])[-1][-1]
		file.close()

		#############################
		# set pr_rule
		#############################
		if pr_rule == 'rule0_c1':
			F = rule0_c1
			self.theta = kwargs['theta']
			self.rho = kwargs['rho']
			self.d = kwargs['d']
			thetalist = []
			rholist = []
		elif pr_rule == 'rule1_c1':
			F = rule1_c1
			self.theta = kwargs['theta']
			self.rho = kwargs['rho']
			self.d = kwargs['d']
			thetalist = []
			rholist = []
			self.last = 0
			self.P = kwargs['P']
		elif pr_rule == 'no_update':
			pass
		else:
			print('Update rule has not been set. Exiting.')
			return

		# add pr to problem parameters
		if pr_rule == 'no_update':
			problem.parameters['pr'] = pr
		else:
			problem.parameters['pr'] = de.operators.GeneralFunction(domain, layout='g', func=F, args=[])

		# original lorenz equations
		problem.add_equation('dt(x) = -PR*x + PR*y')
		problem.add_equation('dt(y) + y = -PR*x - x*z')
		problem.add_equation('dt(z) = -b*z + x*y - b*(ra+PR)')

		# nudged lorenz equations
		eqns = ['dt(u) = -pr*u + pr*v', 'dt(v) + v = -pr*u - u*w', 'dt(w) = -b*w + u*v - b*(ra+pr)']
		if 'x' in nudge:
			eqns[0] += '- mu*(u-x)'
		if 'y' in nudge:
			eqns[1] += '- mu*(v-y)'
		if 'z' in nudge:
			eqns[2] += '- mu*(w-z)'
		for eqn in eqns:
			problem.add_equation(eqn)

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
		if pr_rule == 'rule0_c1' or pr_rule == 'rule1_c1':
			prlist = []
			args = [pr, solver, self]
			if pr_rule == 'rule1_c1':
				args.append({})
		if pr_rule == 'rule1_c1':
			prev_pr = pr
		if pr_rule != 'no_update':
			problem.parameters['pr'].args = args
			problem.parameters['pr'].original_args = args
			self.rule_args = args

		# set up analysis
		analysis = solver.evaluator.add_file_handler(filename=filename, iter=1)
		analysis.add_system(solver.state, layout='g')

		solver.stop_sim_time = np.inf
		solver.stop_iteration = stop_iteration
		start = time.time()

		while solver.ok:
			solver.step(dt)

			# store pr
			if pr_rule != 'no_update':
				current_pr = val(problem.parameters['pr'])
				prlist.append(current_pr)

			#############################
			# update arg_rule arguments
			#############################
			if pr_rule != 'no_update': # applies to all
				args[0] = current_pr
			if pr_rule == 'rule0_c1' or pr_rule == 'rule1_c1':
				thetalist.append(self.theta)
				rholist.append(self.rho)
			if pr_rule == 'rule1_c1':
				if current_pr != prev_pr:
					self.last = 0
					prev_pr = current_pr
				else:
					self.last += 1

			percent = (solver.iteration/stop_iteration)*100
			if percent % 10 == 0:
				prerr = abs(self.PR-current_pr)
				print("{0:.0f}% complete. {1:.2f} sec elapsed. Abs pr error: {2:.8e}.".format(percent, time.time()-start, prerr))

		print('Total runtime: {:.4f} sec.'.format(time.time()-start))

		# write pr to .h5
		path = '/'.join([filename, filename + '_s1', filename + '_s1_p0.h5'])
		f = h5.File(path, 'a')
		f.create_dataset('tasks/pr', data=prlist)
		if pr_rule == 'rule0_c1' or pr_rule == 'rule1_c1':
			f.create_dataset('tasks/theta', data=thetalist)
			f.create_dataset('tasks/rho', data=rholist)
		f.close()

		# adding comment