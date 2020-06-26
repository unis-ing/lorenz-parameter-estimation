import numpy as np
import dedalus.public as de
from dedalus.core.operators import GeneralFunction
import h5py as h5
from rules import *
import time

de.logging_setup.rootlogger.setLevel('ERROR')


class LPE:
	def __init__(self, PR=10, RA=28, B=8/3, NS=2):
		"""
			PR : Prandtl number
			RA : Rayleigh number
			B  : physical parameter
			NS : number of spectral basis modes
		"""

		self.PR = PR
		self.RA = RA
		self.B = B
		self.NS = NS

		basis = de.Chebyshev('s', NS, interval=(0,1), dealias=3/2)
		domain = de.Domain([basis], grid_dtype=np.float64)

		problem = de.IVP(domain, variables=['x', 'y', 'z', 'u', 'v', 'w', 'xt', 'yt', 'zt', 'ut', 'vt', 'wt'])
		problem.parameters['PR'] = PR
		problem.parameters['ra'] = RA
		problem.parameters['b'] = B

		self.basis = basis
		self.domain = domain
		self.problem = problem

	def setup_rule(self, rule, **kwargs):
		"""
		store rule args as class attributes. return the function and arguments associated to the given rule.
			rule  : (str) name of existing rule function
			theta : position error threshold
			rho   : velocity error threshold
			d     : exponential decay factor for thresholds
			Pc    : P* in writeup
			P     : P in writeup
		"""
		# store constant rule args as class attributes
		rn = rule[4] # rule number
		cn = rule[-1] # condition number
		if rn == '0' or rn == '1':
			self.theta = kwargs['theta']
			self.rho = kwargs['rho']
			self.da = kwargs['da']
			self.db = kwargs['db']

			# initialize empty lists for storage
			self.thetalist = []
			self.rholist = []

			if rn == '1':
				self.P = 0
				self.Pc = kwargs['Pc']

		# return rule args
		pr0 = self.pr0
		if rn == '0':
			ruleargs = [self, {0 : pr0}] # dict to track pr
			self.ruleargs = ruleargs
			return rule0_c1, ruleargs

		elif rn == '1':
			if cn == '1':
				ruleargs = [self, {0 : pr0}] # dict to track pr
				self.ruleargs = ruleargs
				return rule1_c1, ruleargs
			elif cn == '2': 
				# dict to track pr and arr to track uerr
				ruleargs = [self, {0 : pr0}, np.full(1 + self.Pc, np.inf)]
				self.ruleargs = ruleargs
				return rule1_c2, ruleargs

		elif rule == 'no_update':
			return 1, 1

		else:
			return 0, 0

	def nudged_eqns(self, nudge):
		"""
		return nudged lorenz equations.
		"""
		eqns = ['dt(u) = -pr*u + pr*v', 
				'dt(v) + v = -pr*u - u*w', 
				'dt(w) = -b*w + u*v - b*(ra+pr)']
		if 'x' in nudge:
			eqns[0] += '- mu*(u-x)'
		if 'y' in nudge:
			eqns[1] += '- mu*(v-y)'
		if 'z' in nudge:
			eqns[2] += '- mu*(w-z)'
		return eqns

	def update_ruleargs(self, rule, ruleargs):
		"""
		updates rulesargs at each iteration. also stores updated rule args.
		"""
		if rule == 'rule0_c1' or rule == 'rule1_c1' or rule == 'rule1_c2':
			self.thetalist.append(self.theta)
			self.rholist.append(self.rho)

			if rule == 'rule1_c1' or rule == 'rule1_c2':
				it = self.solver.iteration
				prs = ruleargs[1]
				prev_pr = prs[it-1]
				curr_pr = prs[it]
				if curr_pr != prev_pr:
					self.P = 0
				else:
					self.P += 1


	def simulate(self, pr0=20, mu=30, dt=1/2048, stop_it=1000, rule='no_update', nudge='x', ts=de.timesteppers.RK443,
				 ic='initial_data/lorenz_data.h5', outfile='analysis', print_every=10, **kwargs):
		"""
			pr0   : initial value for tilde Prandtl
			mu    : nudging parameter
			dt    : time-step
			stop_it : number of iterations
			rule  : update rule for tilde Prandtl
			ts    : Dedalus time stepping scheme
			nudge : (str) array or single string representing nudged coordinate(s)
			ic    : (str) file path to h5 file containing initial conditions
			outfile : (str) file name for analysis to be saved under
			print_every : print progress message every x% to completion
			kwargs: arguments for rule
		"""
		self.pr0 = pr0
		self.mu = mu
		self.rule = rule

		domain = self.domain
		problem = self.problem

		# get pr update function
		F, ruleargs = self.setup_rule(rule, **kwargs)
		if not F: # invalid rule
			print('No rule exists corresponding to the name ', rule, '. Exiting.')
			return

		# add pr to problem namespace
		if F == 1: # no update
			problem.parameters['pr'] = pr0
		else:
			problem.parameters['pr'] = de.operators.GeneralFunction(domain, layout='g', func=F, args=[])
		problem.parameters['mu'] = mu

		# lorenz equations
		problem.add_equation('dt(x) = -PR*x + PR*y')
		problem.add_equation('dt(y) + y = -PR*x - x*z')
		problem.add_equation('dt(z) = -b*z + x*y - b*(ra+PR)')

		# nudged lorenz equations
		eqns = self.nudged_eqns(nudge)
		[problem.add_equation(eqn) for eqn in eqns]

		# d/dt equations
		problem.add_equation('dt(x) - xt = 0')
		problem.add_equation('dt(y) - yt = 0')
		problem.add_equation('dt(z) - zt = 0')
		problem.add_equation('dt(u) - ut = 0')
		problem.add_equation('dt(v) - vt = 0')
		problem.add_equation('dt(w) - wt = 0')

		if F != 1: # for some reason args must be set AFTER equations are added
			problem.parameters['pr'].args = ruleargs
			problem.parameters['pr'].original_args = ruleargs

		# build solver
		solver = problem.build_solver(ts)
		self.solver = solver

		# initialize lorenz system
		file = h5.File(ic, 'r')
		tasks = file['tasks']
		x0 = np.array(tasks['x'])[-1, 0]
		y0 = np.array(tasks['y'])[-1, 0]
		z0 = np.array(tasks['z'])[-1, 0]
		file.close()

		x, y, z = solver.state['x'], solver.state['y'], solver.state['z']
		NS = self.NS
		x['g'] = np.full(NS, fill_value=x0)
		y['g'] = np.full(NS, fill_value=y0)
		z['g'] = np.full(NS, fill_value=z0)

		# set up analysis
		analysis = solver.evaluator.add_file_handler(filename=outfile, iter=1)
		analysis.add_system(solver.state, layout='g')

		solver.stop_sim_time = np.inf
		solver.stop_iteration = stop_it
		start = time.time()

		while solver.ok:
			solver.step(dt)

			if F != 1:
				self.update_ruleargs(rule, ruleargs)

			it = solver.iteration
			percent = (it/stop_it)*100

			# print progress msg
			if percent % print_every == 0:
				progress_msg = '{0:.0f}% complete. {1:.2f} sec elapsed.'.format(percent, time.time()-start)
				if F != 1:
					prs = ruleargs[1]
					curr_pr = prs[it]
					pr_err = abs(self.PR - curr_pr)
					progress_msg += ' Abs pr error: {:.8e}.'.format(pr_err)
				print(progress_msg)

		print('Total runtime: {:.4f} sec.'.format(time.time()-start))

		# write pr to .h5
		if F != 1:
			path = '/'.join([outfile, outfile + '_s1', outfile + '_s1_p0.h5'])
			f = h5.File(path, 'a')

			prs = list(prs.values())[1:]
			f.create_dataset('tasks/pr', data=prs)
			f.create_dataset('tasks/theta', data=self.thetalist)
			f.create_dataset('tasks/rho', data=self.rholist)
			f.close()
