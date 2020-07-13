import numpy as np
import dedalus.public as de
from dedalus.core.operators import GeneralFunction
import h5py as h5
from lpe_helpers import *
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
		problem.parameters['RA'] = RA
		problem.parameters['B'] = B

		self.basis = basis
		self.domain = domain
		self.problem = problem

	def setup_rule(self, **kwargs):
		"""
		store rule args as class attributes. return the function and arguments associated to the given rule.
			rule  : (str) name of existing rule function
			theta : position error threshold
			rho   : velocity error threshold
			d     : exponential decay factor for thresholds
			Pc    : P* in writeup
			P     : P in writeup
		"""
		rule = self.rule
		rn = rule[4] # rule number
		cn = rule[-1] # condition number

		if rn == '0' or rn == '1':
			self.da = kwargs['da']
			self.db = kwargs['db']

			if rn == '1' and self.calc_thold: # option to calc tholds
				theta, rho = self.calculate_thresholds_rule1()
				self.theta = theta
				self.rho = rho
			else:
				self.theta = kwargs['theta']
				self.rho = kwargs['rho']

			# initialize empty lists for storage
			self.thetalist = []
			self.rholist = []

			if rn == '1':
				self.P = 0
				self.Pc = kwargs['Pc']
				if cn == '2':
					self.M = kwargs['M']

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

	def update_ruleargs(self, ruleargs):
		"""
		updates rulesargs at each iteration. also stores updated rule args.
		"""
		rule = self.rule
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

	def calculate_thresholds_rule1(self):
		pr0 = self.pr0
		mu = self.mu
		th_path = get_th_path(self.PR, self.RA, self.mu)

		if os.path.exists(th_path):
			f = h5.File(th_path, 'r')
			theta = np.array(f['theta'])[0]
			rho = np.array(f['rho'])[0]
			f.close()
		else:
			print('Solving nudged equations to calculate thresholds.')
			self.make_thresholds_data()
			p = 'analysis/analysis_s1/analysis_s1_p0.h5'

			f  = h5.File(p, 'r')
			t  = f['tasks']
			x  = np.array(t['x'])[:,0]
			u  = np.array(t['u'])[:,0]
			v  = np.array(t['v'])[:,0]
			xt = np.array(t['xt'])[:,0]
			ut = np.array(t['ut'])[:,0]
			f.close()

			uerr = abs(x - u)
			uterr = abs(xt - ut)

			guess = rule1(pr=pr0, mu=mu, x=x, u=u, v=v)
			guesserr = abs(self.PR - guess)
			# only consider errors past 1,000th index
			i = np.where(guesserr == min(guesserr))
			minuerr = uerr[i]
			minuterr = uterr[i]

			floorlog = np.floor(np.log10(minuerr))
			roundreciprocal = np.ceil(minuerr * 10**(-floorlog))
			theta = roundreciprocal * 10**floorlog
			theta = theta[0]

			floorlog = np.floor(np.log10(minuterr))
			roundreciprocal = np.ceil(minuterr * 10**(-floorlog))
			rho = roundreciprocal * 10**floorlog
			rho = rho[0]

			# write to file
			f = h5.File(th_path, 'a')
			f.create_dataset('theta', data=np.array([theta]))
			f.create_dataset('rho', data=np.array([rho]))
			f.close()

		return theta, rho

	def make_thresholds_data(self):
		"""
		Make data for calculating thresholds by solving nudged Lorenz equations for 5 units of time.
		"""
		PR = self.PR
		RA = self.RA
		B = self.B
		NS = self.NS
		pr0 = self.pr0
		mu = self.mu
		dt = self.dt
		nudge = self.nudge
		ts = self.ts

		basis = de.Chebyshev('s', NS, interval=(0,1), dealias=3/2)
		domain = de.Domain([basis], grid_dtype=np.float64)

		problem = de.IVP(domain, variables=['x', 'y', 'z', 'u', 'v', 'w', 'xt', 'yt', 'zt', 'ut', 'vt', 'wt'])
		problem.parameters['PR'] = PR
		problem.parameters['RA'] = RA
		problem.parameters['B'] = B
		problem.parameters['mu'] = mu
		problem.parameters['pr'] = pr0

		# lorenz equations
		problem.add_equation('dt(x) = -PR*x + PR*y')
		problem.add_equation('dt(y) + y = -PR*x - x*z')
		problem.add_equation('dt(z) = -B*z + x*y - B*(RA+PR)')

		# nudged lorenz equations
		eqns = nudged_eqns(nudge)
		[problem.add_equation(eqn) for eqn in eqns]

		# d/dt equations
		problem.add_equation('dt(x) - xt = 0')
		problem.add_equation('dt(y) - yt = 0')
		problem.add_equation('dt(z) - zt = 0')
		problem.add_equation('dt(u) - ut = 0')
		problem.add_equation('dt(v) - vt = 0')
		problem.add_equation('dt(w) - wt = 0')

		# build solver
		solver = problem.build_solver(de.timesteppers.RK443)

		# set ics
		ic_path = get_ic_path(PR, RA)
		f = h5.File(ic_path, 'r')
		x0 = np.array(f['x'])
		y0 = np.array(f['y'])
		z0 = np.array(f['z'])
		f.close()

		x, y, z = solver.state['x'], solver.state['y'], solver.state['z']
		x['g'] = np.full(NS, fill_value=x0)
		y['g'] = np.full(NS, fill_value=y0)
		z['g'] = np.full(NS, fill_value=z0)	

		# set up analysis
		analysis = solver.evaluator.add_file_handler(filename='analysis', iter=1, max_size=2**(64))
		analysis.add_system(solver.state, layout='g')

		# run for 5 units of time
		stop_it = 5 // dt + 1
		solver.stop_iteration = stop_it

		while solver.ok:
			solver.step(dt)

	def simulate(self, pr0=20, mu=30, dt=1/2048, stop_it=1000, rule='no_update', nudge='x', ts=de.timesteppers.RK443,
				 outfile='analysis', print_every=10, calc_thold=False, **kwargs):
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
		self.dt = dt
		self.rule = rule
		self.nudge = nudge
		self.ts = ts
		self.calc_thold = calc_thold

		domain = self.domain
		problem = self.problem

		# initialize lorenz system
		PR, RA, B, NS = self.PR, self.RA, self.B, self.NS
		ic_path = get_ic_path(PR, RA)
		self.ic_path = ic_path
		if not os.path.exists(ic_path):
			print('Making initial data.')
			make_initial_data(PR, RA, B, NS, dt)

		# get pr update function
		F, ruleargs = self.setup_rule(**kwargs)
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
		problem.add_equation('dt(z) = -B*z + x*y - B*(RA+PR)')

		# nudged lorenz equations
		eqns = nudged_eqns(nudge)
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

		f = h5.File(ic_path, 'r')
		x0 = np.array(f['x'])
		y0 = np.array(f['y'])
		z0 = np.array(f['z'])
		f.close()

		x, y, z = solver.state['x'], solver.state['y'], solver.state['z']
		x['g'] = np.full(NS, fill_value=x0)
		y['g'] = np.full(NS, fill_value=y0)
		z['g'] = np.full(NS, fill_value=z0)

		# set up analysis
		analysis = solver.evaluator.add_file_handler(filename=outfile, iter=1, max_size=2**(64))
		analysis.add_system(solver.state, layout='g')

		solver.stop_sim_time = np.inf
		solver.stop_iteration = stop_it
		start = time.time()

		while solver.ok:
			solver.step(dt)

			if F != 1:
				self.update_ruleargs(ruleargs)

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
