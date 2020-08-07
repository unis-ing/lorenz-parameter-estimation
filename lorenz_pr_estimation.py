from dedalus.core.operators import GeneralFunction
from helpers import *
from rules import *
import time

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
		problem = de.IVP(domain, variables=['x', 'y', 'z',
											'xt', 'yt', 'zt',
											'u', 'v', 'w',
											'ut', 'vt', 'wt'])
		problem.parameters['PR'] = PR
		problem.parameters['RA'] = RA
		problem.parameters['B'] = B

		self.problem = problem
		self.domain = domain

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
		if rule == 'no_update': return 1, 1

		rn = rule[4] # rule number
		cn = rule[-1] # condition number

		# these args apply to all rules
		self.da = kwargs['da']
		self.db = kwargs['db']

		if not self.calc_thold:
			self.theta = kwargs['theta']
			self.rho = kwargs['rho']
		else:
			if rn == '1': # option to calc tholds. ONLY SETUP FOR rule1
				theta, rho = self.calculate_thresholds_rule1()
				self.theta = theta
				self.rho = rho

		if rn == '1':
			self.P = 0
			self.Pc = kwargs['Pc']
			if cn == '2':
				self.M = kwargs['M']

		# initialize empty lists for storage
		self.thetalist = []
		self.rholist = []

		# return rule args
		pr0 = self.pr0
		if rn == '0':
			ruleargs = [self, {0 : pr0}] # dict to track pr
			self.ruleargs = ruleargs
			return rule0_c1, ruleargs

		elif rn == '1':
			if cn == '0':
				ruleargs = [self, {0 : pr0}] # dict to track pr
				self.ruleargs = ruleargs
				return rule1_c0, ruleargs
			elif cn == '1':
				ruleargs = [self, {0 : pr0}] # dict to track pr
				self.ruleargs = ruleargs
				return rule1_c1, ruleargs
			elif cn == '2': 
				# dict to track pr and arr to track uerr
				ruleargs = [self, {0 : pr0}, np.full(1 + self.Pc, np.inf)]
				self.ruleargs = ruleargs
				return rule1_c2, ruleargs

		else:
			return 0, 0

	def update_ruleargs(self, ruleargs):
		"""
		updates rulesargs at each iteration of main loop. also stores updated rule args.
		"""
		rule = self.rule
		self.thetalist.append(self.theta)
		self.rholist.append(self.rho)

		rn = rule[4]
		if rn == '1':
			it = self.solver.iteration
			prs = ruleargs[1]
			prev_pr = prs[it-1]
			curr_pr = prs[it]
			if curr_pr != prev_pr:
				self.P = 0
			else:
				self.P += 1

	def calculate_thresholds_rule1(self):
		PR, RA, pr0, mu = self.PR, self.RA, self.pr0, self.mu
		B, NS, dt, nudge = self.B, self.NS, self.dt, self.nudge

		th_path = get_th_path(PR, RA, mu)

		if os.path.exists(th_path):
			f = h5.File(th_path, 'r')
			theta = np.array(f['theta'])[0]
			rho = np.array(f['rho'])[0]
			f.close()

		else:
			print('Solving nudged equations to calculate thresholds.')
			make_threshold_data(PR, RA, pr0, mu, B, NS, dt, nudge)

			f  = h5.File(analysis_temp_path, 'r')
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
			i = np.where(guesserr == min(guesserr))
			min_pos_err = uerr[i].item()
			min_vel_err = uterr[i].item()

			# set theta to rounded uerr
			floorlog = np.floor(np.log10(min_pos_err))
			roundreciprocal = np.ceil(min_pos_err * 10**(-floorlog))
			theta = roundreciprocal * 10**int(floorlog)

			# set rho to rounded uterr
			floorlog = np.floor(np.log10(min_vel_err))
			roundreciprocal = np.ceil(min_vel_err * 10**(-floorlog))
			rho = roundreciprocal * 10**int(floorlog)

			# write thresholds to file
			f = h5.File(th_path, 'a')
			f.create_dataset('theta', data=np.array([theta]))
			f.create_dataset('rho', data=np.array([rho]))
			f.close()

		# print(theta)
		# print(rho)

		return theta, rho

	def simulate(self, pr0=20, mu=30, dt=1/2048, stop_it=1000, rule='no_update', 
				 nudge='x', print_every=10, calc_thold=False, **kwargs):
		"""
			pr0   : initial value for tilde Prandtl
			mu    : nudging parameter
			dt    : time-step
			stop_it : number of iterations
			rule  : update rule for tilde Prandtl\
			nudge : (str) array or single string representing nudged coordinate(s)
			print_every : print progress message every x% to completion
			kwargs: arguments for rule
		"""
		self.pr0, self.mu, self.dt, self.rule, self.nudge = pr0, mu, dt, rule, nudge
		self.calc_thold = calc_thold

		domain = self.domain
		problem = self.problem

		# initialize lorenz system
		PR, RA, B, NS = self.PR, self.RA, self.B, self.NS
		ic_path = get_ic_path(PR, RA)
		if not os.path.exists(ic_path):
			print('Making initial data.')
			make_initial_data(PR=PR, RA=RA, B=B, NS=NS)
		self.ic_path = ic_path

		# get pr update function
		F, ruleargs = self.setup_rule(**kwargs)
		if not F: # invalid rule
			print('No rule exists corresponding to the name ', rule, '. Exiting.')
			return

		# add mu, pr to problem namespace
		problem.parameters['mu'] = mu
		if rule == 'no_update':
			problem.parameters['pr'] = pr0
		else:
			problem.parameters['pr'] = de.operators.GeneralFunction(domain, layout='g', 
																	func=F, args=[])

		# add lorenz equations
		[problem.add_equation(eqn) for eqn in lorenz_equations]

		# add nudged lorenz equations
		[problem.add_equation(eqn) for eqn in nudged_lorenz_equations(nudge)]

		# add dt equations
		[problem.add_equation(eqn) for eqn in xt_equations]
		[problem.add_equation(eqn) for eqn in ut_equations]

		if rule != 'no_update': 
			# for some reason args must be set AFTER equations are added
			problem.parameters['pr'].args = ruleargs
			problem.parameters['pr'].original_args = ruleargs

		# build solver
		solver = problem.build_solver(scheme)
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
		analysis = solver.evaluator.add_file_handler(filename=analysis_name, iter=1, 
													 max_size=2**(64))
		analysis.add_system(solver.state, layout='g')

		solver.stop_sim_time = np.inf
		solver.stop_iteration = stop_it
		start = time.time()

		while solver.ok:
			solver.step(dt)

			if rule != 'no_update':
				self.update_ruleargs(ruleargs)

			it = solver.iteration
			percent = (it/stop_it)*100

			# print progress msg
			if percent % print_every == 0:
				progress_msg = '{0:.0f}% complete. {1:.2f} sec elapsed.'.format(percent, time.time()-start)
				if rule != 'no_update':
					prs = ruleargs[1]
					curr_pr = prs[it]
					pr_err = abs(self.PR - curr_pr)
					progress_msg += ' Abs pr error: {:.8e}.'.format(pr_err)
				print(progress_msg)

		print('Total runtime: {:.4f} sec.'.format(time.time()-start))

		# write pr to .h5
		if rule != 'no_update':
			f = h5.File(analysis_temp_path, 'a')
			prs = list(prs.values())[1:]
			f.create_dataset('tasks/pr', data=prs)
			f.create_dataset('tasks/theta', data=self.thetalist)
			f.create_dataset('tasks/rho', data=self.rholist)
			f.close()

	def save_data(self):
		src = analysis_temp_path
		dst = self.get_save_path()

		if os.path.isfile(dst):
			ans = input('File already exists. Overwrite? [y/n]')
			if ans == 'n':
				print('Data not saved.')
				return

		copyfile(src, dst)
		print('Saved to', dst)

	def get_save_path(self):
		rule = self.rule
		PR, RA, pr0, mu, dt = self.PR, self.RA, self.pr0, self.mu, self.dt

		name = '_'.join(map(str, [rule, 'PR', PR, 'RA', RA, 'pr0', pr0, 'mu', mu, 'dt', dt]))
		if rule != 'no_update':
			da, db = self.da, self.db
			theta, rho = self.thetalist[0], self.rholist[0]
			name += '_' + '_'.join(map(str, ['da', da, 'db', db, 'th', theta, 'rh', rho]))
		if rule[4] == '1':
			Pc = self.Pc
			name += '_Pc_' + str(Pc)
		if rule == 'rule1_c2':
			M = self.M
			name += '_M_' + str(M)
		name = name.replace('.', '_')

		# set folder
		if rule == 'no_update':
		    folder = test_data_folder
		else:
		    folder = result_data_folder
		dst = folder + name + '.h5'

		return dst