from dedalus.core.operators import GeneralFunction
from helpers import *
from rules import *
from inspect import getfullargspec
import warnings 
import time
import json

# suppress warning from a deprecated h5py function being used in dedalus source code
de.logging_setup.rootlogger.setLevel('ERROR')
warnings.filterwarnings("ignore", category=h5.h5py_warnings.H5pyDeprecationWarning)

# global paths
RESULT_FOLDER = 'result_data/'
NOUPDATE_FOLDER = 'test_data/'
INITIALDATA_FOLDER = 'initial_data/'
THRESHOLD_FOLDER = 'thresholds/'
ANALYSIS_NAME = 'analysis'
TEMP_FILE_PATH = 'analysis/analysis_s1/analysis_s1_p0.h5'

# global simulation variables
B = 8/3
NS = 2
scheme = de.timesteppers.RK443

class LPE:
	def __init__(self, PR=10, RA=28, pr0=20, mu=100, dt=0.001, rule='no_update',
			  nudgecoords='x', **kwargs):
		"""
		Store equation parameters, simulation parameters, algorithm parameters, build solver, and
		store path to data folder.

		Parameters:
			PR : Prandtl number
			RA : Rayleigh number
			pr0 : initial guess for Prandtl number
			mu : nudging parameter
			dt : simulation timestep
			rule : (str) name of existing rule function
			nudgecoords : (str) array or single string representing nudged coordinate(s). ex) 'xy'

		"""
		self.PR = PR
		self.RA = RA
		self.pr0 = pr0
		self.mu = mu
		self.dt = dt
		self.rule = rule
		self.nudgecoords = nudgecoords

		# perform preliminary checks
		self.passed = LPE.validate_rule_parameters(rule, **kwargs)
		if not self.passed:
			return

		# store the condition args in kwargs
		self.store_condition_args(**kwargs)

		# initialize lists to track parameters
		self.prlist = []
		if 'theta' in kwargs and 'rho' in kwargs:
			self.thetalist = []
			self.rholist = []
		if 'Tc' in kwargs:
			self.T = 0

		basis = de.Chebyshev('s', 2, interval=(0,1), dealias=3/2)
		domain = de.Domain([basis], grid_dtype=np.float64)
		problem = de.IVP(domain, variables=['x', 'y', 'z', 'xt', 'yt', 'zt',
											'u', 'v', 'w', 'ut', 'vt', 'wt'])
		self.problem = problem

		# add parameters to problem namespace
		problem.parameters['PR'] = PR
		problem.parameters['RA'] = RA
		problem.parameters['B'] = B
		problem.parameters['mu'] = mu

		#---------------------------------------------------------------------------------
		# add pr to problem namespace
		if rule == 'no_update':
			problem.parameters['pr'] = pr0
		else:
			# get function and argument list to be passed to GeneralFunction
			rule_F = self.get_rule_F()
			F_args = self.get_F_args(**kwargs)

			problem.parameters['pr'] = de.operators.GeneralFunction(domain, layout='g', 
																	func=rule_F, args=[])
			self.F_args = F_args
		#---------------------------------------------------------------------------------

		# add lorenz equations
		problem.add_equation('dt(x) = -PR*x + PR*y')
		problem.add_equation('dt(y) + y = -PR*x - x*z') 
		problem.add_equation('dt(z) = -B*z + x*y - B*(RA+PR)')

		# add nudged lorenz equations
		nudged_eqns = ['dt(u) = -pr*u + pr*v', 
					   'dt(v) + v = -pr*u - u*w', 
					   'dt(w) = -B*w + u*v - B*(RA+pr)']
		if 'x' in nudgecoords:
			nudged_eqns[0] += '- mu*(u-x)'
		if 'y' in nudgecoords:
			nudged_eqns[1] += '- mu*(v-y)'
		if 'z' in nudgecoords:
			nudged_eqns[2] += '- mu*(w-z)'

		for eqn in nudged_eqns:
			problem.add_equation(eqn)

		# relate time derivatives
		problem.add_equation('dt(x) - xt = 0')
		problem.add_equation('dt(y) - yt = 0')
		problem.add_equation('dt(z) - zt = 0')
		problem.add_equation('dt(u) - ut = 0')
		problem.add_equation('dt(v) - vt = 0')
		problem.add_equation('dt(w) - wt = 0')

		#---------------------------------------------------------------------------------
		if rule != 'no_update': 
			# for some reason args must be set AFTER equations are added
			problem.parameters['pr'].args = F_args
			problem.parameters['pr'].original_args = F_args
		#---------------------------------------------------------------------------------

		# build solver
		solver = problem.build_solver(scheme)
		self.solver = solver

		# make data folder and store path
		self.datafolder = self.get_datafolder_path(**kwargs)

		# store all parameters in a dict to be exported to JSON when simulation is run
		params = {k:v for k,v in problem.parameters.items() if k != 'pr'}
		params['pr0'] = pr0
		params['dt'] = dt
		params['rule'] = rule
		for p in kwargs:
			params[p] = kwargs[p]
		self.params = params


	def simulate(self, stop_it=10000, stop_time=np.inf, print_every=10):
		""""
		simulate the Lorenz equations and the nudged Lorenz equations.

		Parameters:
			stop_it: maximum iterations allowed
			stop_time: maximum simulation time allowed
		"""
		if not self.passed:
			print('Parameter validation failed. Cannot proceed with simulation.',
				  'Please check and re-instantiate.')
			return

		rule = self.rule
		solver = self.solver
		dt = self.dt
		solver.stop_sim_time = stop_time
		solver.stop_iteration = stop_it

		# calculate print_every_it for percent bar
		if stop_it != np.inf:
			max_it = stop_it
		else:
			max_it = min(stop_it, stop_time/dt)
		print_every_it = np.floor(max_it / print_every)

		# initialize lorenz system
		PR, RA = self.PR, self.RA
		ic_path = self.get_ic_path()

		if not os.path.exists(ic_path):
			print('Making initial data.')
			self.make_initial_data(PR=PR, RA=RA)

		# set initial conditions
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
		analysis = solver.evaluator.add_file_handler(filename=ANALYSIS_NAME, iter=1, 
													 max_size=2**(64))
		analysis.add_system(solver.state, layout='g')

		start = time.time()
		while solver.ok:
			solver.step(dt)

			if solver.iteration % print_every_it == 0:
				percent_complete = (solver.iteration / max_it) * 100
				progress_msg = '{0:.0f}% complete. {1:.2f} sec elapsed.'.format(percent_complete, 
																				time.time()-start)
				if rule != 'no_update':
					curr_pr = self.get_curr_pr()
					pr_err = abs(self.PR - curr_pr)
					progress_msg += ' Abs pr error: {:.8e}.'.format(pr_err)
				print(progress_msg)

			if rule != 'no_update':
				prev_pr = self.get_prev_pr()
				curr_pr = self.get_curr_pr()

				# update parameter lists
				self.prlist.append(curr_pr)
				self.thetalist.append(self.theta)
				self.rholist.append(self.rho)

				# reset pr_lasttwo argument
				self.F_args[1] = [curr_pr, None]

				# update T
				if prev_pr != curr_pr:
					self.T = 0
				else:
					self.T += dt

		# write parameter lists to the temp file
		if rule != 'no_update':
			f = h5.File(TEMP_FILE_PATH, 'a')
			f.create_dataset('tasks/pr', data=self.prlist)
			f.create_dataset('tasks/theta', data=self.thetalist)
			f.create_dataset('tasks/rho', data=self.rholist)
			f.close()

	def save_data(self, parent_folder=None, check_exists=True):
		"""
			copy the data from the temp file to a new folder. save the parameters as a JSON.
		"""
		datafolder = self.datafolder

		if parent_folder != None and parent_folder[-1] != '/':
			print('Parent folder must include ''/'' at the end. Please correct and re-save.')
			return
		if parent_folder != None:
			i = datafolder.find('rule')
			left = datafolder[:i]
			right = datafolder[i:]
			save_path = left + parent_folder + right
		else:
			save_path = datafolder

		if os.path.isdir(save_path):
			if check_exists:
				ans = input('Data folder already exists. Overwrite? [y/n]')
				if ans == 'n':
					print('Data not saved.')
					return

			rmtree(save_path)

		# make directory and copy temp file
		os.mkdir(save_path)
		data_path = save_path + '/' + 'data.h5'
		copyfile(TEMP_FILE_PATH, data_path)

		param_path = save_path + '/' + 'params.json'
		with open(param_path, 'w') as outfile:
			json.dump(self.params, outfile)

		print('Saved to folder', save_path)

	# ----------------------------- methods for accessing pr ----------------------------------

	def get_prev_pr(self):
		"""
			gets the previous guess. Assumes rule != 'no_update'.
		"""
		pr_lasttwo = self.F_args[1]
		return pr_lasttwo[0]

	def get_curr_pr(self):
		"""
			gets the current guess. Assumes rule != 'no_update'.
		"""
		pr_lasttwo = self.F_args[1]
		return pr_lasttwo[1]

	# --------------------------- helpers for LPE.__init__()  ---------------------------------

	@staticmethod
	def validate_rule_parameters(rule, **kwargs):
		"""
			* check that all necessary parameters for a given rule are passed.
			* returns True if all parameters are found, otherwise returns false
			and prints message with missing parameters.
		"""
		if rule == 'no_update':
			return True

		# list of required arguments
		cond_f = globals()[rule + '_conds_met']
		req_list = getfullargspec(cond_f)[0]

		# check which are missing; lpe will always be present
		missing = [req for req in req_list if (req not in kwargs) and (req != 'lpe')]
		extra = [kw for kw in kwargs if kw not in req_list]

		if len(missing) > 0:
			missing_str = 'Missing rule parameter(s): ' + ', '.join(missing)
			print(missing_str)
		if len(extra) > 0:
			extra_str = 'Extra rule parameter(s) supplied: ' + ', '.join(extra)
			print(extra_str)

		return (len(missing) == 0) and (len(extra) == 0)

	def store_condition_args(self, **kwargs):
		"""
			store arguments for rule conditions as class attributes
		"""

		if 'da' in kwargs:
			self.da = kwargs['da']
		if 'db' in kwargs:
			self.db = kwargs['db']
		if 'theta' in kwargs:
			self.theta = kwargs['theta']
		if 'rho' in kwargs:
			self.rho = kwargs['rho']
		if 'Tc' in kwargs:
			self.Tc = kwargs['Tc']
		if 'M' in kwargs:
			self.M = kwargs['M']
		# add future condition args here

	def get_rule_F(self):
		"""
			* return function to be called by GeneralFunction
			* assumes rule != 'no_update'
		"""
		return globals()[self.rule]

	def get_F_args(self, **kwargs):
		"""
			* returns a list of arguments to be passed to GeneralFunction.
			* assumes rule != 'no_upate'
		"""
		pr_lasttwo = [self.pr0, None]
		arg_list = [self, pr_lasttwo]

		return arg_list

	# -------------------- methods for retrieving folder paths ----------------------------

	def get_ic_path(self):
		"""
			get path to file containing initial data.
		"""
		ic_name = '_'.join(map(str, ['PR', self.PR, 'RA', self.RA]))
		return INITIALDATA_FOLDER + ic_name + '.h5'

	def get_datafolder_path(self, **kwargs):
		"""
			make name to save folder by concatenating parameters.
		"""
		rule, PR, RA, pr0, mu, dt = self.rule, self.PR, self.RA, self.pr0, self.mu, self.dt

		params = [['PR', PR],
				  ['RA', RA],
				  ['pr0', pr0],
				  ['mu', mu],
				  ['dt', dt]]

		if 'da' in kwargs:
			params.append(['da', kwargs['da']])
		if 'db' in kwargs:
			params.append(['db', kwargs['db']])
		if 'theta' in kwargs:
			params.append(['theta', kwargs['theta']])
		if 'rho' in kwargs:
			params.append(['rho', kwargs['rho']])
		if 'Tc' in kwargs:
			params.append(['Tc', kwargs['Tc']])
		if 'M' in kwargs:
			params.append(['M', kwargs['M']])

		# format parameters in a single string
		params_str = '_'.join('{}_{:f}'.format(*p).rstrip('0').rstrip('.') for p in params)
		params_str = params_str.replace('.', '_')

		# add folder
		if rule == 'no_update':
		    parent_folder = NOUPDATE_FOLDER
		else:
		    parent_folder = RESULT_FOLDER
		folder_path = parent_folder + rule + '_' + params_str

		return folder_path

	# -------------------- method for making initial data ----------------------------

	def make_initial_data(self, PR, RA):
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
		problem.add_equation('dt(x) = -PR*x + PR*y')
		problem.add_equation('dt(y) + y = -PR*x - x*z') 
		problem.add_equation('dt(z) = -B*z + x*y - B*(RA+PR)')

		solver = problem.build_solver(scheme)

		x, y, z = solver.state['x'], solver.state['y'], solver.state['z']
		x['g'] = np.full(NS, fill_value=10)
		y['g'] = np.full(NS, fill_value=10)
		z['g'] = np.full(NS, fill_value=10)	

		# run for 5 units of time
		solver.stop_sim_time = 5

		while solver.ok:
			solver.step(dt)

		new_ic_path = INITIALDATA_FOLDER + '_'.join(map(str, ['PR', self.PR, 'RA', self.RA])) + '.h5'
		f = h5.File(new_ic_path, 'a')
		f.create_dataset('x', data=np.array([val(x)]))
		f.create_dataset('y', data=np.array([val(y)]))
		f.create_dataset('z', data=np.array([val(z)]))
		f.close()
