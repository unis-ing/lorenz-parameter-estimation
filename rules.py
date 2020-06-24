
def rule0(pr, mu, x, y, u, v):
	"""
		Assumes knowledge of x, y.
	"""
	return (pr*u - pr*v + mu*(u-x)) / (x-y)

def rule0_c1(*args):
	pr = args[0]
	solver = args[1]
	lpe = args[2]

	x = get(solver, 'x')
	xt = get(solver, 'xt')
	u = get(solver, 'u')
	ut = get(solver, 'ut')

	uerr = abs(x-u)
	uterr = abs(xt-ut)

	if (uerr<=lpe.theta) and (uterr<=lpe.rho) and (uerr>0) and (uterr>0):
		# decrease threshold
		lpe.theta *= lpe.d
		lpe.rho *= lpe.d
		mu = lpe.mu
		y = get(solver, 'y')
		v = get(solver, 'v')
		return rule0(pr, mu, x, y, u, v)
	else:
		return pr

def rule1(pr, mu, x, u, v):
	"""
		Assumes knowledge of x.
	"""
	return (pr*u - pr*v + mu*(u-x)) / (x-v)

def rule1_c1(*args):
	pr = args[0]
	solver = args[1]
	lpe = args[2]
	prdict = args[3]

	x = get(solver, 'x')
	xt = get(solver, 'xt')
	u = get(solver, 'u')
	ut = get(solver, 'ut')

	uerr = abs(x-u)
	uterr = abs(xt-ut)

	print(solver.iteration, ' ', uerr, ' ', uterr)
	it = solver.iteration
	if it in prdict:
		return prdict[it]

	newpr = pr
	# if (solver.iteration>=5000) and (lpe.last>=lpe.P) and (uerr<=lpe.theta) and (uterr<=lpe.rho) and (uerr>0) and (uterr>0):
	if (lpe.last>=lpe.P) and (uerr<=lpe.theta) and (uterr<=lpe.rho) and (uerr>0) and (uterr>0):
		# decrease thresholds
		lpe.theta *= lpe.d
		lpe.rho *= lpe.d
		mu = lpe.mu
		v = get(solver, 'v')
		newpr = rule1(pr, mu, x, u, v)

	prdict[solver.iteration] = newpr
	return newpr

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