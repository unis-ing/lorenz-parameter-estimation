"""
functions implementing different update rules for the LPE class and misc. helpers.

rule functions are named as rule[rule #]_c[condition #]

"""
import numpy as np

def rule0(pr, mu, x, y, u, v):
	"""
		Assumes knowledge of x, y.
	"""
	return (pr*u - pr*v + mu*(u-x)) / (x-y)

def rule0_c1(*args):
	lpe = args[0]
	prs = args[1]
	solver = lpe.solver

 	# add one bc this function is called to calculate the next pr
	it = solver.iteration + 1
	if it in prs:
		return prs[it]

	x, u, xt, ut = get(solver, 'x'), get(solver, 'u'), get(solver, 'xt'), get(solver, 'ut')
	uerr, uterr = abs(x-u), abs(xt-ut)
	pr = prs[it-1]
	newpr = pr # initialize next pr as current pr

	c1 = uerr <= lpe.theta
	c2 = uterr <= lpe.rho
	c3 = (uerr > 0) and (uterr > 0)

	if c1 and c2 and c3:
		# decrease threshold
		lpe.theta *= lpe.d
		lpe.rho *= lpe.d
		mu = lpe.mu
		y = get(solver, 'y')
		v = get(solver, 'v')
		newpr = rule0(pr, mu, x, y, u, v)

	prs[it] = newpr
	return newpr

def rule1(pr, mu, x, u, v):
	"""
		Assumes knowledge of x.
	"""
	return (pr*u - pr*v + mu*(u-x)) / (x-v)

def rule1_c1(*args):
	lpe = args[0]
	prs = args[1]
	solver = lpe.solver

 	# add one bc this function is called to calculate the next pr
	it = solver.iteration + 1
	if it in prs:
		return prs[it]

	x, u, xt, ut = get(solver, 'x'), get(solver, 'u'), get(solver, 'xt'), get(solver, 'ut')
	uerr, uterr = abs(x-u), abs(xt-ut)
	pr = prs[it-1]
	newpr = pr # initialize next pr as current pr

	c1 = lpe.P >= lpe.Pc
	c2 = uerr <= lpe.theta
	c3 = uterr <= lpe.rho
	c4 = it >= 5000
	c5 = (uerr > 0) and (uterr > 0)

	if c1 and c2 and c3 and c4 and c5:
		# decrease thresholds
		lpe.theta *= lpe.da
		lpe.rho *= lpe.db
		mu = lpe.mu
		v = get(solver, 'v')
		newpr = rule1(pr, mu, x, u, v)
	prs[it] = newpr
	return newpr

def rule1_c2(*args):
	"""
	Modified rule1_c1 which increases theta, rho if position error increases by an order of 10
	"""
	lpe = args[0]
	prs = args[1]
	uerrs = args[2]

	solver = lpe.solver

 	# add one bc this function is called to calculate the next pr
	it = solver.iteration + 1
	if it in prs:
		return prs[it]

	x, u, xt, ut = get(solver, 'x'), get(solver, 'u'), get(solver, 'xt'), get(solver, 'ut')
	uerr, uterr = abs(x-u), abs(xt-ut)
	pr = prs[it-1]
	newpr = pr # initialize next pr as current pr

	c1 = ((it >= 4000) and (lpe.P >= lpe.Pc)) or (it < 4000)
	c2 = uerr <= lpe.theta
	c3 = uterr <= lpe.rho
	c4 = (uerr > 0) and (uterr > 0)

	if c1 and c2 and c3 and c4:
		# decrease thresholds
		lpe.theta *= lpe.da
		lpe.rho *= lpe.db
		mu = lpe.mu
		v = get(solver, 'v')
		newpr = rule1(pr, mu, x, u, v)

	# increase thresholds when uerr is high
	if it <= lpe.Pc + 1:
		uerrs[it-1] = uerr
	else:
		uerrs[:] = np.roll(uerrs, -1) # update
		uerrs[-1] = uerr

		m1 = np.mean(uerrs[:lpe.Pc])
		m1 = np.log10(m1)
		m2 = np.mean(uerrs[1:])
		m2 = np.log10(m2)
		if m2 - m1 >= 0.0002:
			lpe.theta /= lpe.da
			lpe.rho /= lpe.db

	prs[it] = newpr
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