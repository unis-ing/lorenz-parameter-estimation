"""
functions implementing different update rules for the LPE class.

rule functions are named as rule[rule #]_c[condition #]

"""
from helpers import *

# ------------------------------ rule0 ------------------------------------------
def rule0(pr, mu, x, y, u, v):
	"""
		Assumes knowledge of x, y.
	"""
	return (pr*(u-v) + mu*(u-x)) / (x-y)

def rule0_c1(*args):
	lpe = args[0]
	pr_lasttwo = args[1]

	if pr_lasttwo[1] != None: # shortcut
		next_pr = pr_lasttwo[1]
	else:
		prev_pr = pr_lasttwo[0]
		next_pr = prev_pr # initialize next pr as current pr

		# check conditions are met
		theta, rho, da, db = lpe.theta, lpe.rho, lpe.da, lpe.db
		conds_met = rule0_c1_conds_met(lpe, theta, rho, da, db)

		if conds_met:
			# decrease threshold
			lpe.theta *= lpe.da
			lpe.rho *= lpe.db

			# calculate next guess
			x, y = get(lpe.solver, 'x'), get(lpe.solver, 'y')
			u, v = get(lpe.solver, 'u'), get(lpe.solver, 'v')
			next_pr = rule0(prev_pr, lpe.mu, x, y, u, v)

	# store next_pr
	pr_lasttwo[1] = next_pr

	return next_pr

def rule0_c1_conds_met(lpe, theta, rho, da, db):
	solver = lpe.solver

	x, u, xt, ut = get(solver, 'x'), get(solver, 'u'), get(solver, 'xt'), get(solver, 'ut')
	uerr, uterr = abs(x-u), abs(xt-ut)

	c1 = uerr <= theta
	c2 = uterr <= rho
	c3 = (uerr > 0) and (uterr > 0) # to account for x initialized to 0

	return c1 and c2 and c3

# ------------------------------ rule1 ------------------------------------------
def rule1(pr, mu, x, u, v):
	"""
		Assumes knowledge of x.
	"""
	return (pr*(u-v) + mu*(u-x)) / (x-v)

def rule1_c1(*args):
	"""
		function called by GeneralFunction to return next guess.
	"""
	lpe = args[0]
	pr_lasttwo = args[1]
 	
	if pr_lasttwo[1] != None: # shortcut
		next_pr = pr_lasttwo[1]
	else:
		prev_pr = pr_lasttwo[0]
		next_pr = prev_pr # initialize next pr as current pr

		# check conditions are met
		theta, rho, da, db, Tc = lpe.theta, lpe.rho, lpe.da, lpe.db, lpe.Tc
		conds_met = rule1_c1_conds_met(lpe, theta, rho, da, db, Tc)

		if conds_met:
			# decrease thresholds
			lpe.theta *= lpe.da
			lpe.rho *= lpe.db

			# calculate next guess
			x, u, v = get(lpe.solver, 'x'), get(lpe.solver, 'u'), get(lpe.solver, 'v')
			next_pr = rule1(prev_pr, lpe.mu, x, u, v)
			pr_lasttwo[1] = next_pr

	# store next_pr
	pr_lasttwo[1] = next_pr

	return next_pr

def rule1_c1_conds_met(lpe, theta, rho, da, db, Tc):
	solver = lpe.solver
	t = solver.sim_time

	x, u, xt, ut = get(solver, 'x'), get(solver, 'u'), get(solver, 'xt'), get(solver, 'ut')
	uerr, uterr = abs(x-u), abs(xt-ut)

	c1 = lpe.T >= Tc
	c2 = uerr <= theta
	c3 = uterr <= rho
	c4 = (uerr > 0) and (uterr > 0) # to account for x initialized to 0

	return c1 and c2 and c3 and c4

#################################################################################