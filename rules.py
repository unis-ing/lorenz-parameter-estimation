<<<<<<< HEAD
import numpy as np

# ------------------------------------------------------------------
#							rules
# ------------------------------------------------------------------


def map_rule_to_f(rule):
    """
    returns get_pr function and rule_f corresponding to rule.

    rule should be a string formated as rule[#]c[#].
    """

    if rule == 'no_update':
        get_pr = apply_nothing
        rule_f = no_update

    else:
        i = rule.find('_c')
        cn = int(rule[i+2:]) # condition num
        rule_name = rule[:i]
        rule_f = globals()[rule_name]

        if cn == 0:
            get_pr = apply_nothing
        if cn == 1:
            get_pr = apply_thresholds_and_Tc
        elif cn == 2:
            get_pr = apply_Tc

    return get_pr, rule_f


def no_update(s, p):
    return p.pr


def rule1(s, p):
    return (p.pr * (s.u - s.v) + p.mu * (s.u - s.x)) / (s.x - s.v)


def rule2(s, p):
    its = int(p.Tc / p.dt)
    return -1 - np.mean(s.z_list[-its:])


def rule2a(s, p):
    its = int(p.Tc / p.dt)
    return -1 - np.mean(s.w_list[-its:])


def apply_nothing(s, p, rule_f):
    """ cn = 0 """
    return rule_f(s, p)


def apply_Tc(s, p, rule_f):
    """ cn = 2 """
    c = p.T >= p.Tc

    if c:
        p.reset_T()
        return rule_f(s, p)
    else:
        p.increase_T()
        return p.pr


def apply_thresholds_and_Tc(s, p, rule_f):
    """ cn = 1 """
    if p.nudge == 'u':
        poserr = abs(s.x - s.u)
        velerr = abs(s.xt - s.ut)
    elif p.nudge == 'v':
        poserr = abs(s.y - s.v)
        velerr = abs(s.yt - s.vt)
    elif p.nudge == 'w':
        poserr = abs(s.z - s.w)
        velerr = abs(s.zt - s.wt)

    c1 = p.T >= p.Tc
    c2 = poserr <= p.a
    c3 = velerr <= p.b
    c4 = poserr > 0 and velerr > 0
    c = c1 & c2 & c3 & c4

    if c:
        p.decrease_a()
        p.decrease_b()
        p.update_a_list()
        p.update_b_list()
        p.reset_T()

        return rule_f(s, p)

    else:
        p.increase_T()
        return p.pr
=======
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
>>>>>>> origin/v2
