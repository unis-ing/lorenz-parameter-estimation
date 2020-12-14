from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import time

from derivs import *
from helpers import *
from rules import *


class LorenzParams:
    """
    Store the current equation and algorithm parameters.
    """

    def __init__(self, PR, RA, B, mu, pr0, dt, rule, nudge, **kwargs):
        assert nudge in ['u', 'v', 'w'], 'Nudge must be "u", "v", or "w".'
        self.PR = PR
        self.RA = RA
        self.B = B
        self.mu = mu
        self.pr0 = pr0  # initial
        self.pr = pr0  # current
        self.dt = dt
        self.rule = rule
        self.nudge = nudge

        # list for storing guesses
        self.prs = [pr]

        # condition number
        cn = int(rule[rule.find('_c')+2:])

        if cn == 0:
            pass

        if cn == 1:
            if 'a0' in kwargs:
                self.a0 = kwargs['a0']  # initial
                self.a = kwargs['a0'] # current
                self.a_list = []
            if 'b0'in kwargs:
                self.b0 = kwargs['b0']  # initial
                self.b = kwargs['b0'] # current
                self.b_list = []
            if 'da' in kwargs:
                self.da = kwargs['da']
            if 'db' in kwargs:
                self.db = kwargs['db']

        if cn == 1 or cn == 2:
            if 'Tc' in kwargs:
                self.Tc = kwargs['Tc']
                self.T = 0

    def reset_T(self):
        self.T = 0

    def update_T(self, t_curr, t_old):
        # increment T by the difference between current time and prev time
        self.T += t_curr - t_old

    def decrease_a(self):
        self.a *= self.da

    def decrease_b(self):
        self.b *= self.db

    def update_a_list(self):
        self.a_list.append(self.a)

    def update_b_list(self):
        self.b_list.append(self.b)


class LorenzState:
    """
    Store the current state of the system.
    """

    def __init__(self, XU, XUt):
        self.x, self.y, self.z, self.u, self.v, self.w = XU
        self.xt, self.yt, self.zt, self.ut, self.vt, self.wt = XUt

        self.z_list = []
        self.w_list = []

        self.t = 0 # previous simulation time

# ------------------------------------------------------------------
#                   function called by odeint
# ------------------------------------------------------------------

def nudged_lorenz(XU, t, s, p, derivs, get_pr, rule_f):
    """
    Function called by odeint.

    Parameters
    ----------------
    derivs : list of functions which implement the time derivatives.
    get_pr : function which returns the next guess.
    rule_f : function which implements the Prandtl estimate (called by get_pr).

    """

    # update p based on XU (values from the last function call)
    next_pr = get_pr(s, p, rule_f, t) # need to pass t in order to update T
    p.pr = next_pr
    p.prs.append(next_pr)

    # update s with positions
    s.x, s.y, s.z, s.u, s.v, s.w = XU

    # update z_list, w_list (only used for rule2z/2w but it doesn't hurt)
    s.z_list.append(s.z)
    s.w_list.append(s.w)

    # calculate new time deriv's with new pr
    xt = derivs[0](s, p)
    yt = derivs[1](s, p)
    zt = derivs[2](s, p)
    ut = derivs[3](s, p)
    vt = derivs[4](s, p)
    wt = derivs[5](s, p)

    # update s with new time derivatives
    s.xt = xt
    s.yt = yt
    s.zt = zt
    s.ut = ut
    s.vt = vt
    s.wt = wt

    # store current time
    s.t = t

    return [xt, yt, zt, ut, vt, wt]

# ------------------------------------------------------------------
#							run odeint
# ------------------------------------------------------------------


def simulate(p, sim_time, deriv_fs=None, complete_msg=True):
    # set get_pr
    get_pr, rule_f = map_rule_to_f(p.rule)

    # set derivs
    if deriv_fs == None:
        deriv_fs = get_deriv_fs(p.nudge)

    # initialize system
    X0, Xt0 = get_ic(p)
    XU0 = np.append(X0, [0.1, 0.1, 0.1])
    XUt0 = np.append(Xt0, [0, 0, 0])
    s = LorenzState(XU0, XUt0)

    # list of times to solve the equation
    t = np.arange(0, sim_time, step=p.dt)

    start = time.time()
    sol, infodict = odeint(nudged_lorenz, XU0, t, args=(
        s, p, deriv_fs, get_pr, rule_f), full_output=True,
        mxstep=100)

    # print completed message
    if complete_msg:
        final_err = abs(p.prs[-1] - p.PR)
        print('Final Prandtl error: {:.8f}. Runtime: {:.4f} s'.format(final_err, 
            time.time() - start))

    # transform list of guesses to be same shape as output
    nfe = infodict['nfe']
    try:
        prs = np.array(p.prs)[nfe]
    except IndexError:
        print('Indexing error in simulation. Returning empty arrays.')
        return np.empty(0), np.empty(0), np.empty(0), infodict

    # this is to ensure prs has the same shape as the solution
    prs = np.append(prs, p.pr0)

    # calculate derivatives
    s_ = LorenzState(sol.T, np.zeros(6))
    derivs = np.array([deriv_fs[i](s_, p) for i in range(6)]).T

    return sol, derivs, prs, infodict

# ------------------------------------------------------------------
#                    determine thresholds a, b
# ------------------------------------------------------------------

def pick_best_threshold(p, tholds, errs):
    """
    Pick a,b from results of test_thresholds.
    """
    if min(errs) >= abs(p.PR - p.pr0):
        print('None of the thresholds tested result in error below the initial error. Returning original thresholds.')
        return p.a0, p.b0

    minerr_tholds = tholds[errs == min(errs)]
    a, b = minerr_tholds[np.sum(minerr_tholds, 1).argmin()]
    return a, b


def test_thresholds(p, num=500, deriv_fs=None, plot=True):
    """
    This function generates a range of values for the a,b parameters and
    randomly tests 500 (default) of them.

    Does not alter p.
    """
    # sim time for a_, b_
    if hasattr(p, 'Tc'):
        sim_time = 1.1 * p.Tc
    else:
        sim_time = 5

    # run first sim to determine range of a, b to test
    p_ = deepcopy(p)
    p_.Tc = np.inf # lazy way to prevent guess from updating
    sol, derivs, _, _ = simulate(p_, sim_time=sim_time, complete_msg=False)

    if p.nudge == 'u':
        i1 = 0
        i2 = 3
    elif p.nudge == 'v':
        i1 = 1
        i2 = 4
    elif p.nudge == 'w':
        i1 = 2
        i2 = 5

    pos_err = abs(sol[:, i1] - sol[:, i2])
    vel_err = abs(derivs[:, i1] - derivs[:, i2])
    a_ = np.min(pos_err)
    b_ = np.min(vel_err)

    # generate grid from [-3, 3]^2 with mesh size 0.01
    a_min = max(0, a_-3)
    a_max = a_ + 3
    b_min = max(0, b_-3)
    b_max = b_ + 3
    grid = np.mgrid[a_min:a_max:0.01,b_min:b_max:0.01].T.reshape(-1,2)

    # randomly choose (a,b) to test
    num_pairs = grid.shape[0]
    C = np.random.choice(range(num_pairs), num)
    tholds = grid[C]

    # sim time for testing
    if hasattr(p, 'Tc'):
        sim_time = 3.1 * p.Tc
    else:
        sim_time = 15

    errs = []
    delete_ind = []
    start = time.time()
    for i in range(C.size):
        ind = C[i]
        p_ = deepcopy(p)
        p_.a = grid[ind, 0]
        p_.b = grid[ind, 1]

        _, _, prs, _ = simulate(p_, sim_time=sim_time, complete_msg=False)

        if prs.size > 0: # skip sims where excessive calls were made
            errs.append(abs(prs[-1] - p.PR))
        else:
            # delete from tholds
            delete_ind.append(i)

        if (i + 1) % (num / 10) == 0:
            print('{0:} % complete. {1:.4f} sec elapsed.'.format(100*(i+1)//num, 
                time.time()-start))

    tholds = np.delete(tholds, delete_ind, axis=0)
    errs = np.array(errs)

    if plot:
        # ---------------------make colormap----------------------------------
        orig = matplotlib.cm.get_cmap('bwr', 6)
        errlog = np.log10(errs)

        if np.all(errlog >= 1): # only red dots occur
            cmap = truncate_colormap(orig, 0.5, 0.99)

        elif np.any(errlog > 1): # red and blue dots occur
            mp = (1 - min(errlog)) / (max(errlog) - min(errlog))
            if mp < 0: 
                cmap = orig
            else: 
                cmap = shiftedColorMap(orig, midpoint=mp, name='shifted')
                
        else: # no red dots occur
            cmap = truncate_colormap(orig, 0, 0.49)

        plt.figure(figsize=(4.5,3.5))

        # plot points w no update
        ind = np.where(errlog == 1)
        plt.scatter(tholds[ind, 0], tholds[ind, 1], color='white', edgecolor='k', s=50)

        # plot everything else
        ind = np.where(errlog != 1)

        sc = plt.scatter(tholds[ind, 0], tholds[ind, 1], s=50, c=errlog[ind], 
            edgecolor='k', cmap=cmap)

        plt.colorbar(sc)
        plt.xlabel('a')
        plt.ylabel('b')
        plt.title('Error for various (a,b)')
        plt.gca().set_facecolor('lightgray')
        plt.show()

    return tholds, errs