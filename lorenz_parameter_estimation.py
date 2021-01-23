from copy import deepcopy
import json
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import time

from derivs import *
from helpers import *
from rules import *

THOLDPLOTS_FOLDER = 'threshold_plots/'

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
        self.prs = [pr0]

        # condition number
        if rule != 'no_update':
            cn = int(rule[rule.find('_c')+2:])

            if cn == 0:
                pass

            if cn == 1 or cn == 3:
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

            if cn == 1 or cn == 2 or cn == 3:
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
    """
    main simulation function.

    Parameters
    ----------------
    p : LorenzParams instance
    sim_time : units of time to run simulation
    deriv_fs : functions defining the time derivatives of the Lorenz + nudged system.
    complete_msg (boolean) : set to True to print message with error and runtime upon
                            completion.
    """
    # initialize system
    X0, Xt0 = get_ic(p)
    XU0 = np.append(X0, [0.1, 0.1, 0.1]) # initialize nudged slightly away from origin
    XUt0 = np.append(Xt0, [0, 0, 0])
    s = LorenzState(XU0, XUt0)

    # list of times to solve the equation
    t = np.arange(0, sim_time, step=p.dt)

    # set get_pr
    get_pr, rule_f = map_rule_to_f(p.rule)

    # set derivs
    if deriv_fs == None:
        deriv_fs = get_deriv_fs(p.nudge)

    # run sim
    start = time.time()
    sol, infodict = odeint(nudged_lorenz, XU0, t, 
                            args=(s, p, deriv_fs, get_pr, rule_f), 
                            full_output=True, mxstep=100)

    # print completed message
    if complete_msg:
        final_err = abs(p.prs[-1] - p.PR)
        print('Final Prandtl error: {:.8f}. Runtime: {:.4f} s'.format(final_err, 
            time.time() - start))

    # reshape list of guesses to match output
    nfe = infodict['nfe']
    try:
        prs = np.array(p.prs)[nfe]
    except IndexError:
        print('Indexing error in simulation. Returning empty arrays.')
        return np.empty(0), np.empty(0), np.empty(0), infodict

    # this is to ensure prs has the same shape as the solution
    prs = np.append(prs, p.prs[-1])

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
        print('None of the thresholds tested result in error below the initial error.',
            'Returning original thresholds.')
        return p.a0, p.b0

    minerr_tholds = tholds[errs == min(errs)]
    a, b = minerr_tholds[np.sum(minerr_tholds, 1).argmin()]
    return a, b


def test_thresholds(p, num_test=500, num_updates=3.1, deriv_fs=None, 
                    make_plot=True, save_plot=False):
    """
    This function generates a range of values for the a,b parameters and
    randomly tests 500 (default) of them.

    Parameters
    ----------------
    p : instance of LorenzParams; must have the attribute 'Tc'.
    num_test : number of values of a0, b0 to test
    num_updates : simulation time for each a0, b0 calculated by Tc * num_updates.
    deriv_fs : functions defining the time derivatives of the Lorenz + nudged system.
    make_plot (boolean) : set to True to plot a0, b0 with parameter errors.
    save_plot (boolean) : set to True to save plot to THOLDPLOTS_FOLDER.
    """
    assert hasattr(p, 'Tc'), 'LorenzParams must have the attribute "Tc".'

    # run sim to determine range of a, b to test
    sim_time = 1.1 * p.Tc
    p_ = deepcopy(p)
    p_.Tc = np.inf # prevent guess from updating
    sol, derivs, _, _ = simulate(p_, sim_time=sim_time, complete_msg=False)

    # calculate position and velocity errors
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

    # generate grid
    a_min = min(pos_err)
    a_max = np.mean(pos_err) * 3
    b_min = min(vel_err)
    b_max = np.mean(vel_err) * 3

    grid = np.mgrid[a_min:a_max:0.01,b_min:b_max:0.01].T.reshape(-1,2)

    # randomly choose (a,b) to test
    num_pairs = grid.shape[0]
    C = np.random.choice(range(num_pairs), num_test)
    tholds = grid[C]

    # sim time for testing
    if hasattr(p, 'Tc'):
        sim_time = 3.1 * p.Tc
    else:
        sim_time = 15

    pr_errs = []
    delete_ind = []
    start = time.time()
    for i in range(num_test):
        ind = C[i]
        p_ = deepcopy(p)
        p_.a = grid[ind, 0]
        p_.b = grid[ind, 1]

        _, _, prs, _ = simulate(p_, sim_time=sim_time, complete_msg=False)

        if prs.size > 0:
            pr_errs.append(abs(prs[-1] - p.PR))
        else: # skip sims where excessive calls were made
            delete_ind.append(i) # delete from tholds

        if (i + 1) % (num_test / 10) == 0:
            print('{0:} % complete. {1:.4f} sec elapsed.'.format(100*(i+1)//num_test, 
                time.time()-start))

    tholds = np.delete(tholds, delete_ind, axis=0)
    pr_errs = np.array(pr_errs)

    C = np.log10(abs(p.PR - p.pr0))
    if make_plot:
        fig, ax = make_thold_plot(p, tholds, pr_errs)

        if save_plot:
            save_path = get_thold_plot_path(p)
            fig.savefig(save_path, dpi=300)

    return tholds, pr_errs, (fig, ax)


def make_thold_plot(p, tholds, pr_errs):
    """
    helper for test_thresholds; makes the a, b plot.
    """
    C = np.log10(abs(p.PR - p.pr0))

    # make colormap
    orig = get_cmap('bwr', 6)
    errlog = np.log10(pr_errs)

    if np.all(errlog >= C): # only red dots occur
        cmap = truncate_colormap(orig, 0.5, 0.99)

    elif np.any(errlog > C): # red and blue dots occur
        mp = (C - min(errlog)) / (max(errlog) - min(errlog))
        if mp < 0: 
            cmap = orig
        else: 
            cmap = shiftedColorMap(orig, midpoint=mp, name='shifted')
            
    else: # only blue dots occur
        cmap = truncate_colormap(orig, 0, 0.49)

    fig, ax = plt.subplots(1, figsize=(3.5,3))

    # plot points w no update
    ind = np.where(errlog == C)
    plt.scatter(tholds[ind, 0], tholds[ind, 1], color='white', edgecolor='k', s=30)

    # plot everything else
    ind = np.where(errlog != C)

    sc = plt.scatter(tholds[ind, 0], tholds[ind, 1], s=30, c=errlog[ind], 
        edgecolor='k', cmap=cmap)

    plt.colorbar(sc)
    plt.title(r'$n=$'+str(num_test))
    plt.xlabel(r'$\alpha_0$')
    plt.ylabel(r'$\beta_0$')
    ax.set_facecolor('lightgray')
    plt.tight_layout()

    return fig, ax


def get_thold_plot_path(p):
    """
    helper for test_thresholds; get path for saving figure.
    """
    s = get_sim_folder(p)
    s = s[:s.find('a_')] + s[s.find('Tc'):-1]
    return THOLDPLOTS_FOLDER + s