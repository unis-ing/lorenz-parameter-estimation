from copy import deepcopy
import json
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
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

            if cn != 2:
                if 'a0' in kwargs:
                    self.a0 = kwargs['a0'] # initial
                    self.a = kwargs['a0'] # current
                if 'b0'in kwargs:
                    self.b0 = kwargs['b0'] # initial
                    self.b = kwargs['b0'] # current
                if 'da' in kwargs:
                    self.da = kwargs['da']
                if 'db' in kwargs:
                    self.db = kwargs['db']

            if cn != 0:
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


class LorenzState:
    """
    Store the current state of the system.
    """

    def __init__(self, XU, XUt):
        self.x, self.y, self.z, self.u, self.v, self.w = XU
        self.xt, self.yt, self.zt, self.ut, self.vt, self.wt = XUt

        self.x_list = []
        self.z_list = []
        self.w_list = []

        self.tfe = [] # list of times at which function is evaluated

# ------------------------------------------------------------------
#                   function called by odeint
# ------------------------------------------------------------------

def nudged_lorenz(t, XU, s, p, derivs, get_pr, rule_f):
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

    # update x_list, z_list, w_list (last two only used for 
    #                               rule2z/2w but it doesn't hurt)
    s.x_list.append(s.x)
    s.z_list.append(s.z)
    s.w_list.append(s.w)

    # store prev time
    s.tfe.append(t)

    # calculate new time deriv's with new pr & update s
    s.xt = xt = derivs[0](s, p)
    s.yt = yt = derivs[1](s, p)
    s.zt = zt = derivs[2](s, p)
    s.ut = ut = derivs[3](s, p)
    s.vt = vt = derivs[4](s, p)
    s.wt = wt = derivs[5](s, p)

    return [xt, yt, zt, ut, vt, wt]

# ------------------------------------------------------------------
#							run odeint
# ------------------------------------------------------------------

def simulate(p, sim_time, deriv_fs=None, complete_msg=True, print_err=True, err_thold=20):
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
    t_span = [0, sim_time]
    t_eval = np.arange(0, sim_time, step=p.dt)

    # set get_pr
    get_pr, rule_f = map_rule_to_f(p.rule)

    # set derivs
    if deriv_fs == None:
        deriv_fs = get_deriv_fs(p.nudge)

    # define event for exiting on high Prandtl error
    def high_error_event(t, *XU):
        if abs(p.PR - p.pr) >= err_thold:
            thold_exceeded = True
            return 0
        else:
            return 1
    high_error_event.terminal = True

    # run sim
    start = time.time()
    sol = solve_ivp(nudged_lorenz, t_span=t_span, y0=XU0, 
                    t_eval=t_eval, method='LSODA',
                    args=(s, p, deriv_fs, get_pr, rule_f),
                    events=high_error_event, max_step=10)

    # termination status
    err_exceeded = sol.status == 1

    # print completed message
    if complete_msg:
        final_err = abs(p.prs[-1] - p.PR)
        print('Final Prandtl error: {:.4e}. Runtime: {:.4f} s'.format(final_err, 
            time.time() - start))

    # reconstruct nfe from odeint
    t_last = (s.tfe[-1] // p.dt) * p.dt
    t_actual = np.arange(0, t_last, step=p.dt)
    nfe = np.searchsorted(s.tfe, t_actual)

    # reshape prs to match shape of t_eval
    try:
        prs = np.array(p.prs)[nfe]
        prs = np.append(prs, prs[-1])
    except IndexError:
        if print_err:
            print('Indexing error in simulation; returning empty arrays. Try',
            're-running with different algorithm parameter values.')

    # calculate derivatives
    y = sol.y
    s_ = LorenzState(y, np.zeros(6))
    derivs = np.array([deriv_fs[i](s_, p) for i in range(6)]).T

    return y.T, derivs, prs, err_exceeded

# ------------------------------------------------------------------
#                    determine thresholds a, b
# ------------------------------------------------------------------

def pick_best_threshold(p, tholds, errs):
    """
    Pick a,b from results of test_thresholds. Return zeros if no values found
    which result in improved parameter error.
    """
    if min(errs) >= abs(p.PR - p.pr0):
        a_interval = '[{:.2f}, {:.2f}]'.format(min(tholds[:,0]), max(tholds[:,0]))
        b_interval = '[{:.2f}, {:.2f}]'.format(min(tholds[:,1]), max(tholds[:,1]))
        print('None of the thresholds (a,b) tested in the region', a_interval, 'x', 
            b_interval, 'result in improved parameter error. Returning zeros.')
        return 0, 0

    minerr_tholds = tholds[errs == min(errs)]
    a, b = minerr_tholds[np.sum(minerr_tholds, 1).argmin()]
    return a, b


def test_thresholds(p, num_test=500, num_updates=3, deriv_fs=None,
                    make_plot=True, save_plot=False):
    """
    This function generates a range of values for the a,b parameters and
    randomly tests 500 (default) of them.

    Parameters
    ----------------
    p : instance of LorenzParams; must have the attribute 'Tc'.
    num_test : number of values of a0, b0 to test
    num_updates : simulation time for each a0, b0 calculated by Tc * (num_updates + 0.1).
    deriv_fs : functions defining the time derivatives of the Lorenz + nudged system.
    make_plot (boolean) : set to True to plot a0, b0 with parameter errors.
    save_plot (boolean) : set to True to save plot to THOLDPLOTS_FOLDER.
    """
    assert hasattr(p, 'Tc'), 'LorenzParams must have the attribute "Tc".'
    print('Testing thresholds.')

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

    # count num elt's in grid
    a_min = round(min(pos_err) * 0.95, 2)
    a_max = round(np.mean(pos_err) * 3, 2)
    b_min = round(min(vel_err) * 0.95, 2)
    b_max = round(np.mean(vel_err) * 3, 2)

    step = 0.01
    num_a = int((a_max - a_min) / step + 1)
    num_b = int((b_max - b_min) / step + 1)
    num_pairs = num_a * num_b

    # generate random indices of elt's in grid
    C = np.random.randint(low=0, high=num_pairs-1, size=num_test)

    # sim time for testing
    sim_time = (num_updates + 0.1) * p.Tc

    pr_errs = []
    tholds = []
    bad_tholds = []

    start = time.time()
    for i in range(num_test):
        c = C[i]
        p_ = deepcopy(p)
        a0 = a_min + step * (c % num_a)
        b0 = b_min + step * (c - c % num_a) / num_a
        p_.a = a0
        p_.b = b0

        _, _, prs, err_exceeded = simulate(p_, sim_time=sim_time, 
                                             complete_msg=False, print_err=False)

        if err_exceeded: # separate tholds where excessive calls were made
            bad_tholds.append([a0, b0])
        else:
            pr_errs.append(abs(prs[-1] - p.PR))
            tholds.append([a0, b0])

        if (i + 1) % (num_test / 10) == 0:
            print('{0:} % complete. {1:.4f} sec elapsed.'.format(100*(i+1)//num_test, 
                time.time()-start))

    pr_errs = np.array(pr_errs)
    tholds = np.array(tholds)
    bad_tholds = np.array(bad_tholds)

    # create plot
    if make_plot:
        fig, ax = make_thold_plot(p, tholds, bad_tholds, pr_errs, num_test)

        # save plot
        if save_plot:
            save_path = get_thold_plot_path(p)
            fig.savefig(save_path, dpi=300)

    return tholds, pr_errs, (fig, ax)


def make_thold_plot(p, tholds, bad_tholds, pr_errs, num_test):
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

    # plot bad tholds
    if bad_tholds.size > 0:
        plt.scatter(bad_tholds[:,0], bad_tholds[:,1], marker='x', color='k', s=30)

    # plot points w no update
    ind = np.where(errlog == C)
    if ind[0].size > 0:
        plt.scatter(tholds[ind, 0], tholds[ind, 1], color='white', edgecolor='k', s=30)

    # plot everything else
    ind = np.where(errlog != C)
    if ind[0].size > 0:
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
    s = s[:s.find('a_')] + s[s.find('da'):-1]
    return THOLDPLOTS_FOLDER + s