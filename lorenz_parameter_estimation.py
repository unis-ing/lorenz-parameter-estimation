from rules import *
from helpers import *
from scipy.integrate import odeint
import numpy as np
from copy import deepcopy
import time
import json
import matplotlib.pyplot as plt

class LorenzParams:
    """
    store the current equation and algorithm parameters.
    """

    def __init__(self, PR, RA, B, mu, pr, dt, rule, nudge, **kwargs):
        assert nudge in ['u', 'v', 'w'], 'nudge must be "u", "v", or "w".'
        self.PR = PR
        self.RA = RA
        self.B = B
        self.mu = mu
        self.pr0 = pr  # initial
        self.pr = pr  # current
        self.dt = dt
        self.rule = rule
        self.nudge = nudge

        # initialize
        self.prs = [pr]

        if 'a' in kwargs:
            self.a0 = kwargs['a']  # initial
            self.a = kwargs['a']
            self.a_list = []
        if 'b'in kwargs:
            self.b0 = kwargs['b']  # initial
            self.b = kwargs['b']
            self.b_list = []
        if 'da' in kwargs:
            self.da = kwargs['da']
        if 'db' in kwargs:
            self.db = kwargs['db']
        if 'Tc' in kwargs:
            self.Tc = kwargs['Tc']
            self.T = 0

    def reset_T(self):
        self.T = 0

    def update_T(self, t_curr, t_old):
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
    store the current state of the system.
    """

    def __init__(self, XU, XUt):
        self.x, self.y, self.z, self.u, self.v, self.w = XU
        self.xt, self.yt, self.zt, self.ut, self.vt, self.wt = XUt

        self.z_list = []
        self.w_list = []

        self.t = 0 # previous simulation time

    def update_t(self, t):
        self.t = t

# ------------------------------------------------------------------
#                   function called by odeint
# ------------------------------------------------------------------

def nudged_lorenz(XU, t, s, p, derivs, get_pr, rule_f):

    # update p based on states from last iteration
    next_pr = get_pr(s, p, rule_f, t) # need to pass time in order to update T
    p.pr = next_pr
    p.prs.append(next_pr)

    # update s with positions
    s.x, s.y, s.z, s.u, s.v, s.w = XU

    # update z_list, w_list
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
    s.update_t(t)

    return [xt, yt, zt, ut, vt, wt]

# ------------------------------------------------------------------
#							run odeint
# ------------------------------------------------------------------

def get_deriv_fs(nudge):
    """
    returns list of deriv functions based on nudge.
    """
    UT = ut
    VT = vt
    WT = wt

    if nudge == 'u':
        UT = ut_nudge
    elif nudge == 'v':
        VT = vt_nudge
    elif nudge == 'w':
        WT = wt_nudge

    return (XT, YT, ZT, UT, VT, WT)


def simulate(p, sim_time, deriv_fs=None, complete_msg=True):
    # set get_pr
    get_pr, rule_f = map_rule_to_f(p.rule)

    # set derivs
    if deriv_fs == None:
        deriv_fs = get_deriv_fs(p.nudge)

    # initialize system
    X0, Xt0 = get_ic(p)
    XU0 = np.array(X0 + [0.1, 0.1, 0.1])
    XUt0 = np.array(Xt0 + [0, 0, 0])
    s = LorenzState(XU0, XUt0)

    t = np.arange(0, sim_time, step=p.dt)

    start = time.time()
    sol, infodict = odeint(nudged_lorenz, XU0, t, args=(
        s, p, deriv_fs, get_pr, rule_f), full_output=True)

    if complete_msg:
        print('Runtime: {:.4f} s'.format(time.time() - start))

    # transform list of guesses to be same shape as output
    nfe = infodict['nfe']
    prs = np.array(p.prs)[nfe]
    prs = np.insert(prs, p.pr0, 0)

    # calculate derivatives
    s_ = LorenzState(sol.T, np.zeros(6))
    derivs = np.array([deriv_fs[i](s_, p) for i in range(6)]).T

    return sol, derivs, prs, infodict


# ------------------------------------------------------------------
#                           save data
# ------------------------------------------------------------------


def get_sim_folder(p):
    labeled_params = [['PR', p.PR],
                      ['RA', p.RA],
                      ['pr0', p.pr0],
                      ['mu', p.mu],
                      ['dt', p.dt]]
    pd = p.__dict__

    if 'a' in pd:
        labeled_params.append(['a', p.a])
    if 'b' in pd:
        labeled_params.append(['b', p.b])
    if 'da' in pd:
        labeled_params.append(['da', p.da])
    if 'db' in pd:
        labeled_params.append(['db', p.db])
    if 'Tc' in pd:
        labeled_params.append(['Tc', p.Tc])

    param_str = '_'.join('{}_{:f}'.format(*i).rstrip('0').rstrip('.')
                         for i in labeled_params)
    param_str = param_str.replace('.', '_')
    path = p.rule + '_' + 'nudge' + '_' + p.nudge + '_' + param_str + '/'

    return path


def save_sim(p, sol, derivs, prs, parent_folder='/'):
    """
    parent_folder (ex. sim/sample/)
    """
    assert parent_folder[-1] == '/', 'parent_folder must end in "/".'
    if parent_folder == '/':
        parent_folder = SAVE_FOLDER

    # make folder in sim
    folder = parent_folder + get_sim_folder(p)
    mkdir(folder)

    # store data as .h5
    data_path = folder + 'data.h5'
    f = h5.File(data_path, 'w')
    f.create_dataset('pr', data=prs)

    f.create_dataset('x', data=sol[:, 0])
    f.create_dataset('y', data=sol[:, 1])
    f.create_dataset('z', data=sol[:, 2])
    f.create_dataset('u', data=sol[:, 3])
    f.create_dataset('v', data=sol[:, 4])
    f.create_dataset('w', data=sol[:, 5])

    f.create_dataset('xt', data=derivs[:, 0])
    f.create_dataset('yt', data=derivs[:, 1])
    f.create_dataset('zt', data=derivs[:, 2])
    f.create_dataset('ut', data=derivs[:, 3])
    f.create_dataset('vt', data=derivs[:, 4])
    f.create_dataset('wt', data=derivs[:, 5])

    if 'a' in p.__dict__:
        f.create_dataset('a', data=p.a_list)
    if 'b' in p.__dict__:
        f.create_dataset('b', data=p.b_list)
    f.close()

    # make dictionary with non-list attributes
    param_dict = p.__dict__.copy()
    param_dict.pop('prs')
    param_dict.pop('pr')

    if 'a' in param_dict:
        param_dict.pop('a')
        param_dict.pop('da')
        param_dict.pop('a_list')
        param_dict.pop('b')
        param_dict.pop('db')
        param_dict.pop('b_list')

    if 'T' in param_dict:
        param_dict.pop('T')

    # store params as .json
    param_path = folder + 'params.json'
    with open(param_path, 'w') as outfile:
        json.dump(param_dict, outfile)

# ------------------------------------------------------------------
#                    determine thresholds a, b
# ------------------------------------------------------------------

def pick_best_threshold(p, tholds, errs):
    """
    Pick a,b from results of test_thresholds.
    """
    if min(errs) >= abs(p.PR - p.pr0):
        print('None of the thresholds tested result in error below the initial error.')
        return False, False
        
    minerr_tholds = tholds[errs == min(errs)]
    a, b = minerr_tholds[np.sum(minerr_tholds, 1).argmin()]
    return a, b


def test_thresholds(p, num=500, deriv_fs=None, plot=True):
    """
    This function generates a range of values for the a,b parameters and
    randomly tests 500 (default) of them.
    """
    # sim time for a_, b_
    if hasattr(p, 'Tc'):
        sim_time = 1.1 * p.Tc
    else:
        sim_time = 5

    # run first sim to determine range of a, b to test
    p_ = deepcopy(p)
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

    # randomly choose num (default=500) points
    C = np.random.choice(range(int(grid.shape[0])), num)

    # sim time for testing
    if hasattr(p, 'Tc'):
        sim_time = 3.1 * p.Tc
    else:
        sim_time = 15

    errs = []
    start = time.time()
    for i in range(C.size):
        ind = C[i]
        p_ = deepcopy(p)
        p_.a = grid[ind,0]
        p_.b = grid[ind,1]

        _, _, prs, _ = simulate(p_, sim_time=sim_time, complete_msg=False)
        errs.append(abs(prs[-1] - p.PR))

        if (i + 1) % (num/10) == 0:
            print('{0:} % complete. {1:.4f} sec elapsed.'.format(100*(i+1)//num, time.time()-start))

    tholds = grid[C]
    errs = np.array(errs)

    if plot:
        plt.figure(figsize=(4.5,4))
        sc = plt.scatter(tholds[:,0], tholds[:,1], c=errs, cmap='cool')
        plt.colorbar(sc)
        plt.xlabel('a')
        plt.ylabel('b')
        plt.title('Error for various (a,b)')
        plt.show()

    return tholds, errs