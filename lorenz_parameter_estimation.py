
from scipy.integrate import odeint
import time
import json
import h5py as h5
from os import mkdir
from os.path import isfile
from rules import *

SAVE_FOLDER = 'sim/'
IC_FOLDER = 'ic/'


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

    def increase_T(self):
        self.T += self.dt

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

# ------------------------------------------------------------------
#				functions defining time derivatives
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


def XT(s, p):
    return p.PR * (s.y - s.x)


def YT(s, p):
    return -p.PR * s.x - s.y - s.x * s.z


def ZT(s, p):
    return s.x * s.y - p.B*(s.z + p.PR + p.RA)


def ut(s, p):
    return p.pr * (s.v - s.u) - p.mu * (s.u - s.x)


def vt(s, p):
    return -p.pr * s.u - s.v - s.u * s.w


def wt(s, p):
    return s.u * s.v - p.B * (s.w + p.pr + p.RA)


def ut_nudge(s, p):  # nudge u with x
    return ut(s, p) - p.mu * (s.u - s.x)


def vt_nudge(s, p):  # nudge v with y
    return vt(s, p) - p.mu * (s.v - s.y)


def wt_nudge(s, p):  # nudge w with z
    return wt(s, p) - p.mu * (s.w - s.z)

# ------------------------------------------------------------------
#                           make ic
# ------------------------------------------------------------------

def make_ic(p):

    def lorenz(X, t, PR=p.PR, RA=p.RA, B=p.B):
        x, y, z = X
        return [PR*(y - x), -PR*x - y - x*z, x*y - B*(z + PR + RA)]

    # picked dt and sim_time arbitrarily -- maybe change later
    dt = 0.001
    sim_time = 3
    t = np.arange(0, sim_time, step=dt)
    X0 = np.full(3, fill_value=10)

    sol = odeint(lorenz, X0, t)
    derivs = np.array(lorenz(sol.T, t)).T
    return sol, derivs

def get_ic_path(p):
    return IC_FOLDER + 'PR_{:.0f}_RA_{:.0f}_B_{:.4f}'.format(p.PR, p.RA, p.B) + '.h5'

def ic_exists(p):
    path = get_ic_path(p)
    return isfile(path)

def save_ic(p, sol, derivs):
    path = get_ic_path(p)

    # store data as .h5
    f = h5.File(path, 'w')
    f.create_dataset('x', data=sol[:, 0])
    f.create_dataset('y', data=sol[:, 1])
    f.create_dataset('z', data=sol[:, 2])
    f.create_dataset('xt', data=derivs[:, 0])
    f.create_dataset('yt', data=derivs[:, 1])
    f.create_dataset('zt', data=derivs[:, 2])
    f.close()

def get_ic(p):
    if ic_exists(p):
        f = h5.File(get_ic_path(p), 'r')
        x0 = np.array(f['x'])[-1]
        y0 = np.array(f['y'])[-1]
        z0 = np.array(f['z'])[-1]
        xt0 = np.array(f['xt'])[-1]
        yt0 = np.array(f['yt'])[-1]
        zt0 = np.array(f['zt'])[-1]
    else:
        print('Making initial conditions.')
        sol, derivs = make_ic(p)
        save_ic(p, sol, derivs)
        x0 = sol[:,0][-1]
        y0 = sol[:,0][-1]
        z0 = sol[:,0][-1]
        xt0 = sol[:,0][-1]
        yt0 = sol[:,0][-1]
        zt0 = sol[:,0][-1]

    return [x0, y0, z0], [xt0, yt0, zt0]

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
#                   function called by odeint
# ------------------------------------------------------------------


def nudged_lorenz(XU, t, s, p, derivs, get_pr, rule_f):
    # update p based on states from last iteration
    next_pr = get_pr(s, p, rule_f)
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
    XU0 = np.array(X0 + [0.1, 0.1, 0.1])
    XUt0 = np.array(Xt0 + [0, 0, 0])
    s = LorenzState(XU0, XUt0)

    t = np.arange(0, sim_time, step=p.dt)

    start = time.time()
    sol, infodict = odeint(nudged_lorenz, XU0, t, args=(
        s, p, deriv_fs, get_pr, rule_f), full_output=True)

    if complete_msg:
        print('Runtime: {:.4f} s'.format(time.time() - start))
        print('Effective Tc: {:.4f}'.format(p.Tc * sol[:, 0].size / len(p.prs)))

    prs = np.array(p.prs)

    # calculate derivatives
    s_ = LorenzState(sol.T, np.zeros(6))
    derivs = np.array([deriv_fs[i](s_, p) for i in range(6)]).T

    return sol, derivs, prs, infodict
