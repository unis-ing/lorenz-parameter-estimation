import h5py as h5
import numpy as np
from os import mkdir
from os.path import isfile
from scipy.integrate import odeint

SAVE_FOLDER = 'sim/'
IC_FOLDER = 'ic/'

# ------------------------------------------------------------------
#				functions defining time derivatives
# ------------------------------------------------------------------


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