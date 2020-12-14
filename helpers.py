import h5py as h5
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from os import mkdir
from os.path import isfile, isdir
from shutil import rmtree
from scipy.integrate import odeint

SIM_FOLDER = 'sim/'
IC_FOLDER = 'ic/'

# ------------------------------------------------------------------
#                           save data
# ------------------------------------------------------------------

def get_sim_folder(p):
    """
        Concatenate parameters to create folder name.
    """
    labeled_params = [['PR', p.PR],
                      ['RA', p.RA],
                      ['pr0', p.pr0],
                      ['mu', p.mu],
                      ['dt', p.dt]]
    pd = p.__dict__

    if 'a' in pd:
        labeled_params.append(['a', p.a0])
    if 'b' in pd:
        labeled_params.append(['b', p.b0])
    if 'da' in pd:
        labeled_params.append(['da', p.da])
    if 'db' in pd:
        labeled_params.append(['db', p.db])
    if 'Tc' in pd:
        labeled_params.append(['Tc', p.Tc])

    # format floats by replacing . with _
    param_str = '_'.join('{}_{:f}'.format(*i).rstrip('0').rstrip('.')
                         for i in labeled_params)
    param_str = param_str.replace('.', '_')

    # add rule name and nudge coord. to the front
    path = p.rule + '_' + 'nudge' + '_' + p.nudge + '_' + param_str + '/'

    return path


def save_sim(p, sol, derivs, prs, parent_folder=SIM_FOLDER, check_exists=True):
    """
    Saves sol and derivs to a file under parent_folder (ex. sim/sample/)
    """
    assert parent_folder[-1] == '/', 'parent_folder must end in "/".'

    # get folder name
    folder = parent_folder + get_sim_folder(p)

    # check if folder exists
    if isdir(folder):
        if check_exists:
            ans = input('Data folder already exists. Overwrite? [y/n] ')
            if ans == 'n':
                print('Data not saved.')
                return
            else: # delete folder and its contents
                rmtree(folder)

    # make new folder
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

    # copy dictionary with parameters
    param_dict = p.__dict__.copy()
    param_dict.pop('prs')
    param_dict.pop('pr')

    # remove attributes that won't be stored
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

    print('Saved data to the folder:', folder)

# ------------------------------------------------------------------
#                      make initial conditions
# ------------------------------------------------------------------

def make_ic(p):
    PR = p.PR
    RA = p.RA
    B = p.B

    def lorenz(X, t):
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
    f.create_dataset('x', data=sol[-1, 0])
    f.create_dataset('y', data=sol[-1, 1])
    f.create_dataset('z', data=sol[-1, 2])
    f.create_dataset('xt', data=derivs[-1, 0])
    f.create_dataset('yt', data=derivs[-1, 1])
    f.create_dataset('zt', data=derivs[-1, 2])
    f.close()

def get_ic(p):
    """
        Retrieves initial conditions for PR, RA, B if the file exists. Otherwise
        makes it.
    """
    if not ic_exists(p):
        print('Making initial conditions.')
        sol, derivs = make_ic(p)
        save_ic(p, sol, derivs)

    # open file with ic
    f = h5.File(get_ic_path(p), 'r')
    x0 = np.array(f['x'])[-1]
    y0 = np.array(f['y'])[-1]
    z0 = np.array(f['z'])[-1]
    xt0 = np.array(f['xt'])[-1]
    yt0 = np.array(f['yt'])[-1]
    zt0 = np.array(f['zt'])[-1]

    return np.array([x0, y0, z0]), np.array([xt0, yt0, zt0])


# ------------------------------------------------------------------
#                     visualization helpers
# ------------------------------------------------------------------
# https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap