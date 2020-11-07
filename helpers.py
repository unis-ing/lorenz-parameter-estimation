"""
helper functions for the LPE class.

"""
import dedalus.public as de
import h5py as h5
import numpy as np
import os, os.path
from shutil import copyfile, rmtree


def get(solver, s):
	"""
		solver : solver object
		s : (str) name of a solver state
	"""
	return val(solver.state[s])

def val(state):
	return state['g'][0]

def log10mean(arr):
	return np.log10(np.mean(arr))

def running_mean(x, N): # https://stackoverflow.com/a/27681394
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
