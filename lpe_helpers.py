"""
helper functions for the LPE class.

"""

def get(solver, s):
	"""
	Input
		s : string which is the name of one of the solver states
	Output
		value at 0th place of state
	"""
	return solver.state[s]['g'][0]

def val(state):
	"""
	Input
		state : solver.state attribute
	"""
	return state['g'][0]

def log10mean(arr):
	return np.log10(np.mean(arr))