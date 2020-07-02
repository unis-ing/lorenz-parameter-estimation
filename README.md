Code, data, and etc. produced in a research project under Professor Vincent Martinez at Hunter College.

## Prerequisites
* `dedalus` is used to solve the Lorenz equations. An installation guide can be found [here](https://dedalus-project.readthedocs.io/en/latest/installation.html).
* `h5py` is used to read and write data.

## Files
* [`lorenz_pr_estimation.py`](https://github.com/unis-ing/lorenz-parameter-estimation/blob/master/lorenz_pr_estimation.py) :  main `LPE` class.
* [`rules.py`](https://github.com/unis-ing/lorenz-parameter-estimation/blob/master/rules.py) : implements different estimates for the Prandtl number.
* [`initial_data/`](https://github.com/unis-ing/lorenz-parameter-estimation/tree/master/initial_data) : .h5 files specifying initial conditions for the Lorenz equations.
