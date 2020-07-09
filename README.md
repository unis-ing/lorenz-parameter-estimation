Code, data, and etc. produced in an ongoing research project under Professor Vincent Martinez at Hunter College.

## Prerequisites
* `dedalus` is used to solve the Lorenz equations. An installation guide can be found [here](https://dedalus-project.readthedocs.io/en/latest/installation.html).
* `h5py` is used to read and write data.

## Files
* [`lorenz_pr_estimation.py`](https://github.com/unis-ing/lorenz-parameter-estimation/blob/master/lorenz_pr_estimation.py) :  main `LPE` class.
* [`rules.py`](https://github.com/unis-ing/lorenz-parameter-estimation/blob/master/rules.py) : implements different estimates for the Prandtl number.
* [`lpe_helpers.py`](https://github.com/unis-ing/lorenz-parameter-estimation/blob/master/lpe_helpers.py) : helpers for the `simulate` method.
* [`initial_data/`](https://github.com/unis-ing/lorenz-parameter-estimation/tree/master/initial_data) : initial conditions to be used in solving the Lorenz equations. Each file corresponds to a Prandtl and Rayleigh number.

## Background
A write-up including the mathematical background, problem statement, and numerical/theoretical results is in progress. ✌️
