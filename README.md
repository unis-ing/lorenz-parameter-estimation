Code, data, and etc. produced in an ongoing research project under Professor Vincent Martinez at Hunter College.

## Prerequisites
* `dedalus` is used to solve the Lorenz equations. An installation guide can be found [here](https://dedalus-project.readthedocs.io/en/latest/installation.html).
* `h5py` is used to read and write data.

To check that all requirements are met, run the sample script in this repository using:
```
python3 example.py
```

## Files
* [`lorenz_pr_estimation.py`](https://github.com/unis-ing/lorenz-parameter-estimation/blob/master/lorenz_pr_estimation.py) :  main `LPE` class.
* [`rules.py`](https://github.com/unis-ing/lorenz-parameter-estimation/blob/master/rules.py) : implements different estimates for the Prandtl number.
* [`helpers.py`](https://github.com/unis-ing/lorenz-parameter-estimation/blob/master/helpers.py) : helpers for the `simulate` method.
* [`example.py`](https://github.com/unis-ing/lorenz-parameter-estimation/blob/master/example.py) : sample script using `LPE` to perform parameter estimation in the Lorenz system.
* [`initial_data/`](https://github.com/unis-ing/lorenz-parameter-estimation/tree/master/initial_data) : initial conditions to be used in solving the Lorenz equations. Each file corresponds to a Prandtl number and Rayleigh number.
* [`result_data/`](https://github.com/unis-ing/lorenz-parameter-estimation/tree/master/result_data) : simulation data used in the "Algorithms" section of the background paper (files larger than 1GB omitted)
* [`thresholds/`](https://github.com/unis-ing/lorenz-parameter-estimation/tree/master/thresholds) : reasonable initial error thresholds for `rule1_c*`. Each file corresponds to a Prandtl number, Rayleigh number, and nudging parameter.

## Write-up
A first draft of the project write-up is available [here](https://github.com/unis-ing/lorenz-parameter-estimation/blob/master/writeup.pdf) 🥳. It includes: a statement of the research problem, background information on the Lorenz equations, algorithms for performing parameter estimation (implemented by the code in this repository), numerical results, and additional questions we have re: the aforementioned problem.
