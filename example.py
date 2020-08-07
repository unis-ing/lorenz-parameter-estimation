"""
	This is an example script showing how the LPE class can be used to perform
	parameter estimation in the Lorenz system.

	Uncomment the last line BEFORE running to save simulation data in the result
	folder. Otherwise you will have to manually copy and rename the file from its
	temporary location.
"""
from lorenz_pr_estimation import *
import matplotlib.pyplot as plt

# set Prandtl and Rayleigh numbers
PR = 10
RA = 28
lpe = LPE(PR=PR, RA=RA)

############################
# simulation parameters
############################

# number of iterations
it = 1000

# initial estimate for Prandtl
pr0 = PR + 10

# nudging parameter
mu = 200

# timestep
dt = 0.001

# position and velocity error thresholds
theta = 0.4
rho = 0.06

# decay rate of theta and rho
da = db = 0.7

# no-update period
Pc = 3000

lpe.simulate(pr0=pr0, mu=mu, dt=dt, stop_it=it, rule='rule1_c1', 
			 da=da, db=db, Pc=Pc, theta=theta, rho=rho)

##################################################
# view simulation data stored at the temp path
##################################################
path = 'analysis/analysis_s1/analysis_s1_p0.h5'
f = h5.File(path, 'r')
tasks = f['tasks']
pr = np.array(tasks['pr'])
x  = np.array(tasks['x'])[:,0]
y  = np.array(tasks['y'])[:,0]
z  = np.array(tasks['z'])[:,0]
u  = np.array(tasks['u'])[:,0]
v  = np.array(tasks['v'])[:,0]
w  = np.array(tasks['w'])[:,0]
xt = np.array(tasks['xt'])[:,0]
yt = np.array(tasks['yt'])[:,0]
zt = np.array(tasks['zt'])[:,0]
ut = np.array(tasks['ut'])[:,0]
vt = np.array(tasks['vt'])[:,0]
wt = np.array(tasks['wt'])[:,0]
f.close()

prerr = abs(pr-PR)

uerr = abs(u-x)
verr = abs(v-y)
werr = abs(w-z)
positionerr = (uerr**2+verr**2+werr**2)**0.5

uterr = abs(ut-xt)
vterr = abs(vt-yt)
wterr = abs(wt-zt)
velocityerr = (uterr**2+vterr**2+wterr**2)**0.5

# generate plot of position, velocity, and Prandtl errors
T = np.linspace(0, x.size*dt, x.size)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4,6), sharex=True)
ax1.plot(T, positionerr)
ax1.set_ylabel('Position Error')
ax1.set_yscale('log')
ax1.xaxis.grid()

ax2.plot(T, velocityerr)
ax2.set_ylabel('Velocity Error')
ax2.set_yscale('log')
ax2.xaxis.grid()

ax3.plot(T, prerr)
ax3.set_xlabel('t')
ax3.set_ylabel(r'$|\sigma-\tilde{\sigma}|$')
ax3.set_yscale('log')
ax3.xaxis.grid()

plt.tight_layout()
plt.show()

# lpe.save_data()