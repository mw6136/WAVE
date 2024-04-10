from fwd_euler import fwd_euler
from getting_analytical_data import get_anal_data

# getting analytical data
N,times,rs_linspace,thetas_linspace,rs,thetas,Z = get_anal_data()


# Constant Values
R = 1.0
A = 1.0
omega = 1.0
c = 1.0
tmax = times[-1]

#create grid size
num_r = N # Number of radial grid points
num_theta = N  # Number of angular grid points
num_t = len(times)  # Number of time steps

fwd_euler(R, A, omega, c, tmax, num_t, num_r, num_theta)