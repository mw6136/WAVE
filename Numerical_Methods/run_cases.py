print("Starting running run_cases.py")

from fwd_euler import fwd_euler
from getting_analytical_data import get_anal_data
from bwd_euler import bwd_euler

'''
# getting analytical data
print("Importing analytical data")
N,times,rs_linspace,thetas_linspace,rs,thetas,Z = get_anal_data()
print("Analytical data imported")

tmax = times[-1]
num_r = N # Number of radial grid points
num_theta = N  # Number of angular grid points
num_t = 1000  # Number of time steps
'''



# Constant Values
R = 1.0
A = 1.0
omega = 1.0
c = 1.0



tmax = 0.5
num_r = 100  # Number of radial grid points
num_theta = 100  # Number of angular grid points
num_t = 1000  # Number of time steps


#print("Starting running Forward Euler")
#fwd_euler(R, A, omega, c, tmax, num_t, num_r, num_theta)
#print("Forward Euler completed")

# print("Starting running Backward Euler")
# bwd_euler(R, A, omega, c, tmax, num_t, num_r, num_theta)
# print("Forward Euler completed")


print("Starting running discrete")
with open("discrete.py") as file:
    exec(file.read())
print("Forward Euler discrete")