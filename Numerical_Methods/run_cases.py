from fwd_euler import fwd_euler
from RK2 import RK2
from fft_solver import fft_solver
from getting_analytical_data import *

import matplotlib.pyplot as plt


def calculate_L2_error(approximate_solution, exact_solution):
    squared_difference = (approximate_solution - exact_solution) ** 2
    mean_squared_difference = np.mean(squared_difference)
    L2_error = np.sqrt(mean_squared_difference)
    return L2_error


print("Starting running run_cases.py")


# getting analytical data
print("Importing analytical data")
N, times, rs_linspace, thetas_linspace, rs, thetas, Z = get_anal_data()
print("Analytical data imported")

tmax = times[-1]
num_r = N  # Number of radial grid points
num_theta = N  # Number of angular grid points
num_t = 1000  # Number of time steps


N, times, rs_linspace, thetas_linspace, rs, thetas, Z_0p5 = get_anal_data_specific_time(0.5)

# Constant Values
R = 1.0
A = 1.0
omega = 1.0
c = 1.0

tmax = 0.5
num_r = 200  # Number of radial grid points
num_theta = 200  # Number of angular grid points
num_t = 1000  # Number of time steps

############### Forward Euler #######################
print("Starting running Forward Euler")
fwd = fwd_euler(R, A, omega, c, tmax, num_t, num_r, num_theta)
print("Forward Euler completed")
L2_fwd = calculate_L2_error(Z_0p5, fwd)
print(L2_fwd)
fwd_error = np.abs(fwd - Z_0p5)

################# RK2 ###############################
print("Starting running RK2")
RK2_val = RK2(R, A, omega, c, tmax, num_t, num_r, num_theta)
print("RK2 completed")
L2_RK2 = calculate_L2_error(Z_0p5, RK2_val)
print(L2_RK2)
RK2_error = np.abs(RK2_val - Z_0p5)

################# FFT ###############################
print("Starting running FFT")
FFT_val = fft_solver(R, A, omega, c, tmax, num_t, num_r, num_theta)
print("FFT completed")
L2_FFT = calculate_L2_error(Z_0p5, FFT_val)
print(L2_FFT)
FFT_error = np.abs(FFT_val - Z_0p5)

################# Discrete ##########################
print("Starting running discrete")
with open("discrete.py") as file:
    exec(file.read())
print("Forward Euler discrete completed")
