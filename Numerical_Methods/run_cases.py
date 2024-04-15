from fwd_euler import fwd_euler


# Constant Values
R = 1.0
A = 1.0
omega = 1.0
c = 1.0
tmax = 0.5

#create grid size
num_r = 100  # Number of radial grid points
num_theta = 100  # Number of angular grid points
num_t = 1000  # Number of time steps

fwd_euler(R, A, omega, c, tmax, num_t, num_r, num_theta)