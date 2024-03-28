import numpy as np
from scipy.special import jn  # Bessel function of the first kind
import matplotlib.pyplot as plt
from tqdm import tqdm

def fwd_euler(R, A, omega, c, tmax, num_timesteps, num_r, num_theta):
    # Parameters


    
    dt = tmax/num_timesteps  # Time step size
    dr = R / (num_r - 1)  # Radial step size
    dtheta = 2 * np.pi / num_theta  # Angular step size

    # Initialize arrays for phi at different time steps
    phi_curr = np.zeros((num_r, num_theta))  # Current time step
    phi_prev = np.zeros((num_r, num_theta))  # Previous time step

    # Initial condition: phi(r, theta, t=0) = 0
    phi_curr[:, :] = 0

    # Boundary condition: phi(r=0, theta, t) is finite
    phi_curr[0, :] = 0
    phi_tot = []
    phi_tot.append(phi_curr)

    # Time-stepping loop
    for n in tqdm(range(1, num_timesteps + 1)):

        # Update phi using forward Euler method
        for i in range(1, num_r - 1):
            for j in range(num_theta):
                # Spatial derivatives
                d2phi_dr2 = (phi_curr[i + 1, j] - 2 * phi_curr[i, j] + phi_curr[i - 1, j]) / (dr ** 2)
                d2phi_dtheta2 = (phi_curr[i, (j + 1) % num_theta] - 2 * phi_curr[i, j] + phi_curr[i, (j - 1) % num_theta]) / (dtheta ** 2)

                # Temporal derivative
                d2phi_dt2 = c ** 2 * (d2phi_dr2 + 1 / i*2 * d2phi_dtheta2)

                # Update phi using forward Euler method
                phi_next = 2 * phi_curr[i, j] - phi_prev[i, j] + dt ** 2 * d2phi_dt2
                # Update phi for the next time step
                phi_prev[i, j] = phi_curr[i, j]
                phi_curr[i, j] = phi_next
        

        # Boundary condition: phi(r=R, theta, t) = A * cos(omega * t) * cos(theta)
        phi_curr[-1, :] = A * np.cos(omega * n * dt) * np.cos(np.linspace(0, 2 * np.pi, num_theta))
        phi_tot.append(phi_curr)
    # Plot the final state of phi at the last time step
    R_grid, Theta_grid = np.meshgrid(np.linspace(0, R, num_r), np.linspace(0, 2 * np.pi, num_theta))
    X = R_grid * np.cos(Theta_grid)
    Y = R_grid * np.sin(Theta_grid)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, phi_curr.T, levels = 100, cmap='viridis')
    plt.colorbar(label='Deformation')
    plt.title('Deformation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()
