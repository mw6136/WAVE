import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from tqdm import tqdm


def fft_solver(R, A, omega, c, tmax, num_timesteps, num_r, num_theta):

    # Parameters
    dt = tmax / num_timesteps  # Time step size
    dr = R / (num_r - 1)  # Radial step size
    dtheta = 2 * np.pi / num_theta  # Angular step size

    # Create radial and angular grids
    r = np.linspace(0, R, num_r)
    theta = np.linspace(0, 2 * np.pi, num_theta)
    R_grid, Theta_grid = np.meshgrid(r, theta, indexing="ij")

    # Initialize arrays for phi at different time steps
    phi_curr = np.zeros((num_r, num_theta))  # Current time step
    phi_prev = np.zeros((num_r, num_theta))  # Previous time step

    # Initial condition: phi(r, theta, t=0) = 0
    phi_curr[:, :] = 0

    # Boundary condition: phi(r=0, theta, t) is finite
    phi_curr[0, :] = 0
    phi_tot = []
    phi_tot.append(phi_curr)

    # Initialize arrays to store the Fourier-transformed solutions
    phi_hat_curr = fft(phi_curr, axis=0)
    phi_hat_prev = fft(phi_prev, axis=0)

    # Time-stepping loop
    for n in tqdm(range(1, num_timesteps + 1)):

        # Calculate frequencies
        freq_r = fftfreq(num_r, dr)
        freq_theta = fftfreq(num_theta, dtheta)

        # Calculate Laplacian in frequency domain
        k_r_squared = (2 * np.pi * freq_r) ** 2
        k_theta_squared = (2 * np.pi * freq_theta) ** 2
        laplacian = -(c**2) * (k_r_squared[:, np.newaxis] + k_theta_squared)

        # Update phi in frequency domain using Forward Euler
        phi_hat_next = (
            2 * phi_hat_curr - phi_hat_prev + dt**2 * laplacian * phi_hat_curr
        )

        # Transform back to spatial domain
        phi_next = np.real(ifft(phi_hat_next, axis=0))

        # Update phi for the next time step
        phi_prev[:] = phi_curr[:]
        phi_curr[:] = phi_next[:]

        # Boundary condition: phi(r=R, theta, t) = A * cos(omega * t) * cos(theta)
        phi_curr[-1, :] = (
            A * np.cos(omega * n * dt) * np.cos(np.linspace(0, 2 * np.pi, num_theta))
        )
        phi_tot.append(phi_curr)

    # Plot the final state of phi at the last time step
    R_grid, Theta_grid = np.meshgrid(
        np.linspace(0, R, num_r), np.linspace(0, 2 * np.pi, num_theta)
    )
    X = R_grid * np.cos(Theta_grid)
    Y = R_grid * np.sin(Theta_grid)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, phi_curr.T, levels=100, cmap="viridis")
    colorbar = plt.colorbar(contour)
    colorbar.set_label("Deformation")
    contour.set_clim(-6, 6)  # Set colorbar limits to -6 to 6
    plt.title("Deformation")
    plt.xlabel("X")
    plt.axis("equal")
    plt.show()
