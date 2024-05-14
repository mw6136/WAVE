import numpy as np
from scipy.special import jn  # Bessel function of the first kind
import matplotlib.pyplot as plt
from tqdm import tqdm


def step_fwd(num_r, num_theta, phi_prev, phi_curr, dr, dtheta, dt, c):
    phi_next = np.zeros_like(phi_curr)
    
    # update r
    for i in range(4, num_r - 1):
        # Update phi
        for j in range(num_theta):
            # Spatial derivatives

            d2phi_dr2 = (
                phi_curr[i + 1, j] - 2 * phi_curr[i, j] + phi_curr[i - 1, j]
            ) / (dr**2)
            d2phi_dtheta2 = (
                phi_curr[i, (j + 1) % num_theta]
                - 2 * phi_curr[i, j]
                + phi_curr[i, (j - 1) % num_theta]
            ) / (dtheta**2)

            # Temporal derivative
            d2phi_dt2 = c**2 * (d2phi_dr2 + 1 / (dr * i) ** 2 * d2phi_dtheta2)

            # Update phi using forward Euler method
            phi_next_val = 2 * phi_curr[i, j] - phi_prev[i, j] + dt**2 * d2phi_dt2

            # Update phi for the next time step
            # phi_prev[i, j] = phi_curr[i, j]

            phi_next[i, j] = phi_next_val
    return phi_curr, phi_next


def fwd_euler(R, A, omega, c, tmax, num_timesteps, num_r, num_theta):
    # Parameters
    dt = tmax / num_timesteps  # Time step size
    dr = R / (num_r - 1)  # Radial step size
    dtheta = 2 * np.pi / num_theta  # Angular step size

    # Initialize arrays for phi at different time steps
    phi_curr = np.zeros((num_r, num_theta))  # Current time step

    # Boundary condition: phi(r=R, theta, t) = A * cos(omega * t) * cos(theta)
    n = 0
    phi_curr[-1, :] = (A * np.cos(omega * n * dt) * np.cos(np.linspace(0, 2 * np.pi, num_theta)))
    phi_prev = phi_curr

    phi_tot = []
    phi_tot.append(phi_curr)

    # Time-stepping loop
    for n in tqdm(range(1, num_timesteps + 1)):

        phi_curr, phi_next = step_fwd(
            num_r, num_theta, phi_prev, phi_curr, dr, dtheta, dt, c
        )

        # Apply BC: phi(r=0, theta, t) is finite
        phi_next[0, :] = 0

        # Boundary condition: phi(r=R, theta, t) = A * cos(omega * t) * cos(theta)
        phi_next[-1, :] = (
            A * np.cos(omega * n * dt) * np.cos(np.linspace(0, 2 * np.pi, num_theta))
        )

        phi_prev = phi_curr
        phi_curr = phi_next

        phi_tot.append(phi_next)

    # Plot the final state of phi at the last time step
    R_grid, Theta_grid = np.meshgrid(
        np.linspace(0, R, num_r), np.linspace(0, 2 * np.pi, num_theta)
    )
    X = R_grid * np.cos(Theta_grid)
    Y = R_grid * np.sin(Theta_grid)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(
        X, Y, phi_next.T, levels=np.linspace(-6, 6, 101), cmap="viridis"
    )
    colorbar = plt.colorbar(contour)
    colorbar.set_label("Deformation")
    plt.title("Deformation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()
    return phi_next
