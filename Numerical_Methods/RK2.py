import numpy as np
from scipy.special import jn  # Bessel function of the first kind
import matplotlib.pyplot as plt
from tqdm import tqdm





def step_RK2(num_r, num_theta, phi_prev, phi_current, dr, dtheta, dt, c):
        def f(phi_curr, i, j):
            d2phi_dr2 = (phi_curr[i + 1, j] - 2 * phi_curr[i, j] + phi_curr[i - 1, j]) / (dr ** 2)
            d2phi_dtheta2 = (phi_curr[i, (j + 1) % num_theta] - 2 * phi_curr[i, j] + phi_curr[i, (j - 1) % num_theta]) / (dtheta ** 2)
            d2phi_dt2 = c ** 2 * (d2phi_dr2 + 1 / (dr*i)**2 * d2phi_dtheta2)
            return d2phi_dt2
        
        phi_next = np.zeros_like(phi_current)
        phi_2 = np.zeros_like(phi_current)

        k1_v = np.zeros_like(phi_current)
        for i in range(5, num_r - 1):
            #Update phi
            for j in range(num_theta):
                #k1_x = v
                k1_v[i,j] = dt * f(phi_current, i, j)
        phi_2[2: num_r - 1, :] = phi_current[2: num_r - 1, :] + (phi_current[2: num_r - 1, :] - phi_prev[2: num_r - 1, :])/2 + dt/4 * k1_v[2: num_r - 1, :]

        k2_v = np.zeros_like(phi_current)
        for i in range(5, num_r - 1):
            #Update phi
            for j in range(num_theta):
                #k1_x = v
                k2_v[i,j] = dt * f(phi_2, i, j)

        #k = (k1_v + k2_v)/2
        phi_next[2: num_r - 1, :] = phi_current[2: num_r - 1, :] + (phi_current[2: num_r - 1, :] - phi_prev[2: num_r - 1, :]) + dt * k1_v[2: num_r - 1, :]
        return phi_current, phi_next


def RK2(R, A, omega, c, tmax, num_timesteps, num_r, num_theta):
    # Parameters
    dt = tmax/num_timesteps*1.01  # Time step size
    dr = R / (num_r - 1)  # Radial step size
    dtheta = 2 * np.pi / num_theta  # Angular step size

    # Initialize arrays for phi at different time steps
    phi_curr = np.zeros((num_r, num_theta))  # Current time step

    # Boundary condition: phi(r=R, theta, t) = A * cos(omega * t) * cos(theta)
    n = 0
    phi_curr[-1, :] = A * np.cos(omega * n * dt) * np.cos(np.linspace(0, 2 * np.pi, num_theta))
    phi_prev = phi_curr

    phi_tot = []
    phi_tot.append(phi_curr)
    
 
    
    
    # Time-stepping loop
    for n in tqdm(range(1, num_timesteps + 1)):
        

        phi_curr, phi_next = step_RK2(num_r, num_theta, phi_prev, phi_curr, dr, dtheta, dt, c)
        
      #  print(np.max(np.abs(phi_curr - phi_next)))
        #Apply BC: phi(r=0, theta, t) is finite
        phi_next[0,:] = 0
        # Boundary condition: phi(r=R, theta, t) = A * cos(omega * t) * cos(theta)
        phi_next[-1, :] = A * np.cos(omega * n * dt) * np.cos(np.linspace(0, 2 * np.pi, num_theta))
        
        phi_prev = phi_curr
        phi_curr = phi_next
        
        phi_tot.append(phi_next)
    # Plot the final state of phi at the last time step
    R_grid, Theta_grid = np.meshgrid(np.linspace(0, R, num_r), np.linspace(0, 2 * np.pi, num_theta))
    X = R_grid * np.cos(Theta_grid)
    Y = R_grid * np.sin(Theta_grid)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, phi_next.T, levels=np.linspace(-6, 6, 101), cmap='viridis')
    colorbar = plt.colorbar(contour)
    colorbar.set_label('Deformation')
    plt.title('Deformation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()
    return phi_next