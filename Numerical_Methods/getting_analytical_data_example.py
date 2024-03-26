#!/usr/bin/env python

from astropy.io import ascii
import numpy as np

############# doc ############
"""
The result of this code is 4 np arrays: `times`, `X`, `Y`, and `Z`
No variables are hardcoded in this (execept the path)
You may need to change the path below if something moves directories

`times` is a 1D array containing the time values at which the fuinction was evaluated
`X` and `Y` are N by N arrays in the same format as np.meshgrid
`Z` is a 3D array (num timesteps by N by N) and is indexed as Z[timestep,x_index,y_index]

Note that since we're modeling something cylindrical, `Z` has real values inside of it where the corresponding x and y values are 
such that sqrt(x^2 + y^2) ≤ R = 1, and NaNs outside of this.
"""

# relative paths to the analytical data
times_ascii = ascii.read("../Analytical_Solution/data/times_data.txt")
X_ascii = ascii.read("../Analytical_Solution/data/X_data.txt")
Y_ascii = ascii.read("../Analytical_Solution/data/Y_data.txt")
Z_3D_ascii = ascii.read("../Analytical_Solution/data/Z_3D_data.txt")

# Convert table to dataframe
times_data_frame = times_ascii.to_pandas()
X_data_frame = X_ascii.to_pandas()
Y_data_frame = Y_ascii.to_pandas()
Z_data_frame = Z_3D_ascii.to_pandas()

# Convert the dataframe to an np array
times = np.array(times_data_frame.values)
X = np.array(X_data_frame.values)
Y = np.array(Y_data_frame.values)
Z_wide = np.array(Z_data_frame.values)

times = times[0]

num_points_per_side = int(Z_wide.shape[0])
num_time_steps = int(Z_wide.shape[1] / num_points_per_side)

Z = np.zeros((num_time_steps, num_points_per_side, num_points_per_side))
for t in list(range(num_time_steps)):
    for i in list(range(num_points_per_side)):
        for j in list(range(num_points_per_side)):
            Z[t,i,j] = Z_wide[i,t*num_points_per_side +  j]