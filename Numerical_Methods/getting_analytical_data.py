from astropy.io import ascii
import numpy as np

############# doc ############
"""
The result of this code is 6 np arrays: `times`, `rs_linspace`, `thetas_linspace`,
`rs`, `thetas`, and `Z`, as well as the variable N.

No variables are hardcoded in this (execept the relative paths).
You may need to change the path below if something moves directories or changes names.

`times` is a 1D array containing the time values at which the fuinction was evaluated

`N` is the number of points that both 0 ≤ r ≤ 1 and 0 ≤ theta ≤ 2pi are discretized into

`rs` and `thetas` are N by N arrays that are the result of:
    rs_linspace = np.linspace(0,1,N)
    thetas_linspace = np.linspace(0,2*np.pi,N)

    [thetas,rs] = meshgrid(thetas_linspace,rs_linspace);

`Z` is a 3D array (N by N by len(times)) and is indexed as Z[r_index,theta_index,time]
where, for example, if you do Z[5,0,3], you will get the Z value at rs_linspace[5], thetas_linspace[0],
and times[3].
"""

def get_anal_data():
    # relative paths to the analytical data (except Z)
    times_ascii = ascii.read("../Analytical_Solution/data/times.txt")
    r_ascii = ascii.read("../Analytical_Solution/data/r_data.txt")
    theta_ascii = ascii.read("../Analytical_Solution/data/theta_data.txt")

    # Convert table to dataframe (except Z)
    times_data_frame = times_ascii.to_pandas()
    r_data_frame = r_ascii.to_pandas()
    theta_data_frame = theta_ascii.to_pandas()


    # Convert the dataframe to an np array (except Z)
    times = np.array(times_data_frame.values)
    rs = np.array(r_data_frame.values)
    thetas = np.array(theta_data_frame.values)

    # obtaining N
    N = np.shape(rs)[0]

    # make linspaces
    rs_linspace = np.linspace(0,1,N)
    thetas_linspace = np.linspace(0,2*np.pi,N)

    # fixing the weird formatting of times
    times = times[0]

    # getting the Z data
    Z = np.zeros([N,N,len(times)])
    for i in list(range(len(times))):
        i += 1
        Zi_ascii = ascii.read("../Analytical_Solution/data/Z" + str(i) + ".txt")
        Zi_data_frame = Zi_ascii.to_pandas()
        Zi = np.array(Zi_data_frame.values)
        Z[:,:,i-1] = Zi

    return N,times,rs_linspace,thetas_linspace,rs,thetas,Z