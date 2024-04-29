import numpy as np
from getting_analytical_data import get_anal_data
from numba import njit,jit

N,times,rs_linspace,thetas_linspace,rs,thetas,Z_anal = get_anal_data()

# takes NxN matrix, returns 1xN^2 flattened matrix. u[i,j] --> u[i+Nj=k]
# @jit(forceobj=True)
@njit
def flatten(matrix):
#     return matrix.reshape(-1)
    size = len(matrix)
    line = np.zeros((size**2))
    count = 0
    for i in list(range(size)):
        for j in list(range(size)):
            line[count] = matrix[i,j]
            count += 1
    return line


A = np.zeros((N**2,N**2))