# Code source: https://modred.readthedocs.io/en/stable/tutorial_model_reduction.html

import os
import numpy as np
import modred as mr

# Create directory for output files
out_dir = 'rom_ex1_out'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

N = 20
# ----- GENERATE THE MODEL MATRICES -----
# Apply the method of lines on the heat equation to generate the A matrix
# Length of the rod = 1 m
# Number of segments = number of discretization points - 1 (as 2 ends take up 2 points)
length = 1
num_segments = N - 1
# Thermal diffusivity alpha
alpha = 0.1
segment_length = length / num_segments
# Constant
c = alpha / (segment_length ** 2)

# Generate A matrix
A_mat = np.zeros(shape=(N, N))
for row in range(A_mat.shape[0]):
    for col in range(A_mat.shape[1]):
        if row == col:
            A_mat[row][col] = -2
        elif abs(row - col) == 1:
            A_mat[row][col] = 1
        else:
            A_mat[row][col] = 0
# Multiply constant to all elements in A
A = c * A_mat

# Generate B matrix
# Two sources of heat at each end of the rod
num_heaters = 2
B_mat = np.zeros(shape=(N, num_heaters))
# First heater on the left
B_mat[0][0] = 1
# Second heater on the right
B_mat[N - 1][num_heaters - 1] = 1
# Multiply constant to all elements in B
B = c * B_mat


num_basis_vecs = 10

# Create random modes
basis_vecs = np.random.random((nx, num_basis_vecs))

# Perform Galerkin projection and save data
LTI_proj = mr.LTIGalerkinProjectionArrays(basis_vecs)
A_reduced, B_reduced, C_reduced = LTI_proj.compute_model(
    A.dot(basis_vecs), B, C.dot(basis_vecs))
LTI_proj.put_model(
    '%s/A_reduced.txt' % out_dir, '%s/B_reduced.txt' % out_dir,
    '%s/C_reduced.txt' % out_dir)
