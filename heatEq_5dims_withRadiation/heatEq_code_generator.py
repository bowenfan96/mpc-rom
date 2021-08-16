# This file exists to parse A and B matrices into pyomo entries
# While it seems silly, this is necessary because pyomo's simulator (which calls casadi) does not support matrices
# Namely, it cannot simulate variables indexed by more than 1 set (so each variable can only be indexed by time)
# It also doesn't support if statements within the model, so this seems to be the only way

import numpy as np

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

for i in range(1, 19):
    print("self.model.x{} = Var(self.model.time)".format(i))
    print("self.model.x{}_dot = DerivativeVar(self.model.x{}, wrt=self.model.time)".format(i, i))
    print("self.model.x{}[0].fix(x_init[{}])".format(i, i))

for i in range(1, 19):
    print(
        '''
        def _ode_x{}(m, _t):
            return m.x{}_dot[_t] == self.A[{}][{}-1] * m.x{}[_t] + self.A[{}][{}] * m.x{}[_t] + self.A[{}][{}+1] * m.x{}[_t]
        self.model.x{}_ode = Constraint(self.model.time, rule=_ode_x{})\n
        '''.format(i, i, i, i, i-1, i, i, i, i, i, i+1, i, i)
    )

for i in range(1, 19):
    print("temp_x.append(value(self.model.x{}[time]))".format(i))

for i in range(20):
    print(" + (m.x{}[_t] ** 4 - env_temp ** 4)".format(i))
