# This file exists to parse A and B matrices into pyomo entries
# While it seems silly, this is necessary because pyomo's simulator does not support matrices
# Namely, it cannot simulate variables indexed by more than 1 set (so each variable can only be indexed by time)

import numpy as np

A_matrix_file = "A.csv"
B_matrix_file = "B.csv"