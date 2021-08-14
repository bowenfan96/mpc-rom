import numpy as np
import pysindy as ps
import pandas as pd

matrices_folder = "matrices/"
results_folder = "results_csv/"
plots_folder = "results_plots/"

data = pd.read_csv("simple_system_unsorted_data/simple_proper.csv", sep=',')

# dt = 1/6

split_points = np.arange(start=11, stop=1100, step=11, dtype=int)
# print(split_points)

x0 = data.filter(regex='x1').to_numpy().flatten()
x0_list = np.split(x0, split_points)

x1 = data.filter(regex='x2').to_numpy().flatten()
x1_list = np.split(x1, split_points)

u = data.filter(regex='u').to_numpy().flatten()
u_list = np.split(u, split_points)

X = []
for x0, x1 in zip(x0_list, x1_list):
    X.append(np.hstack((x0.reshape(-1,1), x1.reshape(-1,1))))
print(X)

model = ps.SINDy(t_default=0.1)
model.fit(X, u=u_list, multiple_trajectories=True)
model.print()
