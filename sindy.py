import numpy as np
import pysindy as ps
import pandas as pd

matrices_folder = "matrices/simple/"
results_folder = "results_csv/simple/"
plots_folder = "results_plots/simple/"

data = pd.read_csv(results_folder + "all_simple.csv", sep=',')

dt = 1/6

split_points = np.arange(start=6, stop=696, step=6, dtype=int)
# print(split_points)

x0 = data.filter(regex='mpc_x_0').to_numpy().flatten()
x0_list = np.split(x0, split_points)

x1 = data.filter(regex='mpc_x_1').to_numpy().flatten()
x1_list = np.split(x1, split_points)

u = data.filter(regex='u_0').to_numpy().flatten()
u_list = np.split(u, split_points)

X = np.stack((x0_list, x1_list), axis=-1)
print(X)
X = list(X)
print(X)

model = ps.SINDy()
model.fit(X, u=u_list, multiple_trajectories=True)
model.print()
