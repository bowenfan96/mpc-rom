import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy import optimize
from torch.utils.data import DataLoader
from torchinfo import summary
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Enable this for hyperparameter tuning
# from functools import partial
# from ray import tune
# from ray.tune import CLIReporter

results_folder = "expReplay_results/edge09/"


class Net(nn.Module):
    def __init__(self, x_dim, u_dim, hidden_size=7):
        super(Net, self).__init__()

        self.input = nn.Linear((x_dim + u_dim), hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)

        # Output prediction on constraint obedience and cost to go, so 2 nodes
        self.h3 = nn.Linear(hidden_size, 2)

        nn.init.kaiming_uniform_(self.input.weight)
        nn.init.kaiming_uniform_(self.h1.weight)
        nn.init.kaiming_uniform_(self.h2.weight)
        nn.init.kaiming_uniform_(self.h3.weight)

    def forward(self, x_in, u_in):
        xu_in = torch.hstack((x_in, u_in))
        xu_h1 = F.leaky_relu(self.input(xu_in))
        xu_h2 = F.leaky_relu(self.h1(xu_h1))
        xu_h3 = F.leaky_relu(self.h2(xu_h2))
        out = F.leaky_relu(self.h3(xu_h3))

        ctg = out[:, 0].view(-1, 1)
        constraint = out[:, 1].view(-1, 1)

        return ctg, constraint


class SimpleNNController:
    def __init__(self, x_dim, u_dim):
        self.net = Net(x_dim, u_dim)

        self.scaler_x0 = preprocessing.MinMaxScaler()
        self.scaler_x1 = preprocessing.MinMaxScaler()
        self.scaler_u = preprocessing.MinMaxScaler()
        self.scaler_ctg = preprocessing.MinMaxScaler()
        self.scaler_constraint = preprocessing.MinMaxScaler()

    def process_and_normalize_data(self, dataframe):
        x0 = dataframe.filter(regex="x0")
        x1 = dataframe.filter(regex="x1")
        u = dataframe.filter(regex="u")
        ctg = dataframe.filter(regex="ctg")
        constraint = dataframe.filter(regex="path_diff")

        x0 = x0.to_numpy(dtype=np.float32)
        x1 = x1.to_numpy(dtype=np.float32)
        u = u.to_numpy(dtype=np.float32)
        ctg = ctg.to_numpy(dtype=np.float32)
        constraint = constraint.to_numpy(dtype=np.float32)

        self.scaler_x0.fit(x0)
        self.scaler_x1.fit(x1)
        self.scaler_u.fit(u)
        self.scaler_ctg.fit(ctg)
        self.scaler_constraint.fit(constraint)

        x0 = self.scaler_x0.transform(x0)
        x1 = self.scaler_x1.transform(x1)
        u = self.scaler_u.transform(u)
        ctg = self.scaler_ctg.transform(ctg)
        constraint = self.scaler_constraint.transform(constraint)

        x0_tensor = torch.tensor(x0)
        x1_tensor = torch.tensor(x1)
        u_tensor = torch.tensor(u)
        ctg_tensor = torch.tensor(ctg)
        constraint_tensor = torch.tensor(constraint)

        x_tensor = torch.hstack((x0_tensor, x1_tensor))

        return x_tensor, u_tensor, ctg_tensor, constraint_tensor

    def fit(self, dataframe):
        x, u, ctg, cst = self.process_and_normalize_data(dataframe)
        self.net.train()

        minibatch = torch.utils.data.TensorDataset(x, u, ctg, cst)
        mb_loader = torch.utils.data.DataLoader(minibatch, batch_size=120, shuffle=False)

        ctg_optimizer = optim.SGD(self.net.parameters(), lr=0.05)
        cst_optimizer = optim.SGD(self.net.parameters(), lr=0.05)
        ctg_criterion = nn.MSELoss()
        cst_criterion = nn.MSELoss()

        for epoch in range(400):
            for x_mb, u_mb, ctg_mb, cst_mb in mb_loader:
                ctg_optimizer.zero_grad()
                cst_optimizer.zero_grad()

                ctg_pred, cst_pred = self.net(x_mb, u_mb)

                loss_ctg = ctg_criterion(ctg_pred, ctg_mb)
                loss_cst = cst_criterion(cst_pred, cst_mb)
                loss = loss_ctg + loss_cst
                loss.backward()

                ctg_optimizer.step()
                cst_optimizer.step()

            # Test on the whole dataset at this epoch
            self.net.eval()
            with torch.no_grad():
                ctg_pred, cst_pred = self.net(x, u)

                loss_ctg = ctg_criterion(ctg_pred, ctg)
                loss_cst = cst_criterion(cst_pred, cst)

                print("Epoch {}: loss_ctg = {} and loss_cst = {}".format(epoch, loss_ctg, loss_cst))
            self.net.train()

    def predict_ctg_cst(self, x, u):
        # Process x
        x = np.array(x, dtype=np.float32)
        x = x.flatten()
        x0 = np.array(x[0]).reshape(1, 1)
        x1 = np.array(x[1]).reshape(1, 1)
        u = np.array(u, dtype=np.float32).reshape(1, 1)

        x0_scaled = self.scaler_x0.transform(x0)
        x1_scaled = self.scaler_x1.transform(x1)
        u_scaled = self.scaler_u.transform(u)

        x0_tensor = torch.tensor(x0_scaled)
        x1_tensor = torch.tensor(x1_scaled)
        u_tensor = torch.tensor(u_scaled)
        x_tensor = torch.hstack((x0_tensor, x1_tensor))

        self.net.eval()
        with torch.no_grad():
            ctg_pred, cst_pred = self.net(x_tensor, u_tensor)

        ctg_pred = ctg_pred.detach().numpy()
        cst_pred = cst_pred.detach().numpy()

        # Scale back ctg and cst
        ctg_pred = self.scaler_ctg.inverse_transform(ctg_pred)
        cst_pred = self.scaler_constraint.inverse_transform(cst_pred)
        ctg_pred = ctg_pred.item()
        cst_pred = cst_pred.item()

        # u must be between [-20, 20], so if basinhopper tries an invalid u, we penalize the ctg
        if np.abs(u) > 20:
            return ctg_pred, 1E9
        else:
            return ctg_pred, cst_pred

    def get_u_opt(self, x, mode="basinhopper"):
        x = np.array(x).flatten().reshape((1, 2))

        if mode == "grid":
            best_u = -20
            best_ctg = np.inf

            for u in np.linspace(start=-20, stop=20, num=81):
                ctg_pred, cst_pred = self.predict_ctg_cst(x, u)
                if ctg_pred < best_ctg and cst_pred <= 0:
                    best_u = u
                    best_ctg = ctg_pred

            # Add some noise to encourage exploration
            best_u_with_noise = best_u + np.random.uniform(low=-0.2, high=0.2, size=None)
            print("Best u given x = {} is {}, adding noise = {}"
                  .format(x.flatten().round(4), round(best_u, 4), round(best_u_with_noise, 4))
                  )
            return best_u_with_noise

        elif mode == "basinhopper":
            def basinhopper_helper(u_bh, *arg_x_bh):
                x_bh = arg_x_bh[0]
                x_bh = np.array(x_bh).flatten()
                ctg_pred_bh, cst_pred_bh = self.predict_ctg_cst(x_bh, u_bh)
                # If constraints are broken, then we apply a 10x penalty to the cost to go
                # The greater the penalty, the more we drive a wedge between local minima for the gradient descent
                if cst_pred_bh > 0:
                    ctg_pred_bh = ctg_pred_bh * 10
                return ctg_pred_bh

            # Configure options for the local minimizer (Powell)
            gd_options = {}
            # gd_options["maxiter"] = 2
            # gd_options["disp"] = True
            # gd_options["eps"] = 1

            # Specify bounds to send to the Powell minimizer
            bounds = optimize.Bounds(lb=np.array([-20], dtype=int), ub=np.array([20], dtype=int))

            # Powell is chosen because it is the only gradientless method that can handle bounds
            # We need to it to gradientless because our input to output mapping is not differentiable
            min_kwargs = {
                "args": x,
                "method": 'Powell',
                "options": gd_options,
                "bounds": bounds
            }
            result = optimize.basinhopping(
                func=basinhopper_helper, x0=[0], niter=5, minimizer_kwargs=min_kwargs
            )
            # result["x"] is the optimal u, don't be confused by the name!
            best_u = np.array(result["x"]).flatten()[0]
            best_u_with_noise = best_u + np.random.uniform(low=-0.2, high=0.2, size=None)
            print("Best u given x = {} is {}, adding noise = {}"
                  .format(x.flatten().round(4), round(best_u, 4), round(best_u_with_noise, 4))
                  )
            return best_u_with_noise


def pickle_model(model, round_num):
    pickle_filename = results_folder + "R{}_".format(round_num+1) + "simple_nn_controller.pickle"
    with open(pickle_filename, "wb") as file:
        pickle.dump(model, file)
    print("Pickled model to " + pickle_filename)
    return pickle_filename


def train_and_pickle(round_num, trajectory_df_filename):
    print("Training with dataset: " + trajectory_df_filename)
    data = pd.read_csv(trajectory_df_filename)
    simple_nn = SimpleNNController(x_dim=2, u_dim=1)
    simple_nn.fit(data)
    pickle_filename = pickle_model(simple_nn, round_num)
    return pickle_filename


if __name__ == "__main__":
    data = pd.read_csv("simple_240_trajectories_df.csv")
    simple_nn = SimpleNNController(x_dim=2, u_dim=1)
    simple_nn.fit(data)

    with open("simple_nn_controller_240.pickle", "wb") as pickle_file:
        pickle.dump(simple_nn, pickle_file)
