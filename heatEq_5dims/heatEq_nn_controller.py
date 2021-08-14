from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from torch.utils.data import DataLoader
from torchinfo import summary
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import optimize

# Enable this for hyperparameter tuning
# from functools import partial
# from ray import tune
# from ray.tune import CLIReporter

import time as python_timer

data_folder = "data/"
results_folder = "expReplay_results/point07_12nodes_h1ctg/"


class xMOR(nn.Module):
    def __init__(self, x_dim, x_rom_dim=5):
        super(xMOR, self).__init__()

        self.input = nn.Linear(x_dim, (x_dim + x_rom_dim) // 2)
        self.h1 = nn.Linear((x_dim + x_rom_dim) // 2, (x_dim + x_rom_dim) // 2)
        self.h2 = nn.Linear((x_dim + x_rom_dim) // 2, (x_dim + x_rom_dim) // 2)
        self.h3 = nn.Linear((x_dim + x_rom_dim) // 2, x_rom_dim)

        nn.init.kaiming_uniform_(self.input.weight)
        nn.init.kaiming_uniform_(self.h1.weight)
        nn.init.kaiming_uniform_(self.h2.weight)
        nn.init.kaiming_uniform_(self.h3.weight)

    def forward(self, x_in):
        x_h1 = F.leaky_relu(self.input(x_in))
        x_h2 = F.leaky_relu(self.h1(x_h1))
        x_h3 = F.leaky_relu(self.h2(x_h2))
        x_rom_out = F.leaky_relu(self.h3(x_h3))

        return x_rom_out


class Net(nn.Module):
    def __init__(self, x_rom_dim, u_dim, hidden_size=12):
        super(Net, self).__init__()

        self.input = nn.Linear((x_rom_dim + u_dim), hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        # Output prediction on constraint obedience and cost to go, so 2 nodes
        self.h3 = nn.Linear(hidden_size, 2)

        nn.init.kaiming_uniform_(self.input.weight)
        nn.init.kaiming_uniform_(self.h1.weight)
        nn.init.kaiming_uniform_(self.h2.weight)
        nn.init.kaiming_uniform_(self.h3.weight)

    def forward(self, x_rom_in, u_in):
        xu_in = torch.hstack((x_rom_in, u_in))

        xu_h1 = F.leaky_relu(self.input(xu_in))
        xu_h2 = F.leaky_relu(self.h1(xu_h1))
        xu_h3 = F.leaky_relu(self.h2(xu_h2))
        out = F.leaky_relu(self.h3(xu_h3))

        ctg = out[:, 0].view(-1, 1)
        constraint = out[:, 1].view(-1, 1)

        return ctg, constraint


class HeatEqNNController:
    def __init__(self, x_dim, x_rom_dim, u_dim):
        self.x_mor = xMOR(x_dim, x_rom_dim)
        self.net = Net(x_rom_dim, u_dim, hidden_size=7)

        self.scaler_x = preprocessing.MinMaxScaler()
        self.scaler_u = preprocessing.MinMaxScaler()
        self.scaler_ctg = preprocessing.MinMaxScaler()
        self.scaler_constraint = preprocessing.MinMaxScaler()

    def process_and_normalize_data(self, dataframe):
        x = []
        for i in range(20):
            x.append(dataframe["x{}".format(i)].to_numpy(dtype=np.float32))
        u0 = dataframe["u0"]
        u1 = dataframe["u1"]
        ctg = dataframe["ctg"]
        constraint = dataframe["path_diff"]

        # Tranpose x to obtain a 2D array with shape (num_trajectories * time, 20)
        x = np.array(x).transpose()

        u0 = u0.to_numpy(dtype=np.float32).reshape(-1, 1)
        u1 = u1.to_numpy(dtype=np.float32).reshape(-1, 1)
        ctg = ctg.to_numpy(dtype=np.float32).reshape(-1, 1)
        constraint = constraint.to_numpy(dtype=np.float32).reshape(-1, 1)

        u0_and_u1 = np.vstack((u0, u1))

        # x is fit with n rows and 20 columns, so we need to reshape it to this for transform
        self.scaler_x.fit(x)
        self.scaler_u.fit(u0_and_u1)
        self.scaler_ctg.fit(ctg)
        self.scaler_constraint.fit(constraint)

        x = self.scaler_x.transform(x)
        u0 = self.scaler_u.transform(u0)
        u1 = self.scaler_u.transform(u1)
        ctg = self.scaler_ctg.transform(ctg)
        constraint = self.scaler_constraint.transform(constraint)

        x_tensor = torch.tensor(x)
        u0_tensor = torch.tensor(u0)
        u1_tensor = torch.tensor(u1)
        ctg_tensor = torch.tensor(ctg)
        constraint_tensor = torch.tensor(constraint)

        u_tensor = torch.hstack((u0_tensor, u1_tensor))

        return x_tensor, u_tensor, ctg_tensor, constraint_tensor

    def fit(self, dataframe):
        x, u, ctg, cst = self.process_and_normalize_data(dataframe)

        self.x_mor.train()
        self.net.train()

        minibatch = torch.utils.data.TensorDataset(x, u, ctg, cst)
        mb_loader = torch.utils.data.DataLoader(minibatch, batch_size=120, shuffle=False)

        param_wrapper = nn.ParameterList()
        param_wrapper.extend(self.x_mor.parameters())
        param_wrapper.extend(self.net.parameters())

        ctg_optimizer = optim.SGD(param_wrapper, lr=0.05)
        cst_optimizer = optim.SGD(param_wrapper, lr=0.05)
        ctg_criterion = nn.MSELoss()
        cst_criterion = nn.MSELoss()

        for epoch in range(500):
            for x_mb, u_mb, ctg_mb, cst_mb in mb_loader:
                ctg_optimizer.zero_grad()
                cst_optimizer.zero_grad()

                x_rom_mb = self.x_mor(x_mb)
                ctg_pred, cst_pred = self.net(x_rom_mb, u_mb)

                loss_ctg = ctg_criterion(ctg_pred, ctg_mb)
                loss_cst = cst_criterion(cst_pred, cst_mb)
                loss = loss_ctg + loss_cst
                loss.backward()

                ctg_optimizer.step()
                cst_optimizer.step()

            # Test on the whole dataset at this epoch
            self.x_mor.eval()
            self.net.eval()
            with torch.no_grad():
                x_rom = self.x_mor(x)
                ctg_pred, cst_pred = self.net(x_rom, u)

                loss_ctg = ctg_criterion(ctg_pred, ctg)
                loss_cst = cst_criterion(cst_pred, cst)

                print("Epoch {}: loss_ctg = {} and loss_cst = {}".format(epoch, loss_ctg, loss_cst))
            self.x_mor.train()
            self.net.train()

    def predict_ctg_cst(self, x, u):
        # Process x and u
        # x.shape should be (20, )
        # u.shape should be (2, )
        # x is fit with n rows and 20 columns, so we need to reshape it to this for transform
        x = np.array(x, dtype=np.float32).flatten().reshape(-1, 20)
        u = np.array(u, dtype=np.float32).flatten()

        x_scaled = self.scaler_x.transform(x)
        u0_scaled = self.scaler_u.transform(u[0].reshape(1, -1))
        u1_scaled = self.scaler_u.transform(u[1].reshape(1, -1))

        x_tensor = torch.tensor(x_scaled)
        u0_tensor = torch.tensor(u0_scaled)
        u1_tensor = torch.tensor(u1_scaled)
        u_tensor = torch.hstack((u0_tensor, u1_tensor))

        self.x_mor.eval()
        self.net.eval()
        with torch.no_grad():
            x_rom = self.x_mor(x_tensor)
            ctg_pred, cst_pred = self.net(x_rom, u_tensor)

        ctg_pred = ctg_pred.detach().numpy()
        cst_pred = cst_pred.detach().numpy()

        # Scale back ctg and cst
        ctg_pred = self.scaler_ctg.inverse_transform(ctg_pred)
        cst_pred = self.scaler_constraint.inverse_transform(cst_pred)
        ctg_pred = ctg_pred.item()
        cst_pred = cst_pred.item()

        # u must be between [173, 373], so if basinhopper tries an invalid u, we penalize the ctg
        if np.any(u < 173) or np.any(u > 373):
            return ctg_pred, -1E9
        else:
            return ctg_pred, cst_pred

    def get_u_opt(self, x, mode="grid"):
        x = np.array(x).flatten()

        if mode == "grid":
            best_u = [273, 273]
            best_ctg = np.inf

            for u0 in np.linspace(start=173, stop=373, num=100):
                for u1 in np.linspace(start=173, stop=373, num=100):
                    ctg_pred, cst_pred = self.predict_ctg_cst(x, [u0, u1])

                    # print(ctg_pred, cst_pred)

                    if ctg_pred < best_ctg and cst_pred <= 0:
                        best_u = [u0, u1]
                        best_ctg = ctg_pred
                    # else:
                    #     print("DID NOT TRIGGER")

            best_u = np.array(best_u).flatten()
            # Add some noise to encourage exploration
            best_u0_with_noise = best_u[0] + np.random.randint(low=-2, high=2, size=None)
            best_u1_with_noise = best_u[1] + np.random.randint(low=-2, high=2, size=None)
            best_u_with_noise = np.array((best_u0_with_noise, best_u1_with_noise)).flatten()
            print("Best u given x = {} is {}, adding noise = {}"
                  .format(x.flatten().round(4), best_u.round(4), best_u_with_noise.round(4))
                  )
            # Make sure the noise doesn't go over bounds
            if best_u_with_noise[0] > 373:
                best_u_with_noise[0] = 373
            elif best_u_with_noise[0] < 73:
                best_u_with_noise[0] = 73
            if best_u_with_noise[1] > 373:
                best_u_with_noise[1] = 373
            elif best_u_with_noise[1] < 73:
                best_u_with_noise[1] = 73

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
            bounds = optimize.Bounds(lb=np.array([173, 173], dtype=int), ub=np.array([373, 373], dtype=int))

            # Powell is chosen because it is the only gradientless method that can handle bounds
            # We need to it to gradientless because our input to output mapping is not differentiable
            min_kwargs = {
                "args": x,
                "method": 'Powell',
                "options": gd_options,
                "bounds": bounds
            }
            result = optimize.basinhopping(
                func=basinhopper_helper, x0=[273, 273], niter=10, minimizer_kwargs=min_kwargs
            )
            # result["x"] is the optimal u, don't be confused by the name!
            u_opt = np.array(result["x"]).flatten()
            # Add some noise to encourage exploration
            u0_opt_with_noise = u_opt[0] + np.random.uniform(low=-2, high=2, size=None)
            u1_opt_with_noise = u_opt[1] + np.random.uniform(low=-2, high=2, size=None)
            best_u_with_noise = np.array((u0_opt_with_noise, u1_opt_with_noise)).flatten()
            print("Best u given x = {} is {}, adding noise = {}"
                  .format(x.flatten().round(4), u_opt.round(4), best_u_with_noise.round(4))
                  )

            # Make sure the noise doesn't go over bounds
            if best_u_with_noise[0] > 373:
                best_u_with_noise[0] = 373
            elif best_u_with_noise[0] < 73:
                best_u_with_noise[0] = 73
            if best_u_with_noise[1] > 373:
                best_u_with_noise[1] = 373
            elif best_u_with_noise[1] < 73:
                best_u_with_noise[1] = 73
            best_u_with_noise[0] = int(best_u_with_noise[0])
            best_u_with_noise[1] = int(best_u_with_noise[1])

            return best_u_with_noise


def pickle_model(model, round_num):
    pickle_filename = results_folder + "R{}_".format(round_num+1) + "heatEq_nn_controller.pickle"
    with open(pickle_filename, "wb") as file:
        pickle.dump(model, file)
    print("Pickled model to " + pickle_filename)
    return pickle_filename


def train_and_pickle(round_num, trajectory_df_filename):
    print("Training with dataset: " + trajectory_df_filename)
    data = pd.read_csv(trajectory_df_filename)
    simple_nn = HeatEqNNController(x_dim=20, x_rom_dim=5, u_dim=2)
    simple_nn.fit(data)
    pickle_filename = pickle_model(simple_nn, round_num)
    return pickle_filename


def load_pickle(filename="heatEq_nn_controller_5dim.pickle"):
    with open(filename, "rb") as model:
        pickled_nn = pickle.load(model)
    print("Pickle loaded: " + filename)
    return pickled_nn


if __name__ == "__main__":
    data = pd.read_csv(data_folder + "heatEq_240_trajectories_rng.csv")
    heatEq_nn = HeatEqNNController(x_dim=20, x_rom_dim=5, u_dim=2)
    heatEq_nn.fit(data)

    # with open("heatEq_nn_controller_5dim.pickle", "wb") as pickle_file:
    #     pickle.dump(heatEq_nn, pickle_file)

    # heatEq_nn = load_pickle()
    # x = np.full(shape=(20, ), fill_value=273)
    # heatEq_nn.get_u_opt(x)

