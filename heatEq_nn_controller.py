from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
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

results_folder = "heatEq_replay_results/"


class xMOR(nn.Module):
    def __init__(self, x_dim, x_rom_dim=10):
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
        self.x_mor = xMOR(x_dim, x_rom_dim=10)
        self.net = Net(x_rom_dim, u_dim, hidden_size=12)

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

        x = np.array(x).transpose()
        u0 = u0.to_numpy(dtype=np.float32).reshape(-1, 1)
        u1 = u1.to_numpy(dtype=np.float32).reshape(-1, 1)
        ctg = ctg.to_numpy(dtype=np.float32).reshape(-1, 1)
        constraint = constraint.to_numpy(dtype=np.float32).reshape(-1, 1)

        u0_and_u1 = np.vstack((u0, u1))

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
        x = np.array(x, dtype=np.float32).flatten()
        u = np.array(u, dtype=np.float32).flatten()

        # u must be between [73, 473], so if basinhopper tries an invalid u, we penalize the ctg
        if not 73 <= u.any() <= 473:
            return np.inf, np.inf

        x_scaled = self.scaler_x.transform(x)
        u0_scaled = self.scaler_u.transform(u[0])
        u1_scaled = self.scaler_u.transform(u[1])

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

        return ctg_pred, cst_pred

    def get_u_opt(self, x, mode="grid"):
        x = np.array(x).flatten()

        if mode == "grid":
            best_u = [273, 273]
            best_ctg = np.inf

            for u0 in np.linspace(start=73, stop=473, num=200):
                for u1 in np.linspace(start=73, stop=473, num=200):
                    ctg_pred, cst_pred = self.predict_ctg_cst(x, [u0, u1])
                    if ctg_pred < best_ctg and cst_pred <= 0:
                        best_u = [u0, u1]
                        best_ctg = ctg_pred

            best_u = np.array(best_u).flatten()
            # Add some noise to encourage exploration
            best_u0_with_noise = best_u[0] + np.random.randint(low=-2, high=2, size=None)
            best_u1_with_noise = best_u[1] + np.random.randint(low=-2, high=2, size=None)
            best_u_with_noise = np.array((best_u0_with_noise, best_u1_with_noise)).flatten()
            print("Best u given x = {} is {}, adding noise = {}"
                  .format(x.flatten().round(4), best_u.round(4), best_u_with_noise.round(4))
                  )
            return best_u_with_noise

        elif mode == "basinhopper":
            def basinhopper_helper(u_bh, *arg_x_bh):
                x_bh = arg_x_bh[0]
                x_bh = np.array(x_bh).flatten()
                ctg_pred_bh, cst_pred_bh = self.predict_ctg_cst(x_bh, u_bh)
                # If constraints are broken, then we apply a 5x penalty to the cost to go
                # The greater the penalty, the more we drive a wedge between local minima for the gradient descent
                if cst_pred_bh > 0:
                    ctg_pred_bh = ctg_pred_bh * 5
                return ctg_pred_bh

            # Configure options for the gradient descent optimizer
            gd_options = {}
            # gd_options["maxiter"] = 1000
            gd_options["disp"] = True
            # gd_options["eps"] = 1

            # Nelder-mead is chosen because it is a gradientless method
            min_kwargs = {
                "args": x,
                "method": 'nelder-mead',
                "options": gd_options
            }
            result = optimize.basinhopping(
                func=basinhopper_helper, x0=[273, 273], minimizer_kwargs=min_kwargs
            )
            # result["x"] is the optimal u, don't be confused by the name!
            u_opt = np.array(result["x"]).flatten()
            return u_opt


def pickle_model(model, round_num):
    pickle_filename = results_folder + "R{}_".format(round_num+1) + "heatEq_nn_controller.pickle"
    with open(pickle_filename, "wb") as file:
        pickle.dump(model, file)
    print("Pickled model to " + pickle_filename)
    return pickle_filename


def train_and_pickle(round_num, trajectory_df_filename):
    print("Training with dataset: " + trajectory_df_filename)
    data = pd.read_csv(trajectory_df_filename)
    simple_nn = HeatEqNNController(x_dim=20, x_rom_dim=10, u_dim=2)
    simple_nn.fit(data)
    pickle_filename = pickle_model(simple_nn, round_num)
    return pickle_filename


if __name__ == "__main__":
    data = pd.read_csv("heatEq_240_trajectories_df.csv")
    heatEq_nn = HeatEqNNController(x_dim=20, x_rom_dim=10, u_dim=2)
    heatEq_nn.fit(data)

    with open("heatEq_nn_controller_240.pickle", "wb") as pickle_file:
        pickle.dump(heatEq_nn, pickle_file)
