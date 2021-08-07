import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchinfo import summary

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

matrices_folder = "matrices/simple/"
results_folder = "results_csv/simple/"
plots_folder = "results_plots/simple/"


# One subnet with 2 hidden layers for each time step
class Net(nn.Module):
    def __init__(self, x_dim, u_dim):
        super(Net, self).__init__()
        self.num_nodes = x_dim + u_dim

        self.hidden1_layer = nn.Linear(self.num_nodes, self.num_nodes*2)
        self.hidden2_layer = nn.Linear(self.num_nodes*2, self.num_nodes*4)
        self.hidden3_layer = nn.Linear(self.num_nodes*4, self.num_nodes*8)
        self.hidden4_layer = nn.Linear(self.num_nodes*8, self.num_nodes*4)
        self.hidden5_layer = nn.Linear(self.num_nodes*4, self.num_nodes*2)
        self.output_layer = nn.Linear(self.num_nodes*2, self.num_nodes)

        nn.init.kaiming_uniform_(self.hidden1_layer.weight)
        nn.init.kaiming_uniform_(self.hidden2_layer.weight)
        nn.init.kaiming_uniform_(self.hidden3_layer.weight)
        nn.init.kaiming_uniform_(self.hidden4_layer.weight)
        nn.init.kaiming_uniform_(self.hidden5_layer.weight)
        nn.init.kaiming_uniform_(self.output_layer.weight)

    def forward(self, x_in, u_in):
        xu_in = torch.hstack((x_in, u_in))
        xu_h1 = F.leaky_relu(self.hidden1_layer(xu_in))
        xu_h2 = F.leaky_relu(self.hidden2_layer(xu_h1))
        xu_h3 = F.leaky_relu(self.hidden3_layer(xu_h2))
        xu_h4 = F.leaky_relu(self.hidden4_layer(xu_h3))
        xu_h5 = F.leaky_relu(self.hidden5_layer(xu_h4))
        xu_out = F.leaky_relu(self.output_layer(xu_h5))

        return xu_out


def process_data(data):
    xi = data.filter(regex='xi_')
    ui = data.filter(regex='ui_')
    xi = xi.to_numpy(dtype=np.float32)
    ui = ui.to_numpy(dtype=np.float32)
    xf = data.filter(regex='xf_')
    uf = data.filter(regex='uf_')
    xf = xf.to_numpy(dtype=np.float32)
    uf = uf.to_numpy(dtype=np.float32)
    # scaler_x = preprocessing.MinMaxScaler()
    # scaler_u = preprocessing.MinMaxScaler()
    # xi = scaler_x.fit_transform(xi)
    # xf = scaler_x.fit_transform(xf)
    # ui = scaler_u.fit_transform(ui)
    # uf = scaler_u.fit_transform(uf)
    xi_tensor = torch.tensor(xi)
    ui_tensor = torch.tensor(ui)
    xf_tensor = torch.tensor(xf)
    uf_tensor = torch.tensor(uf)
    # print(xi_tensor, ui_tensor, xf_tensor, uf_tensor)
    # time.sleep(2)
    return xi_tensor, ui_tensor, xf_tensor, uf_tensor
    # return xi, ui, xf, uf

class NnCtrlSim:
    def __init__(self, x_dim, u_dim):
        self.net = Net(x_dim, u_dim)

    def fit(self, data):
        xi, ui, xf, uf = process_data(data)
        self.net.train()

        # Wrap the tensors into a dataset, then load the data
        data_mb = torch.utils.data.TensorDataset(xi, ui, xf, uf)
        data_loader = torch.utils.data.DataLoader(data_mb, batch_size=50)
        # Create optimizer to use update rules
        optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        # Specify criterion used
        criterion = nn.MSELoss()

        for epoch in range(1000):

            # xu_pred = self.net(xi, ui)
            # xuf = torch.hstack((xf, uf))
            # loss = criterion(xu_pred, xuf)
            # print("Epoch " + str(epoch) + ": Loss is " + str(loss))
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # Minibatch gradient descent
            # x, u and ctg are grouped in minibatches
            for xi_mb, ui_mb, xf_mb, uf_mb in data_loader:
                optimizer.zero_grad()
                xu_out = self.net(xi_mb, ui_mb)

                x1_pred = xu_out[:, 0]
                x2_pred = xu_out[:, 1]
                u_pred = xu_out[:, 2]

                x1_actl = xf_mb[:, 0]
                x2_actl = xf_mb[:, 1]
                u_actl = uf_mb[0]

                print(x1_pred)
                print(x1_actl)

                time.sleep(5)

                loss_x1 = criterion(x1_pred, x1_actl)
                loss_x2 = criterion(x2_pred, x2_actl)
                loss_u = criterion(u_pred, u_actl)

                loss = loss_x1 + loss_x2 + loss_u

                loss.backward()
                optimizer.step()

            # Test entire dataset at this epoch
            with torch.no_grad():
                x1_pred, x2_pred, u_pred = self.net(xi, ui)

                x1_actl = xi[0]
                x2_actl = xi[1]
                u_actl = ui[0]

                loss_x1 = criterion(x1_pred, x1_actl)
                loss_x2 = criterion(x2_pred, x2_actl)
                loss_u = criterion(u_pred, u_actl)

                loss = loss_x1 + loss_x2 + loss_u

                print("Epoch " + str(epoch) + ": Loss is " + str(loss))


if __name__ == "__main__":
    data = pd.read_csv(results_folder + "all_simple_reshaped.csv", sep=',')
    rom = NnCtrlSim(2, 1)
    rom.fit(data)
    # Print a model.summary to show hidden layer information
    summary(rom.net.to("cpu"), verbose=2)