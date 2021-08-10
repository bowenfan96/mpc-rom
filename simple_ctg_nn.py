import time

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

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

matrices_folder = "matrices/"
results_folder = "results_csv/"
plots_folder = "results_plots/"


class uNet(nn.Module):
    def __init__(self, x_dim, u_dim):
        super(uNet, self).__init__()

        # self.hidden1_layer = nn.Linear(self.num_nodes, self.num_nodes*2)
        # self.hidden2_layer = nn.Linear(self.num_nodes*2, self.num_nodes*4)
        # self.hidden3_layer = nn.Linear(self.num_nodes*4, self.num_nodes*8)
        # self.hidden4_layer = nn.Linear(self.num_nodes*8, self.num_nodes*4)
        # self.hidden5_layer = nn.Linear(self.num_nodes*4, self.num_nodes*2)
        # self.output_layer = nn.Linear(self.num_nodes*2, self.num_nodes)

        self.hidden1_layer = nn.Linear((x_dim+u_dim), x_dim * 4)
        self.hidden2_layer = nn.Linear(x_dim * 4, x_dim * 4)
        self.hidden3_layer = nn.Linear(x_dim * 4, x_dim * 4)
        self.hidden4_layer = nn.Linear(x_dim*4, x_dim*4)
        self.hidden5_layer = nn.Linear(x_dim*4, x_dim*4)
        self.output_layer = nn.Linear(x_dim * 4, 1)

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
        x_out = (self.output_layer(xu_h5))

        return x_out


class NnCtrlSim:
    def __init__(self, x_dim, u_dim):
        self.unet = uNet(x_dim, u_dim)

    def process_data(self, data):
        xi1 = data.filter(regex='x1')
        xi2 = data.filter(regex='x2')
        ui = data.filter(regex='u')
        ctg = data.filter(regex='ctg')
        xi1 = xi1.to_numpy(dtype=np.float32)
        xi2 = xi2.to_numpy(dtype=np.float32)
        ui = ui.to_numpy(dtype=np.float32)
        ctg = ctg.to_numpy(dtype=np.float32)

        # xf1 = data.filter(regex='xf1')
        # xf2 = data.filter(regex='xf2')
        # uf = data.filter(regex='uf')
        # xf1 = xf1.to_numpy(dtype=np.float32)
        # xf2 = xf2.to_numpy(dtype=np.float32)
        # uf = uf.to_numpy(dtype=np.float32)

        self.scaler_x1 = preprocessing.MinMaxScaler()
        self.scaler_x2 = preprocessing.MinMaxScaler()
        self.scaler_u = preprocessing.MinMaxScaler()
        self.scaler_ctg = preprocessing.MinMaxScaler()

        self.scaler_x1.fit(xi1)
        self.scaler_x2.fit(xi2)
        self.scaler_u.fit(ui)
        self.scaler_ctg.fit(ctg)

        xi1 = self.scaler_x1.transform(xi1)
        xi2 = self.scaler_x2.transform(xi2)
        ui = self.scaler_u.transform(ui)
        ctg = self.scaler_ctg.transform(ctg)

        # xf1 = self.scaler_x1.transform(xf1)
        # xf2 = self.scaler_x2.transform(xf2)
        # uf = self.scaler_u.transform(uf)

        xi1_tensor = torch.tensor(xi1)
        xi2_tensor = torch.tensor(xi2)
        ui_tensor = torch.tensor(ui)
        ctg_tensor = torch.tensor(ctg)

        # xf1_tensor = torch.tensor(xf1)
        # xf2_tensor = torch.tensor(xf2)
        # uf_tensor = torch.tensor(uf)

        xi_tensor = torch.hstack((xi1_tensor, xi2_tensor))
        # xf_tensor = torch.hstack((xf1_tensor, xf2_tensor))

        # return xi_tensor, ui_tensor, xf_tensor, uf_tensor
        return xi_tensor, ui_tensor, ctg_tensor

    def fit(self, data):
        x, u, ctg = self.process_data(data)
        self.unet.train()

        # Wrap the tensors into a dataset, then load the data
        data_mb = torch.utils.data.TensorDataset(x, u, ctg)
        data_loader = torch.utils.data.DataLoader(data_mb, batch_size=10, shuffle=True)
        # Create optimizer to use update rules
        u_optimizer = optim.SGD(self.unet.parameters(), lr=0.005)
        # Specify criterion used
        u_criterion = nn.MSELoss()

        # x_optimizer = optim.SGD(self.xnet.parameters(), lr=0.01)
        # x_criterion = nn.MSELoss()

        for epoch in range(500):
            # Minibatch gradient descent
            for x_mb, u_mb, ctg_mb in data_loader:
                u_optimizer.zero_grad()
                ctg_pred = self.unet(x_mb, u_mb)
                loss = u_criterion(ctg_pred, ctg_mb)
                loss.backward()
                u_optimizer.step()

            # Test entire dataset at this epoch
            with torch.no_grad():
                ctg_pred = self.unet(x, u)
                loss = u_criterion(ctg_pred, ctg)

                print("Epoch " + str(epoch) + ": Loss is " + str(loss))

    def predict_u(self, x):
        x1 = np.array(x[0]).reshape(1, 1)
        x2 = np.array(x[1]).reshape(1, 1)

        x1 = self.scaler_x1.transform(x1)
        x2 = self.scaler_x2.transform(x2)

        x1 = torch.FloatTensor(x1)
        x2 = torch.FloatTensor(x2)
        x = torch.hstack((x1, x2))

        with torch.no_grad():
            u_out = self.unet(x)

        # print(u_out)

        u_pred = self.scaler_u.inverse_transform(u_out.reshape(1, -1))

        return u_pred

    def predict_x(self, x, u):
        x1 = np.array(x[0]).reshape(1, 1)
        x2 = np.array(x[1]).reshape(1, 1)

        x1 = self.scaler_x1.transform(x1)
        x2 = self.scaler_x2.transform(x2)
        u = self.scaler_u.transform(u[0])

        x1 = torch.FloatTensor(x1)
        x2 = torch.FloatTensor(x2)
        u = torch.FloatTensor(u)

        x = torch.hstack((x1, x2))

        with torch.no_grad():
            x_out = self.xnet(x, u)
        # print(x_out)
        x_out = torch.flatten(x_out)
        # print(x_out)
        x1_pred = self.scaler_x1.inverse_transform(np.array(x_out[0]).reshape(1, 1))
        x2_pred = self.scaler_x2.inverse_transform(np.array(x_out[1]).reshape(1, 1))

        x1_pred = x1_pred.flatten()
        x2_pred = x2_pred.flatten()

        x_pred = [x1_pred, x2_pred]
        return x_pred

    def predict_ctg(self, x_rom, u_rom):
        """
        This function is meant to be called externally by the controller
        :param x_rom:
        :param u_rom:
        :return:
        """
        print("Hi imma x_rom")
        print(x_rom)
        print("Hi imma u_rom")
        print(u_rom)

        x1 = x_rom[0][0]
        x2 = x_rom[0][1]

        if abs(u_rom) > 20:
            return np.inf

        x1_scaled = self.scaler_x1.transform([[x1]])
        x2_scaled = self.scaler_x2.transform([[x2]])
        u_scaled = self.scaler_u.transform([[u_rom]])

        x1_rom = torch.tensor(np.array(x1_scaled), dtype=torch.float)
        x2_rom = torch.tensor(np.array(x2_scaled), dtype=torch.float)
        u_rom = torch.tensor(np.array(u_scaled).reshape(1, -1), dtype=torch.float)

        x_rom_scaled = torch.hstack((x1_rom, x2_rom))

        with torch.no_grad():
            ctg_pred = self.unet(x_rom_scaled, u_rom)
        # Convert back to numpy array
        ctg_pred = ctg_pred.detach().numpy()
        # Scale back ctg
        ctg_pred = self.scaler_ctg.inverse_transform(ctg_pred)
        ctg_pred = ctg_pred.flatten()
        ctg_pred = float(ctg_pred)
        # print(ctg_pred)
        return ctg_pred


def pickle_mor_nn(mor_nn_trained):
    """
    Pickle the trained neural unet
    :param mor_nn_trained: Trained model reduction neural unet
    :return: Save the pickled file
    """
    with open('simple_proper_wctg.pickle', 'wb') as model:
        pickle.dump(mor_nn_trained, model)
    print("\nSaved model to simple_proper_wctg.pickle\n")


def predict():
    with open('deep_nn_simple.pickle', 'rb') as model:
        predict_model = pickle.load(model)
        # x = [
        #     [-4.269957139, -10.90194926],
        #     [-5.210666418, -7.961239966],
        #     [-5.871530207, -5.300376159],
        #     [-6.279179327, -2.892727025],
        #     [-6.457710338, -0.714196006],
        #     [-6.518915113, -0.513248059],
        #     [-6.567025239, -0.450008295],
        #     [-6.598891061, -0.191613998],
        #     [-6.599235865, 0.178549555],
        #     [-6.56544078, 0.49212674],
        #     [-6.525576775, 0.308217992]
        # ]

    x = [-4.269957139, -10.90194926]

    u = []

    for t in range(11):
        u_curr = predict_model.predict_u(x)
        u.append(u_curr)
        x = predict_model.predict_x(x, u)
        print(x)

    plt.plot(np.array(u).flatten())
    print(np.array(u).flatten())
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("simple_proper_rng_controls_init_fix_clean.csv", sep=',')
    rom = NnCtrlSim(2, 1)
    rom.fit(data)
    # Print a model.summary to show hidden layer information
    summary(rom.unet.to("cpu"), verbose=2)

    pickle_mor_nn(rom)
    # predict()

