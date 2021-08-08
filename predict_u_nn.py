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

        self.hidden1_layer = nn.Linear(x_dim, x_dim*4)
        self.hidden2_layer = nn.Linear(x_dim*4, x_dim*4)
        self.hidden3_layer = nn.Linear(x_dim*4, x_dim*4)
        # self.hidden4_layer = nn.Linear(x_dim*8, x_dim*4)
        # self.hidden5_layer = nn.Linear(x_dim*4, x_dim*2)
        self.output_layer = nn.Linear(x_dim*4, u_dim)

        nn.init.kaiming_uniform_(self.hidden1_layer.weight)
        nn.init.kaiming_uniform_(self.hidden2_layer.weight)
        nn.init.kaiming_uniform_(self.hidden3_layer.weight)
        # nn.init.kaiming_uniform_(self.hidden4_layer.weight)
        # nn.init.kaiming_uniform_(self.hidden5_layer.weight)
        nn.init.kaiming_uniform_(self.output_layer.weight)

    def forward(self, x_in):
        xu_h1 = F.leaky_relu(self.hidden1_layer(x_in))
        xu_h2 = F.leaky_relu(self.hidden2_layer(xu_h1))
        xu_h3 = F.leaky_relu(self.hidden3_layer(xu_h2))
        # xu_h4 = F.leaky_relu(self.hidden4_layer(xu_h3))
        # xu_h5 = F.leaky_relu(self.hidden5_layer(xu_h4))
        u_out = (self.output_layer(xu_h3))

        return u_out


class xNet(nn.Module):
    def __init__(self, x_dim, u_dim):
        super(xNet, self).__init__()

        # self.hidden1_layer = nn.Linear(self.num_nodes, self.num_nodes*2)
        # self.hidden2_layer = nn.Linear(self.num_nodes*2, self.num_nodes*4)
        # self.hidden3_layer = nn.Linear(self.num_nodes*4, self.num_nodes*8)
        # self.hidden4_layer = nn.Linear(self.num_nodes*8, self.num_nodes*4)
        # self.hidden5_layer = nn.Linear(self.num_nodes*4, self.num_nodes*2)
        # self.output_layer = nn.Linear(self.num_nodes*2, self.num_nodes)

        self.hidden1_layer = nn.Linear((x_dim+u_dim), x_dim * 4)
        self.hidden2_layer = nn.Linear(x_dim * 4, x_dim * 4)
        self.hidden3_layer = nn.Linear(x_dim * 4, x_dim * 4)
        # self.hidden4_layer = nn.Linear(x_dim*8, x_dim*4)
        # self.hidden5_layer = nn.Linear(x_dim*4, x_dim*2)
        self.output_layer = nn.Linear(x_dim * 4, x_dim)

        nn.init.kaiming_uniform_(self.hidden1_layer.weight)
        nn.init.kaiming_uniform_(self.hidden2_layer.weight)
        nn.init.kaiming_uniform_(self.hidden3_layer.weight)
        # nn.init.kaiming_uniform_(self.hidden4_layer.weight)
        # nn.init.kaiming_uniform_(self.hidden5_layer.weight)
        nn.init.kaiming_uniform_(self.output_layer.weight)

    def forward(self, x_in, u_in):
        xu_in = torch.hstack((x_in, u_in))
        xu_h1 = F.leaky_relu(self.hidden1_layer(xu_in))
        xu_h2 = F.leaky_relu(self.hidden2_layer(xu_h1))
        xu_h3 = F.leaky_relu(self.hidden3_layer(xu_h2))
        # xu_h4 = F.leaky_relu(self.hidden4_layer(xu_h3))
        # xu_h5 = F.leaky_relu(self.hidden5_layer(xu_h4))
        x_out = (self.output_layer(xu_h3))

        return x_out


class NnCtrlSim:
    def __init__(self, x_dim, u_dim):
        self.unet = uNet(x_dim, u_dim)
        self.xnet = xNet(x_dim, u_dim)

    def process_data(self, data):
        x1 = data.filter(regex='x1')
        x2 = data.filter(regex='x2')
        u = data.filter(regex='u')
        x1 = x1.to_numpy(dtype=np.float32)
        x2 = x2.to_numpy(dtype=np.float32)
        u = u.to_numpy(dtype=np.float32)

        self.scaler_x1 = preprocessing.MinMaxScaler()
        self.scaler_x2 = preprocessing.MinMaxScaler()
        self.scaler_u = preprocessing.MinMaxScaler()

        self.scaler_x1.fit(x1)
        self.scaler_x2.fit(x2)
        self.scaler_u.fit(u)

        x1 = self.scaler_x1.transform(x1)
        x2 = self.scaler_x2.transform(x2)
        u = self.scaler_u.transform(u)

        x1_tensor = torch.tensor(x1)
        x2_tensor = torch.tensor(x2)
        u_tensor = torch.tensor(u)

        x_tensor = torch.hstack((x1_tensor, x2_tensor))

        return x_tensor, u_tensor

    def fit(self, data):
        x, u = self.process_data(data)
        self.unet.train()
        self.xnet.train()

        # Wrap the tensors into a dataset, then load the data
        data_mb = torch.utils.data.TensorDataset(x, u)
        data_loader = torch.utils.data.DataLoader(data_mb, batch_size=11)
        # Create optimizer to use update rules
        u_optimizer = optim.SGD(self.unet.parameters(), lr=0.01)
        # Specify criterion used
        u_criterion = nn.MSELoss()

        x_optimizer = optim.SGD(self.xnet.parameters(), lr=0.01)
        x_criterion = nn.MSELoss()

        for epoch in range(500):
            # Minibatch gradient descent
            for x_mb, u_mb in data_loader:
                u_optimizer.zero_grad()
                x_optimizer.zero_grad()
                u_pred = self.unet(x_mb)
                x_pred = self.xnet(x_mb, u_mb)

                uloss = u_criterion(u_pred, u_mb)
                xloss = x_criterion(x_pred, x_mb)

                uloss.backward()
                xloss.backward()
                u_optimizer.step()
                x_optimizer.step()

            # Test entire dataset at this epoch
            with torch.no_grad():
                u_pred = self.unet(x)
                x_pred = self.xnet(x, u)

                uloss = u_criterion(u_pred, u)
                xloss = x_criterion(x_pred, x)

                print("Epoch " + str(epoch) + ": Loss is " + str(uloss), str(xloss))

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


def pickle_mor_nn(mor_nn_trained):
    """
    Pickle the trained neural unet
    :param mor_nn_trained: Trained model reduction neural unet
    :return: Save the pickled file
    """
    with open('deep_nn_simple.pickle', 'wb') as model:
        pickle.dump(mor_nn_trained, model)
    print("\nSaved model to mor_nn_simple.pickle\n")


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

    plt.plot(np.array(u).flatten())
    print(np.array(u).flatten())
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("simple_proper.csv", sep=',')
    rom = NnCtrlSim(2, 1)
    rom.fit(data)
    # Print a model.summary to show hidden layer information
    summary(rom.unet.to("cpu"), verbose=2)

    pickle_mor_nn(rom)
    # predict()

