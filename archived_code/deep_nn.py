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

matrices_folder = "matrices/"
results_folder = "results_csv/"
plots_folder = "results_plots/"


# One subnet with 2 hidden layers for each time step
class Net(nn.Module):
    def __init__(self, x_dim, u_dim):
        super(Net, self).__init__()
        self.num_nodes = x_dim + u_dim

        # self.hidden1_layer = nn.Linear(self.num_nodes, self.num_nodes*2)
        # self.hidden2_layer = nn.Linear(self.num_nodes*2, self.num_nodes*4)
        # self.hidden3_layer = nn.Linear(self.num_nodes*4, self.num_nodes*8)
        # self.hidden4_layer = nn.Linear(self.num_nodes*8, self.num_nodes*4)
        # self.hidden5_layer = nn.Linear(self.num_nodes*4, self.num_nodes*2)
        # self.output_layer = nn.Linear(self.num_nodes*2, self.num_nodes)


        self.hidden1_layer = nn.Linear(self.num_nodes, self.num_nodes*2)
        self.hidden2_layer = nn.Linear(self.num_nodes*2, self.num_nodes*4)
        self.hidden3_layer = nn.Linear(self.num_nodes*4, self.num_nodes*2)
        self.output_layer = nn.Linear(self.num_nodes*2, self.num_nodes)

        nn.init.kaiming_uniform_(self.hidden1_layer.weight)
        nn.init.kaiming_uniform_(self.hidden2_layer.weight)
        nn.init.kaiming_uniform_(self.hidden3_layer.weight)
        # nn.init.kaiming_uniform_(self.hidden4_layer.weight)
        # nn.init.kaiming_uniform_(self.hidden5_layer.weight)
        nn.init.kaiming_uniform_(self.output_layer.weight)

    def forward(self, xu_in):
        xu_h1 = F.leaky_relu(self.hidden1_layer(xu_in))
        xu_h2 = F.leaky_relu(self.hidden2_layer(xu_h1))
        xu_h3 = F.leaky_relu(self.hidden3_layer(xu_h2))
        # xu_h4 = F.leaky_relu(self.hidden4_layer(xu_h3))
        # xu_h5 = F.leaky_relu(self.hidden5_layer(xu_h2))
        xu_out = (self.output_layer(xu_h3))

        return xu_out




    # print(xi_tensor, ui_tensor, xf_tensor, uf_tensor)
    # time.sleep(2)
    # return xi_tensor, ui_tensor, xf_tensor, uf_tensor
    # return xi, ui, xf, uf

class NnCtrlSim:
    def __init__(self, x_dim, u_dim):
        self.net = Net(x_dim, u_dim)

    def process_data(self, data):
        x1i = data.filter(regex='xi_0')
        x2i = data.filter(regex='xi_1')
        ui = data.filter(regex='ui_0')
        x1i = x1i.to_numpy(dtype=np.float32)
        x2i = x2i.to_numpy(dtype=np.float32)
        ui = ui.to_numpy(dtype=np.float32)

        x1f = data.filter(regex='xf_0')
        x2f = data.filter(regex='xf_1')
        uf = data.filter(regex='uf_0')
        x1f = x1f.to_numpy(dtype=np.float32)
        x2f = x2f.to_numpy(dtype=np.float32)
        uf = uf.to_numpy(dtype=np.float32)

        self.scaler_x1 = preprocessing.MinMaxScaler()
        self.scaler_x2 = preprocessing.MinMaxScaler()
        self.scaler_u = preprocessing.MinMaxScaler()

        self.scaler_x1.fit(x1i)
        self.scaler_x2.fit(x2i)
        self.scaler_u.fit(ui)

        x1i = self.scaler_x1.transform(x1i)
        x1f = self.scaler_x1.transform(x1f)
        x2i = self.scaler_x2.transform(x2i)
        x2f = self.scaler_x2.transform(x2f)
        ui = self.scaler_u.transform(ui)
        uf = self.scaler_u.transform(uf)

        x1i_tensor = torch.tensor(x1i)
        x2i_tensor = torch.tensor(x2i)
        ui_tensor = torch.tensor(ui)
        x1f_tensor = torch.tensor(x1f)
        x2f_tensor = torch.tensor(x2f)
        uf_tensor = torch.tensor(uf)

        xu_i = torch.hstack((x1i_tensor, x2i_tensor, ui_tensor))
        xu_f = torch.hstack((x1f_tensor, x2f_tensor, uf_tensor))

        return xu_i, xu_f

    def fit(self, data):
        xu_i, xu_f = self.process_data(data)
        self.net.train()

        # Wrap the tensors into a dataset, then load the data
        data_mb = torch.utils.data.TensorDataset(xu_i, xu_f)
        data_loader = torch.utils.data.DataLoader(data_mb, batch_size=11)
        # Create optimizer to use update rules
        optimizer = optim.SGD(self.net.parameters(), lr=0.005)
        # Specify criterion used
        criterion = nn.MSELoss()
        criterion_x1 = nn.MSELoss()
        criterion_x2 = nn.MSELoss()
        criterion_u = nn.MSELoss()

        for epoch in range(1000):
            # optimizer.zero_grad()
            #
            # xu_out = self.unet(xi, ui)
            #
            # x1_pred = xu_out[:, 0]
            # x2_pred = xu_out[:, 1]
            # u_pred = xu_out[:, 2]
            #
            # x1_actl = xf[:, 0]
            # x2_actl = xf[:, 1]
            # u_actl = uf[:, 0]
            #
            # loss_x1 = criterion_x1(x1_pred, x1_actl)
            # loss_x2 = criterion_x2(x2_pred, x2_actl)
            # loss_u = criterion_u(u_pred, u_actl)
            #
            # loss = loss_x1 + loss_x2 + loss_u
            #
            # loss.backward()
            # optimizer.step()
            #
            # print("Epoch " + str(epoch) + ": Loss is " + str(loss))

            # Minibatch gradient descent
            # x, u and ctg are grouped in minibatches
            for xu_i, xu_f in data_loader:
                # print("Initial")
                # print(xu_i)
                # time.sleep(2)
                xu_out = self.net(xu_i)

                # print("Actual")
                # print(xu_f)
                # print("Predicted")
                # print(xu_out)
                # time.sleep(2)

                # x1_pred = xu_out[:, 0]
                # print(x1_pred)
                # time.sleep(2)
                # x2_pred = xu_out[:, 1]
                # u_pred = xu_out[:, 2]

                # print(xf_mb)

                # x1_actl = xf_mb[:, 0]
                # print(x1_actl)
                # time.sleep(5)

                # x2_actl = xf_mb[:, 1]
                # u_actl = uf_mb[:, 0]

                # print("Actual")
                # print(x1_actl)
                # print(x2_actl)
                # print(u_actl)
                # print("Predicted")
                # print(x1_pred)
                # print(x2_pred)
                # print(u_pred)
                # #
                # time.sleep(2)

                # loss_x1 = criterion_x1(x1_pred, x1_actl)
                # loss_x2 = criterion_x2(x2_pred, x2_actl)
                # loss_u = criterion_u(u_pred, u_actl)
                #
                # loss = loss_x1 + loss_x2 + loss_u

                # xu_actl = torch.hstack((xf_mb, uf_mb))
                loss = criterion(xu_f, xu_out)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


            # Test entire dataset at this epoch
            with torch.no_grad():
                xu_out = self.net(xu_i)

                # x1_pred = xu_out[:, 0]
                # x2_pred = xu_out[:, 1]
                # u_pred = xu_out[:, 2]

                # x1_actl = xf[:, 0]
                # x2_actl = xf[:, 1]
                # u_actl = uf[:, 0]

                # loss_x1 = criterion_x1(x1_pred, x1_actl)
                # loss_x2 = criterion_x2(x2_pred, x2_actl)
                # loss_u = criterion_u(u_pred, u_actl)

                # xu_actl = torch.hstack((xf, uf))
                loss = criterion(xu_f, xu_out)

                # loss = loss_x1 + loss_x2 + loss_u

                print("Epoch " + str(epoch) + ": Loss is " + str(loss))

    def predict(self, xu_i):
        x1i = xu_i[0]
        x2i = xu_i[1]
        ui = xu_i[2]

        x1i = self.scaler_x1.transform(x1i.reshape(1, -1))
        x2i = self.scaler_x2.transform(x2i.reshape(1, -1))
        ui = self.scaler_u.transform(ui.reshape(1, -1))

        x1i = torch.FloatTensor(x1i)
        x2i = torch.FloatTensor(x2i)
        ui = torch.FloatTensor(ui)
        xu_i = torch.hstack((x1i, x2i, ui))

        with torch.no_grad():
            xu_out = self.net(xu_i)

        print(xu_out)

        # xu_out = xu_out.numpy()

        x1f = xu_out[:, 0]
        x2f = xu_out[:, 1]
        uf = xu_out[:, 2]

        print(x1f, x2f)

        x1f = self.scaler_x1.inverse_transform(x1f.reshape(1, -1))
        x2f = self.scaler_x2.inverse_transform(x2f.reshape(1, -1))
        uf = self.scaler_u.inverse_transform(uf.reshape(1, -1))

        return np.hstack((x1f, x2f, uf))
        # return xu_out

def pickle_mor_nn(mor_nn_trained):
    """
    Pickle the trained neural unet
    :param mor_nn_trained: Trained model reduction neural unet
    :return: Save the pickled file
    """
    with open('deep_nn_simple.pickle', 'wb') as model:
        pickle.dump(mor_nn_trained, model)
    print("\nSaved model to mor_nn_simple.pickle\n")

if __name__ == "__main__":
    data = pd.read_csv(results_folder + "all_simple_reshaped.csv", sep=',')
    rom = NnCtrlSim(2, 1)
    rom.fit(data)
    # Print a model.summary to show hidden layer information
    summary(rom.net.to("cpu"), verbose=2)

    pickle_mor_nn(rom)