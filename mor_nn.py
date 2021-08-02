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

# Neural net structure:
# x1, x2, x3, etc > i1, i2, etc
# u1, u2, u3, etc > j1, j2, etc
# i1, i2, etc > x_tilde1, x_tilde2, etc
# j1, j2, etc > u_tilde1, u_tilde2, etc
# x_t1, x_t2, etc, u_t1, u_t2, etc > k1, k2, etc
# k1, k2, etc > ctg


# Create the auxiliary neural networks
class Xnn(nn.Module):
    def __init__(self, x_dim, x_rom):
        super(Xnn, self).__init__()
        # Neural net structure: xi > ki > zi
        self.k_x = nn.Linear(x_dim, (x_dim + x_rom) // 2)
        self.z_x = nn.Linear((x_dim + x_rom) // 2, x_rom)

        nn.init.kaiming_uniform_(self.k_x.weight)
        nn.init.kaiming_uniform_(self.z_x.weight)

    def forward(self, x_in):
        # x_in = torch.flatten(data, start_dim=1)
        # Input to intermediate layer activation
        x_k = F.leaky_relu(self.k_x(x_in))
        # Intermediate to compressed layer
        x_z = F.leaky_relu(self.z_x(x_k))
        return x_z


class Unn(nn.Module):
    def __init__(self, u_dim, u_rom):
        super(Unn, self).__init__()
        self.k_u = nn.Linear(u_rom, (u_dim + u_rom) // 2)
        self.z_u = nn.Linear((u_dim + u_rom) // 2, u_dim)

        nn.init.kaiming_uniform_(self.k_u.weight)
        nn.init.kaiming_uniform_(self.z_u.weight)

    def forward(self, u_in):
        u_k = F.leaky_relu(self.k_u(u_in))
        u_z = F.leaky_relu(self.z_u(u_k))
        return u_z


class UnnDecoder(nn.Module):
    def __init__(self, u_dim, u_rom):
        super(UnnDecoder, self).__init__()
        self.o1_u = nn.Linear(u_dim, (u_dim + u_rom) // 2)
        self.o2_u = nn.Linear((u_dim + u_rom) // 2, u_dim)

        nn.init.kaiming_uniform_(self.o1_u.weight)
        nn.init.kaiming_uniform_(self.o2_u.weight)

    def forward(self, u_rom_in):
        u_out1 = F.leaky_relu(self.o1_u(u_rom_in))
        u_out2 = F.leaky_relu(self.o2_u(u_out1))
        return u_out2


class CtgNn(nn.Module):
    def __init__(self, x_rom, u_rom):
        super(CtgNn, self).__init__()
        self.xu1 = nn.Linear((x_rom + u_rom), ((x_rom + u_rom + 1) // 2))
        # Output is just 1 column - the cost to go value
        self.xu2 = nn.Linear(((x_rom + u_rom + 1) // 2), 1)

        nn.init.kaiming_uniform_(self.xu1.weight)
        nn.init.kaiming_uniform_(self.xu2.weight)

    def forward(self, x_rom, u_rom):
        # Predict cost to go
        xu_rom_in = torch.hstack((x_rom, u_rom))
        xu_rom_out = F.leaky_relu(self.xu1(xu_rom_in))
        ctg_pred = F.leaky_relu(self.xu2(xu_rom_out))
        return ctg_pred


class ObjNn(nn.Module):
    def __init__(self, x_dim, x_rom, u_dim, u_rom):
        super(ObjNn, self).__init__()
        self.x_mor = Xnn(x_dim, x_rom)
        self.u_mor = Unn(u_dim, u_rom)
        self.u_decoder = UnnDecoder(u_dim, u_rom)
        self.ctg = CtgNn(x_rom, u_rom)

    def forward(self, x_in, u_in):
        x_rom = self.x_mor(x_in)
        u_rom = self.u_mor(u_in)

        # Predict cost to go
        ctg_pred = self.ctg(x_rom, u_rom)
        # Decode compressed u
        u_decoded = self.u_decoder(u_rom)

        return ctg_pred, u_decoded


# Model Order Reducer
class MOR:
    def __init__(self, data, config=None):

        # Hyperparameters
        if config is not None:
            self.nb_epoch = config["num_epochs"]
            self.batch_size = config["batch_size"]
            self.learning_rate = config["learning_rate"]

            # Desired dimension of reduced model
            self.x_rom = config["x_rom"]
            # Desired dimension of reduced model
            self.u_rom = config["u_rom"]
            # We report loss on validation data to RayTune if we are in tuning mode
            self.is_tuning = True

        # If we are not tuning, then set the hyperparameters to the optimal ones we already found
        else:
            self.num_epoch = 100
            self.batch_size = 5
            self.learning_rate = 0.05
            # Desired dimension of reduced model
            self.x_rom = 2
            # Desired dimension of reduced model
            self.u_rom = 2
            self.is_tuning = False

        # Initialise parameters
        processed_data = self.process_data(data)

        # Input size is the same as output size
        self.x_dim = 4
        self.u_dim = 2

        # Initialise neural net
        self.model_reducer = ObjNn(self.x_dim, self.x_rom, self.u_dim, self.u_rom)

    @staticmethod
    def process_data(data):
        # Split x, u and cost to go columns
        x = data.filter(regex='x_')
        u = data.filter(regex='u_')
        # Cost to go
        ctg = data.filter(regex='ctg')

        # Convert from pandas to numpy arrays
        x = x.to_numpy(dtype=np.float32)
        u = u.to_numpy(dtype=np.float32)
        ctg = ctg.to_numpy(dtype=np.float32)

        # Scale all variables to between 0 and 1 (suitable for Relu)
        x_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(x)
        u_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(u)
        ctg_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(ctg)

        x_scaled_tensor = torch.tensor(x_scaled)
        u_scaled_tensor = torch.tensor(u_scaled)
        ctg_scaled_tensor = torch.tensor(ctg_scaled)

        return x_scaled_tensor, u_scaled_tensor, ctg_scaled_tensor

    def fit(self, data):
        # Process data, convert pandas to tensor
        x, u, ctg = self.process_data(data)

        # Set model to training model so gradients are updated
        self.model_reducer.train()

        # Wrap the tensors into a dataset, then load the data
        data = torch.utils.data.TensorDataset(x, u, ctg)
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size)

        # Create optimizer to use update rules
        optimizer = optim.SGD(self.model_reducer.parameters(), lr=self.learning_rate)

        # Specify criterion used
        criterion = nn.MSELoss()

        # Train the neural network
        for epoch in range(self.num_epoch):

            # Full dataset gradient descent
            # output = self.model_reducer(data)
            # loss = criterion(output, data)
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # Minibatch gradient descent
            # x, u and ctg are grouped in minibatches
            for x_mb, u_mb, ctg_mb in data_loader:
                # Model reducer takes x_in, u_in
                ctg_pred, u_decoded = self.model_reducer(x_mb, u_mb)

                # Loss is the loss of both ctg and decoded u
                loss_ctg = criterion(ctg_pred, ctg_mb)
                loss_u = criterion(u_decoded, u_mb)
                loss = loss_ctg + loss_u
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            # Test entire dataset at this epoch
            with torch.no_grad():
                ctg_pred, u_decoded = self.model_reducer(x, u)
                # Loss is the loss of both ctg and decoded u
                loss_ctg = criterion(ctg_pred, ctg)
                loss_u = criterion(u_decoded, u)
                loss = loss_ctg + loss_u

            # Print loss
            print('Epoch ' + str(epoch) + ': ctg- ' + str(loss_ctg.item()) + '| u- ' + str(loss_u.item()))

        return self

    def predict_ctg(self, x_rom, u_rom):
        """
        This function is meant to be called externally by the controller
        :param x_rom:
        :param u_rom:
        :return:
        """
        print(x_rom)
        print(u_rom)
        x_rom = torch.tensor(np.array(x_rom), dtype=torch.float)
        u_rom = torch.tensor(np.array(u_rom), dtype=torch.float)
        with torch.no_grad():
            ctg_pred = self.model_reducer.ctg(x_rom, u_rom)
        return ctg_pred

    def decode_u(self, u_rom):
        u_rom = torch.tensor(np.array(u_rom))
        with torch.no_grad():
            u_pred = self.model_reducer.u_decoder(u_rom)
        return u_pred

    def encode_x(self, x_full):
        x_full = torch.tensor(np.array(x_full))
        with torch.no_grad():
            x_rom = self.model_reducer.x_mor(x_full)
        return x_rom


def train():
    data = pd.read_csv(results_folder + "mpc_x_u_ctg.csv", sep=','
                       # , usecols=["x_0", "x_1", "x_2", "x_3"]
                       )
    print(data)
    rom = MOR(data)
    rom.fit(data)
    # Print a model.summary to show hidden layer information
    summary(rom.model_reducer.to("cpu"), verbose=2)

    pickle_mor_nn(rom)


def pickle_mor_nn(mor_nn_trained):
    """
    Pickle the trained neural net
    :param mor_nn_trained: Trained model reduction neural net
    :return: Save the pickled file
    """
    with open('mor_nn.pickle', 'wb') as model:
        pickle.dump(mor_nn_trained, model)
    print("\nSaved model to mor_nn.pickle\n")


if __name__ == "__main__":
    train()
