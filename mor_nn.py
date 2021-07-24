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

# Neural net structure:
# x1, x2, x3, etc > k1, k2, etc
# u1, u2, u3, etc > j1, j2, etc
# k1, k2, etc > z1
# j1, j2, etc > z2
# z1, z2 > obj


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


class Wrapper(nn.Module):
    def __init__(self, x_dim, x_rom, u_dim, u_rom):
        super(Wrapper, self).__init__()
        self.x_mor = Xnn(x_dim, x_rom)
        self.u_mor = Unn(u_dim, u_rom)

    def forward(self, x_in):
        x_rom = self.encoder(x_in)
        x_out = self.decoder(x_rom)
        return x_out


class MOR():
    def __init__(self, data, config=None):
        super(MOR, self).__init__()

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
            self.num_epoch = 200
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
        self.x_dim = processed_data.shape[1]
        self.u_dim = processed_data.shape[1]

        # Initialise neural net
        self.model_reducer = Wrapper(self.x_dim, self.x_rom, self.u_dim, self.u_rom)

    @staticmethod
    def process_data(data):
        # Convert to torch tensors
        data = data.to_numpy(dtype=np.float32)
        data_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        data = torch.tensor(data_scaled)
        return data

    def fit(self, data):
        # Process data, convert pandas to tensor
        data = self.process_data(data)

        # Set model to training model so gradients are updated
        self.model_reducer.train()

        # Wrap the tensors into a dataset, then load the data
        # data = torch.utils.data.TensorDataset(data)
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size)

        # Create optimizer to use update rules
        optimizer = optim.SGD(self.model_reducer.parameters(), lr=self.learning_rate)

        # Specify criterion used
        criterion = nn.MSELoss()

        # Train the neural network
        for epoch in range(self.num_epoch):

            # Full dataset gradient descent
            # output = self.autoencoder(data)
            # loss = criterion(output, data)
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # Minibatch gradient descent
            for minibatch_data in data_loader:
                output = self.model_reducer(minibatch_data)
                loss = criterion(output, minibatch_data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Test entire dataset at this epoch
            with torch.no_grad():
                output = self.model_reducer(data)
                loss = criterion(output, data)

            # Print loss
            print('The loss of epoch ' + str(epoch) + ' is ' + str(loss.item()))

        return self


def autoencoder_train():
    data = pd.read_csv("df_export.csv", sep=',', usecols=["x_0", "x_1", "x_2", "x_3"])

    print(data)

    rom = MOR(data)
    rom.fit(data)
    # Print a model.summary to show hidden layer information
    summary(rom.model_reducer.to("cpu"), verbose=2)

# def autoencoder_test():


if __name__ == "__main__":
    autoencoder_train()
    # autoencoder_test()