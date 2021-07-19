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
# Encoder: Full model - Intermediate - Reduced
# Decoder: Reduced - Intermediate - Full


class Encoder(nn.Module):
    def __init__(self, model_dim, reduced_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Linear(model_dim, (model_dim + reduced_dim) // 2)
        self.layer2 = nn.Linear((model_dim + reduced_dim) // 2, reduced_dim)

        nn.init.kaiming_uniform_(self.layer1.weight)
        nn.init.kaiming_uniform_(self.layer2.weight)

    def forward(self, x_in):
        # x_in = torch.flatten(data, start_dim=1)
        # Input to intermediate layer activation
        x_layer1 = F.leaky_relu(self.layer1(x_in))
        # Intermediate to compressed layer - no activation
        x_rom = self.layer2(x_layer1)
        return x_rom


class Decoder(nn.Module):
    def __init__(self, model_dim, reduced_dim):
        super(Decoder, self).__init__()
        self.layer3 = nn.Linear(reduced_dim, (model_dim + reduced_dim) // 2)
        self.layer4 = nn.Linear((model_dim + reduced_dim) // 2, model_dim)

        nn.init.kaiming_uniform_(self.layer3.weight)
        nn.init.kaiming_uniform_(self.layer4.weight)

    def forward(self, x_rom):
        x_layer3 = F.leaky_relu(self.layer3(x_rom))
        x_out = torch.sigmoid(self.layer4(x_layer3))
        return x_out


class Wrapper(nn.Module):
    def __init__(self, model_dim, reduced_dim):
        super(Wrapper, self).__init__()
        self.encoder = Encoder(model_dim, reduced_dim)
        self.decoder = Decoder(model_dim, reduced_dim)

    def forward(self, x_in):
        x_rom = self.encoder(x_in)
        x_out = self.decoder(x_rom)
        return x_out


class Autoencoder():
    def __init__(self, data, config=None):
        super(Autoencoder, self).__init__()

        # Hyperparameters
        if config is not None:
            self.nb_epoch = config["num_epochs"]
            self.batch_size = config["batch_size"]
            self.learning_rate = config["learning_rate"]

            # Desired dimension of reduced model
            self.reduced_dim_size = config["reduced_dim"]
            # We report loss on validation data to RayTune if we are in tuning mode
            self.is_tuning = True

        # If we are not tuning, then set the hyperparameters to the optimal ones we already found
        else:
            self.num_epoch = 200
            self.batch_size = 5
            self.learning_rate = 0.05
            self.reduced_dim_size = 2
            self.is_tuning = False

        # Initialise parameters
        processed_data = self.process_data(data)

        # Input size is the same as output size
        self.model_dim = processed_data.shape[1]

        print(self.model_dim)
        print(self.reduced_dim_size)

        # Initialise autoencoder neural net
        self.autoencoder = Wrapper(self.model_dim, self.reduced_dim_size)

        return

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
        self.autoencoder.train()

        # Wrap the tensors into a dataset, then load the data
        # data = torch.utils.data.TensorDataset(data)
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size)

        # Create optimizer to use update rules
        optimizer = optim.SGD(self.autoencoder.parameters(), lr=self.learning_rate)

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
                output = self.autoencoder(minibatch_data)
                loss = criterion(output, minibatch_data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Test entire dataset at this epoch
            with torch.no_grad():
                output = self.autoencoder(data)
                loss = criterion(output, data)

            # Print loss
            print('The loss of epoch ' + str(epoch) + ' is ' + str(loss.item()))

        return self


def autoencoder_train():
    data = pd.read_csv("df_export.csv", sep=',', usecols=["x_0", "x_1", "x_2", "x_3"])

    print(data)

    autoencoder = Autoencoder(data)
    autoencoder.fit(data)
    # Print a model.summary to show hidden layer information
    summary(autoencoder.autoencoder.to("cpu"), verbose=2)

# def autoencoder_test():


if __name__ == "__main__":
    autoencoder_train()
    # autoencoder_test()
