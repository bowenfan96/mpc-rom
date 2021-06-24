import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchinfo import summary

import pickle
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

class Net(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Size of output is same as input
        self.fc2 = nn.Linear(hidden_size, input_size)

        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, data):
        # Input to hidden activation
        data = F.leaky_relu(self.fc1(data))
        # Hidden to output - no activation
        data = self.fc2(data)
        return data


class Autoencoder():
    def __init__(self, data, num_epoch=1000, config=None):
        super(Autoencoder, self).__init__()

        # Hyperparameters
        if config is not None:
            self.nb_epoch = config["num_epochs"]
            self.batch_size = config["batch_size"]
            self.learning_rate = config["learning_rate"]

            # Minimize hidden size (this is a hyperparameter?)
            self.hidden_size = config["hidden_size"]
            # We report loss on validation dataset to RayTune if we are in tuning mode
            self.is_tuning = True

        # If we are not tuning, then set the hyperparameters to the optimal ones we already found
        else:
            self.num_epoch = num_epoch
            self.batch_size = 6
            self.learning_rate = 0.049
            self.hidden_size = 2
            self.is_tuning = False

        # Initialise parameters
        processed_data = self.process_data(data)

        # Input size is the same as output size
        self.input_size = processed_data.shape[1]

        # Initialise neural network
        self.net = Net(self.input_size, self.hidden_size)

        return

    @staticmethod
    def process_data(data):
        # Convert to torch tensors
        data = torch.tensor(data.to_numpy(dtype=np.float32))
        return data

    def fit(self, data):
        # Process data, convert pandas to tensor
        data = self.process_data(data)

        # Set model to training model so gradients are updated
        self.net.train()

        # Wrap the tensors into a dataset, then load the data
        # data = torch.utils.data.TensorDataset(data)
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size)

        # Create optimizer to use update rules
        optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate)

        # Specify criterion used
        criterion = nn.MSELoss()

        # Train the neural network
        for epoch in range(self.num_epoch):

            # Full dataset gradient descent (for debugging only, poor accuracy)
            # output = self.net(data)
            # loss = criterion(output, data)
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # Minibatch gradient descent
            for minibatch_data in data_loader:
                output = self.net(minibatch_data)
                loss = criterion(output, minibatch_data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Test entire dataset at this epoch
            with torch.no_grad():
                output = self.net(data)
                loss = criterion(output, data)

            # Print loss
            print('The loss of epoch ' + str(epoch) + ' is ' + str(loss.item()))

        return self


def autoencoder_train():
    data = pd.read_csv("data.csv", sep=' ')

    autoencoder = Autoencoder(data, num_epoch=1000)

    autoencoder.fit(data)

    # Print a model.summary to show hidden layer information
    summary(autoencoder.net.to("cpu"), verbose=2)

# def autoencoder_test():


if __name__ == "__main__":
    autoencoder_train()
    # autoencoder_test()