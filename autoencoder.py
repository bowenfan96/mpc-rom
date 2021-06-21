import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Autoencoder():
    def __init__(self, x, num_epoch=1000, config=None):
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
            self.hidden_size = 17
            self.is_tuning = False

        # Initialise parameters
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]

        # Initialise neural network
        self.net = Net(self.input_size, self.hidden_size)

        return

    def fit(self, x):
        # Set model to training model so gradients are updated
        self.net.train()

        # Wrap the tensors into a dataset, then load the data
        data = torch.utils.data.TensorDataset(x)
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size)

        # Create optimizer to use update rules
        optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate)

        # Specify criterion used
        criterion = nn.MSELoss()

        # Train the neural network
        for epoch in range(self.nb_epoch):
            for x_train_mb, y_train_mb in data_loader:
                y_pred = self.net(x_train_mb)
                loss = criterion(y_pred, y_train_mb)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                predicted = self.net(x)
                loss = criterion(predicted, x)

            # Print loss
            print('The loss of epoch ' + str(epoch) + ' is ' + str(loss.item()))

        return self


def autoencoder_train():
    data = pd.read_csv("data.csv")

    autoencoder = Autoencoder(data, num_epoch=100)
    autoencoder.fit(data)

def autoencoder_test():


if __name__ == "__main__":
    autoencoder_train()
    autoencoder_test()