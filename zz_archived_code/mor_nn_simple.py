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

# Neural unet structure:
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
        # Neural unet structure: xi > ki > zi
        self.x_input_layer = nn.Linear(x_dim, (x_dim + x_rom) // 2)
        self.x_hidden_layer = nn.Linear((x_dim + x_rom) // 2, (x_dim + x_rom) // 2)
        self.x_rom_layer = nn.Linear((x_dim + x_rom) // 2, x_rom)

        nn.init.kaiming_uniform_(self.x_input_layer.weight)
        nn.init.kaiming_uniform_(self.x_hidden_layer.weight)
        nn.init.kaiming_uniform_(self.x_rom_layer.weight)

    def forward(self, x_in):
        # x_in = torch.flatten(data, start_dim=1)
        # Hidden 1 activation
        x_h1 = F.leaky_relu(self.x_input_layer(x_in))
        # Hidden 2 activation
        x_h2 = F.leaky_relu(self.x_hidden_layer(x_h1))
        # Output - No activation
        x_rom = self.x_rom_layer(x_h2)
        return x_rom


class Unn(nn.Module):
    def __init__(self, u_dim, u_rom):
        super(Unn, self).__init__()
        self.u_input_layer = nn.Linear(u_rom, (u_dim + u_rom) // 2)
        self.u_hidden_layer = nn.Linear((u_dim + u_rom) // 2, (u_dim + u_rom) // 2)
        self.u_rom_layer = nn.Linear((u_dim + u_rom) // 2, u_dim)

        nn.init.kaiming_uniform_(self.u_input_layer.weight)
        nn.init.kaiming_uniform_(self.u_hidden_layer.weight)
        nn.init.kaiming_uniform_(self.u_rom_layer.weight)

    def forward(self, u_in):
        # Hidden 1 activation
        u_h1 = F.leaky_relu(self.u_input_layer(u_in))
        # Hidden 2 activation
        u_h2 = F.leaky_relu(self.u_hidden_layer(u_h1))
        # Output - No activation
        u_rom = self.u_rom_layer(u_h2)
        return u_rom


class UnnDecoder(nn.Module):
    def __init__(self, u_dim, u_rom):
        super(UnnDecoder, self).__init__()
        self.o1_u = nn.Linear(u_dim, (u_dim + u_rom) // 2)
        self.o2_u = nn.Linear((u_dim + u_rom) // 2, (u_dim + u_rom) // 2)
        self.o3_u = nn.Linear((u_dim + u_rom) // 2, u_dim)

        nn.init.kaiming_uniform_(self.o1_u.weight)
        nn.init.kaiming_uniform_(self.o2_u.weight)
        nn.init.kaiming_uniform_(self.o3_u.weight)

    def forward(self, u_rom_in):
        u_out1 = F.leaky_relu(self.o1_u(u_rom_in))
        u_out2 = F.leaky_relu(self.o2_u(u_out1))
        u_out3 = self.o3_u(u_out2)
        # Return decoded u
        return u_out3


class CtgNn(nn.Module):
    def __init__(self, x_rom, u_rom):
        super(CtgNn, self).__init__()
        self.ctg1 = nn.Linear((x_rom + u_rom), ((x_rom + u_rom + 1) * 2))
        self.ctg2 = nn.Linear((x_rom + u_rom + 1) * 2, (x_rom + u_rom + 1) * 2)
        # Output is just 1 column - the cost to go value
        self.ctg3 = nn.Linear((x_rom + u_rom + 1) * 2, (x_rom + u_rom + 1) * 2)
        self.ctg4 = nn.Linear((x_rom + u_rom + 1) * 2, (x_rom + u_rom + 1) * 2)
        self.ctg5 = nn.Linear((x_rom + u_rom + 1) * 2, 1)

        nn.init.kaiming_uniform_(self.ctg1.weight)
        nn.init.kaiming_uniform_(self.ctg2.weight)
        nn.init.kaiming_uniform_(self.ctg3.weight)
        nn.init.kaiming_uniform_(self.ctg4.weight)
        nn.init.kaiming_uniform_(self.ctg5.weight)

    def forward(self, x_rom, u_rom):
        # Predict cost to go
        xu_rom_in = torch.hstack((x_rom, u_rom))
        xu_rom_out1 = F.leaky_relu(self.ctg1(xu_rom_in))
        xu_rom_out2 = F.leaky_relu(self.ctg2(xu_rom_out1))
        xu_rom_out3 = F.leaky_relu(self.ctg3(xu_rom_out2))
        xu_rom_out4 = F.leaky_relu(self.ctg4(xu_rom_out3))
        ctg_pred = F.relu(self.ctg5(xu_rom_out4))
        return ctg_pred


class MorNn(nn.Module):
    def __init__(self, x_dim, x_rom, u_dim, u_rom, activate_u_nn=False):
        super(MorNn, self).__init__()
        self.ctg = CtgNn(x_rom, u_rom)

        if activate_u_nn:
            self.u_mor = Unn(u_dim, u_rom)
            self.u_decoder = UnnDecoder(u_dim, u_rom)

        self.u_nn_activated = activate_u_nn

    def forward(self, x_in, u_in):
        if self.u_nn_activated:
            x_rom = self.x_mor(x_in)
            u_rom = self.u_mor(u_in)

            # Predict cost to go
            ctg_pred = self.ctg(x_rom, u_rom)
            # Decode compressed u
            u_decoded = self.u_decoder(u_rom)
            return ctg_pred, u_decoded
        else:
            x_rom = self.x_mor(x_in)
            # Predict cost to go
            ctg_pred = self.ctg(x_rom, u_in)
            return ctg_pred


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
            # To encode u or not (not needed if we only have a few controllers)
            self.u_nn_activated = config["encode_u"]
            # We report loss on validation data to RayTune if we are in tuning mode
            self.is_tuning = True

        # If we are not tuning, then set the hyperparameters to the optimal ones we already found
        else:
            self.num_epoch = 500
            self.batch_size = 6
            self.learning_rate = 0.005
            # Desired dimension of reduced model
            self.x_rom = 2
            # Desired dimension of reduced model
            self.u_rom = 1
            self.u_nn_activated = False
            self.is_tuning = False

        # Initialise parameters
        processed_data = self.process_data(data)

        # Input size is the same as output size
        self.x_dim = 2
        self.u_dim = 1

        # Initialise neural unet
        self.model_reducer = CtgNn(2, 1)

    def process_data(self, data):
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
        self.x_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.x_scaler.fit(x)
        x_scaled = self.x_scaler.transform(x)
        self.u_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.u_scaler.fit(u)
        u_scaled = self.u_scaler.transform(u)
        self.ctg_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.ctg_scaler.fit(ctg)
        ctg_scaled = self.ctg_scaler.transform(ctg)

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

        if self.u_nn_activated:
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
                    optimizer.zero_grad()
                    # Model reducer takes x_in, u_in
                    ctg_pred = self.model_reducer(x_mb, u_mb)

                    # Loss is the loss of both ctg and decoded u
                    loss = criterion(ctg_pred, ctg_mb)
                    # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
                    loss.backward()
                    optimizer.step()

                # Test entire dataset at this epoch
                with torch.no_grad():
                    ctg_pred = self.model_reducer(x, u)
                    # Loss is the loss of both ctg and decoded u
                    loss = criterion(ctg_pred, ctg)

                # Print loss
                print('Epoch ' + str(epoch) + ': ctg- ' + str(loss.item()))

        else:
            # Train the neural network
            for epoch in range(self.num_epoch):
                # Minibatch gradient descent
                # x, u and ctg are grouped in minibatches
                for x_mb, u_mb, ctg_mb in data_loader:
                    optimizer.zero_grad()
                    # Model reducer takes x_in, u_in
                    ctg_pred = self.model_reducer(x_mb, u_mb)

                    # Loss is the loss of both ctg and decoded u
                    loss = criterion(ctg_pred, ctg_mb)
                    # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
                    loss.backward()
                    optimizer.step()

                # Test entire dataset at this epoch
                with torch.no_grad():
                    ctg_pred = self.model_reducer(x, u)
                    # Loss is the loss of both ctg and decoded u
                    loss = criterion(ctg_pred, ctg)

                # Print loss
                print('Epoch ' + str(epoch) + ': ctg- ' + str(loss.item()))

        return self

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
        x_rom = torch.tensor(np.array(x_rom), dtype=torch.float)
        u_rom = torch.tensor(np.array(u_rom).reshape(1, -1), dtype=torch.float)
        with torch.no_grad():
            ctg_pred = self.model_reducer.ctg(x_rom, u_rom)
        # Convert back to numpy array
        ctg_pred = ctg_pred.detach().numpy()
        # Scale back ctg
        ctg_pred = self.ctg_scaler.inverse_transform(ctg_pred.reshape(1, -1))
        ctg_pred = ctg_pred.flatten()
        ctg_pred = float(ctg_pred)
        return ctg_pred

    def decode_u(self, u_rom):
        u_rom = torch.tensor(np.array(u_rom), dtype=torch.float)
        print("hi im tensor urom")
        print(u_rom)
        with torch.no_grad():
            u_pred = self.model_reducer.u_decoder(u_rom)
        # Convert back to numpy array
        u_pred = u_pred.detach().numpy()
        print("hi imma upred")
        print(u_pred)
        # Scale back u
        u_pred = self.u_scaler.inverse_transform(u_pred.reshape(1, -1))
        return u_pred

    def encode_x(self, x_full):
        # Scale x
        x_full = np.array(x_full)
        x_full = self.x_scaler.transform(x_full)
        x_full = torch.tensor(x_full, dtype=torch.float)
        with torch.no_grad():
            x_rom = self.model_reducer.x_mor(x_full)
        # Convert back to numpy array
        x_rom = x_rom.detach().numpy()
        return x_rom

    def encode_u(self, u_full):
        # Scale u
        u_full = np.array(u_full).reshape(1, -1)
        u_full = self.u_scaler.transform(u_full)
        u_full = torch.tensor(u_full, dtype=torch.float)
        with torch.no_grad():
            u_rom = self.model_reducer.u_mor(u_full)
        # Convert back to numpy array
        u_rom = u_rom.detach().numpy()
        return u_rom

    def predict(self, x_full, u_full):
        x_full = np.array(x_full).reshape(1, -1)
        u_full = np.array(u_full).reshape(1, -1)
        x_full = self.x_scaler.transform(x_full)
        u_full = self.u_scaler.transform(u_full)
        x_full = torch.tensor(x_full, dtype=torch.float)
        u_full = torch.tensor(u_full, dtype=torch.float)
        with torch.no_grad():
            ctg_pred = self.model_reducer(x_full, u_full)
        print(ctg_pred)
        ctg_pred = self.ctg_scaler.inverse_transform(ctg_pred)
        print(ctg_pred)
        return ctg_pred.flatten()


def train():
    data = pd.read_csv(results_folder + "all_simple.csv", sep=','
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
    Pickle the trained neural unet
    :param mor_nn_trained: Trained model reduction neural unet
    :return: Save the pickled file
    """
    with open('mor_nn_simple.pickle', 'wb') as model:
        pickle.dump(mor_nn_trained, model)
    print("\nSaved model to mor_nn_simple.pickle\n")


if __name__ == "__main__":
    train()
