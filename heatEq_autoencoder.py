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
import pysindy


class Encoder(nn.Module):
    def __init__(self, x_dim, x_rom_dim):
        super(Encoder, self).__init__()
        h1_nodes = min(x_rom_dim, (x_dim + x_rom_dim) // 2)
        h2_nodes = min(x_rom_dim, (x_dim + x_rom_dim) // 4)
        h3_nodes = min(x_rom_dim, (x_dim + x_rom_dim) // 4)

        self.input = nn.Linear(x_dim, h1_nodes)
        self.h1 = nn.Linear(h1_nodes, h2_nodes)
        self.h2 = nn.Linear(h2_nodes, h3_nodes)
        self.h3 = nn.Linear(h3_nodes, x_rom_dim)

        nn.init.kaiming_uniform_(self.input.weight)
        nn.init.kaiming_uniform_(self.h1.weight)
        nn.init.kaiming_uniform_(self.h2.weight)
        nn.init.kaiming_uniform_(self.h3.weight)

    def forward(self, x_in):
        xe1 = F.leaky_relu(self.input(x_in))
        xe2 = F.leaky_relu(self.h1(xe1))
        xe3 = F.leaky_relu(self.h2(xe2))
        x_rom_out = (self.h3(xe3))
        return x_rom_out


class Decoder(nn.Module):
    def __init__(self, x_dim, x_rom_dim):
        super(Decoder, self).__init__()
        h1_nodes = min(x_rom_dim, (x_dim + x_rom_dim) // 4)
        h2_nodes = min(x_rom_dim, (x_dim + x_rom_dim) // 4)
        h3_nodes = min(x_rom_dim, (x_dim + x_rom_dim) // 2)

        self.input = nn.Linear(x_rom_dim, h1_nodes)
        self.h1 = nn.Linear(h1_nodes, h2_nodes)
        self.h2 = nn.Linear(h2_nodes, h3_nodes)
        self.h3 = nn.Linear(h3_nodes, x_dim)

        nn.init.kaiming_uniform_(self.input.weight)
        nn.init.kaiming_uniform_(self.h1.weight)
        nn.init.kaiming_uniform_(self.h2.weight)
        nn.init.kaiming_uniform_(self.h3.weight)

    def forward(self, x_rom_in):
        xe1 = F.leaky_relu(self.input(x_rom_in))
        xe2 = F.leaky_relu(self.h1(xe1))
        xe3 = F.leaky_relu(self.h2(xe2))
        x_full_out = (self.h3(xe3))
        return x_full_out


class Autoencoder:
    def __init__(self, x_dim, x_rom_dim=10):
        self.encoder = Encoder(x_dim, x_rom_dim)
        self.decoder = Decoder(x_dim, x_rom_dim)

        self.scaler_x = preprocessing.MinMaxScaler()

    def process_and_normalize_data(self, dataframe):
        x = []
        for i in range(20):
            x.append(dataframe["x{}".format(i)].to_numpy(dtype=np.float32))

        # x is fit with n rows and 20 columns, so we need to reshape it to this for transform
        # Tranpose x to obtain a 2D array with shape (num_trajectories * time, 20)
        x = np.array(x).transpose()
        self.scaler_x.fit(x)
        x = self.scaler_x.transform(x)
        x_tensor = torch.tensor(x)
        return x_tensor

    def fit(self, dataframe):
        x = self.process_and_normalize_data(dataframe)

        self.encoder.train()
        self.decoder.train()

        mb_loader = torch.utils.data.DataLoader(x, batch_size=120, shuffle=False)

        param_wrapper = nn.ParameterList()
        param_wrapper.extend(self.encoder.parameters())
        param_wrapper.extend(self.decoder.parameters())

        optimizer = optim.SGD(param_wrapper, lr=0.05)
        criterion = nn.MSELoss()

        for epoch in range(1000):
            for x_mb in mb_loader:
                optimizer.zero_grad()
                x_rom_mb = self.encoder(x_mb)
                x_full_pred_mb = self.decoder(x_rom_mb)
                loss = criterion(x_full_pred_mb, x_mb)
                loss.backward()
                optimizer.step()

            # Test on the whole dataset at this epoch
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                x_rom = self.encoder(x)
                x_full_pred = self.decoder(x_rom)
                loss = criterion(x_full_pred, x)
                print("Epoch {}: Loss = {}".format(epoch, loss))
            self.encoder.train()
            self.decoder.train()

    def encode(self, dataframe):
        x = self.process_and_normalize_data(dataframe)
        self.encoder.eval()
        with torch.no_grad():
            x_rom = self.encoder(x)

        x_rom = x_rom.numpy()
        # x_rom = self.scaler_x.inverse_transform(x_rom)

        x_rom_df_cols = []
        for i in range(x_rom.shape[1]):
            x_rom_df_cols.append("x{}_rom".format(i))
        x_rom_df = pd.DataFrame(x_rom, columns=x_rom_df_cols)

        print(x_rom_df)


def load_pickle(filename="heatEq_autoencoder.pickle"):
    with open(filename, "rb") as model:
        pickled_autoencoder = pickle.load(model)
    print("Pickle loaded: " + filename)
    return pickled_autoencoder


if __name__ == "__main__":
    data = pd.read_csv("heatEq_240_trajectories_df.csv")
    autoencoder = Autoencoder(x_dim=20, x_rom_dim=10)
    autoencoder.fit(data)

    autoencoder.encode(data)


