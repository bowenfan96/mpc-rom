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

from sklearn.preprocessing import PolynomialFeatures
from deeptime.sindy import STLSQ
from deeptime.sindy import SINDy

from scipy import optimize


class Encoder(nn.Module):
    def __init__(self, x_dim, x_rom_dim):
        super(Encoder, self).__init__()
        h1_nodes = min(x_rom_dim, (x_dim + x_rom_dim) // 2)
        h2_nodes = min(x_rom_dim, (x_dim + x_rom_dim) // 2)
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
        xe1 = F.elu(self.input(x_in))
        xe2 = F.elu(self.h1(xe1))
        xe3 = F.elu(self.h2(xe2))
        x_rom_out = (self.h3(xe3))

        # xe1 = F.leaky_relu(self.input(x_in))
        # xe2 = F.leaky_relu(self.h1(xe1))
        # xe3 = F.leaky_relu(self.h2(xe2))
        # x_rom_out = (self.h3(xe3))

        return x_rom_out


class Decoder(nn.Module):
    def __init__(self, x_dim, x_rom_dim):
        super(Decoder, self).__init__()
        h1_nodes = min(x_rom_dim, (x_dim + x_rom_dim) // 4)
        h2_nodes = min(x_rom_dim, (x_dim + x_rom_dim) // 2)
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
        xe1 = F.elu(self.input(x_rom_in))
        xe2 = F.elu(self.h1(xe1))
        xe3 = F.elu(self.h2(xe2))
        x_full_out = (self.h3(xe3))

        # xe1 = F.leaky_relu(self.input(x_rom_in))
        # xe2 = F.leaky_relu(self.h1(xe1))
        # xe3 = F.leaky_relu(self.h2(xe2))
        # x_full_out = (self.h3(xe3))

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

    def transform_data_without_fit(self, dataframe):
        x = []
        for i in range(20):
            x.append(dataframe["x{}".format(i)].to_numpy(dtype=np.float32))

        # x is fit with n rows and 20 columns, so we need to reshape it to this for transform
        # Tranpose x to obtain a 2D array with shape (num_trajectories * time, 20)
        x = np.array(x).transpose()
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
        print(dataframe)
        x = self.transform_data_without_fit(dataframe)
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
        return x_rom_df

    def decode(self, x_rom_nparr):
        # Expected shape of x_rom_nparr is (x_rom_dim, )
        # Reshape to match decoder dimensions
        x_rom_nparr = x_rom_nparr.reshape(1, 5)
        self.decoder.eval()
        with torch.no_grad():
            x_decoded = self.decoder(x_rom_nparr)

        # Scale x_decoded into x_full
        x_decoded = self.scaler_x.inverse_transform(x_decoded)

        # Return x_decoded as a numpy array, not pandas dataframe (to the basinhopper)
        return x_decoded


def load_pickle(filename):
    with open(filename, "rb") as model:
        pickled_autoencoder = pickle.load(model)
    print("Pickle loaded: " + filename)
    return pickled_autoencoder


def sindy(ae_model, dataframe_fit, dataframe_score):
    # x_rom from autoencoder, returned as dataframe with shape (2400, 10)
    x_rom_fit = ae_model.encode(dataframe_fit).to_numpy()
    x_rom_score = ae_model.encode(dataframe_score).to_numpy()

    # Try to scale x_rom to the same order to magnitude as u
    x_rom_fit = x_rom_fit * 10
    x_rom_score = x_rom_score * 10

    # Sindy needs to know the controller signals
    u0_fit = dataframe_fit["u0"].to_numpy().flatten()
    u1_fit = dataframe_fit["u1"].to_numpy().flatten()
    u0_score = dataframe_score["u0"].to_numpy().flatten()
    u1_score = dataframe_score["u1"].to_numpy().flatten()

    # We need to split x_rom and u to into a list of 240 trajectories for sindy
    num_trajectories = 180
    u0_list_fit = np.split(u0_fit, num_trajectories)
    u1_list_fit = np.split(u1_fit, num_trajectories)
    u_list_fit = []
    for u0_fit, u1_fit in zip(u0_list_fit, u1_list_fit):
        u_list_fit.append(np.hstack((u0_fit.reshape(-1, 1), u1_fit.reshape(-1, 1))))
    x_rom_list_fit = np.split(x_rom_fit, num_trajectories, axis=0)

    num_trajectories = 60
    u0_list_score = np.split(u0_score, num_trajectories)
    u1_list_score = np.split(u1_score, num_trajectories)
    u_list_score = []
    for u0_score, u1_score in zip(u0_list_score, u1_list_score):
        u_list_score.append(np.hstack((u0_score.reshape(-1, 1), u1_score.reshape(-1, 1))))
    x_rom_list_score = np.split(x_rom_score, num_trajectories, axis=0)

    # print(u_list_fit)
    # print(u_list_score)

    # ----- SINDY FROM PYSINDY -----
    # Get the polynomial feature library
    # include_interaction = False precludes terms like x0x1, x2x3
    poly_library = pysindy.PolynomialLibrary(include_interaction=False, degree=1)

    # Smooth our possibly noisy data (as it is generated with random spiky controls) (doesn't work)
    smoothed_fd = pysindy.SmoothedFiniteDifference()

    # Tell Sindy that the data is recorded at 0.1s intervals
    # sindy_model = pysindy.SINDy(t_default=0.1)
    sindy_model = pysindy.SINDy(t_default=0.1, feature_library=poly_library)
    # sindy_model = pysindy.SINDy(t_default=0.1, differentiation_method=smoothed_fd)

    # sindy_model.fit(x=x_rom, u=np.hstack((u0.reshape(-1, 1), u1.reshape(-1, 1))))
    sindy_model.fit(x=x_rom_list_fit, u=u_list_fit, multiple_trajectories=True)
    sindy_model.print()
    score = sindy_model.score(x=x_rom_list_score, u=u_list_score, multiple_trajectories=True)
    print(score)

    return


def discover_objectives(ae_model):
    # Encode the final full state of the MPC controlled system
    x_full_setpoint_dict = {}

    x = [281.901332, 286.207153, 290.410114, 294.513652, 298.529802,
         302.456662, 306.284568, 309.997554, 313.567900, 316.947452,
         320.059310, 322.793336, 325.009599, 326.553958, 327.286519,
         327.112413, 325.983627, 323.826794, 320.424323, 315.694468]

    for i in range(20):
        x_full_setpoint_dict["x{}".format(i)] = x[i]

    x_full_setpoint_df = pd.DataFrame(x_full_setpoint_dict, index=[0])
    # This is our setpoint for the reduced model
    x_rom_setpoints = ae_model.encode(x_full_setpoint_df).to_numpy()

    print(x_rom_setpoints)
    return x_rom_setpoints


if __name__ == "__main__":
    data = pd.read_csv("heatEq_240_trajectories_df.csv")
    autoencoder = Autoencoder(x_dim=20, x_rom_dim=5)
    autoencoder.fit(data)
    # with open("heatEq_autoencoder_10dim.pickle", "wb") as model:
    #     pickle.dump(autoencoder, model)


    # Get x_rom initial values
    # x_init = np.full(shape=(1, 20), fill_value=273)
    # x_init_df_col = []
    # for i in range(20):
    #     x_init_df_col.append("x{}".format(i))
    # x_init_df = pd.DataFrame(x_init, columns=x_init_df_col)
    # autoencoder = load_pickle("heatEq_autoencoder_5dim.pickle")
    # x_rom_init_scaled = autoencoder.encode(x_init_df).to_numpy() * 100
    # print(x_rom_init_scaled)
    # Initial values for x_rom (scaled by 100x):
    #      x0_rom     x1_rom        x2_rom     x3_rom      x4_rom
    #   -38.80465    -4.298748   65.039635 -147.96707   -52.63922


    # data_fit = pd.read_csv("heatEq_180_trajectories_df_sindy_fit.csv")
    # data_score = pd.read_csv("heatEq_60_trajectories_df_sindy_score.csv")
    # autoencoder = load_pickle("heatEq_autoencoder_5dim.pickle")
    # sindy(autoencoder, data_fit, data_score)

    # Discover setpoint for x_rom
    # autoencoder = load_pickle("heatEq_autoencoder_5dim.pickle")
    # x_rom_setpoints = discover_objectives(autoencoder)
    # x_rom_setpoints_scaled = x_rom_setpoints * 100
    # print(x_rom_setpoints_scaled)

    # Discovered setpoints for x_rom (scaled)
    # -85.02493   -18.394802  100.83086  -235.56361   -75.54978
