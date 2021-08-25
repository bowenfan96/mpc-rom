import pyomo.core
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pyomo.core import Expression
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

from sklearn import svm

from pyomo.environ import value


class Encoder(nn.Module):
    def __init__(self, x_dim, x_rom_dim):
        super(Encoder, self).__init__()
        h1_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 2)     # max(3, 23//2) = 11
        h2_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 2)     # max(3, 23//2) = 11
        h3_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 4)     # max(3, 23//4) = 5

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
        h1_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 4)     # max(3, 23//4) = 5
        h2_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 2)     # max(3, 23//2) = 11
        h3_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 2)     # max(3, 23//2) = 11

        self.input = nn.Linear(x_rom_dim, h1_nodes)
        self.h1 = nn.Linear(h1_nodes, h2_nodes)
        self.h2 = nn.Linear(h2_nodes, h3_nodes)
        self.h3 = nn.Linear(h3_nodes, x_dim)

        nn.init.kaiming_uniform_(self.input.weight)
        nn.init.kaiming_uniform_(self.h1.weight)
        nn.init.kaiming_uniform_(self.h2.weight)
        nn.init.kaiming_uniform_(self.h3.weight)

    def forward(self, x_rom_in):
        xe1 = torch.tanh(self.input(x_rom_in))
        xe2 = torch.tanh(self.h1(xe1))
        xe3 = torch.tanh(self.h2(xe2))
        x_full_out = (self.h3(xe3))

        # xe1 = F.leaky_relu(self.input(x_rom_in))
        # xe2 = F.leaky_relu(self.h1(xe1))
        # xe3 = F.leaky_relu(self.h2(xe2))
        # x_full_out = (self.h3(xe3))

        return x_full_out


class Autoencoder:
    def __init__(self, x_dim, x_rom_dim=3):
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

    def fit(self, dataframe, test_dataframe):
        x = self.process_and_normalize_data(dataframe)
        x_test = self.transform_data_without_fit(test_dataframe)

        self.encoder.train()
        self.decoder.train()

        mb_loader = torch.utils.data.DataLoader(x, batch_size=100, shuffle=True)

        param_wrapper = nn.ParameterList()
        param_wrapper.extend(self.encoder.parameters())
        param_wrapper.extend(self.decoder.parameters())

        optimizer = optim.SGD(param_wrapper, lr=0.01)
        criterion = nn.MSELoss()

        for epoch in range(2000):
            for x_mb in mb_loader:
                optimizer.zero_grad()
                x_rom_mb = self.encoder(x_mb)
                x_full_pred_mb = self.decoder(x_rom_mb)
                loss = criterion(x_full_pred_mb, x_mb)
                loss.backward()
                optimizer.step()

            # Test on the test dataset at this epoch
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                x_rom = self.encoder(x_test)
                x_full_pred = self.decoder(x_rom)
                loss = criterion(x_full_pred, x_test)
                mae = metrics.mean_absolute_error(x_full_pred, x_test)
                mape = metrics.mean_absolute_percentage_error(x_full_pred, x_test)
                print("Held out test dataset - Epoch {}: MSE = {}, MAE = {}, MAPE = {}".format(epoch, loss, mae, mape))
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
        x_rom_nparr = np.array(x_rom_nparr, dtype=np.float32).flatten().reshape(1, 3)
        x_rom_tensor = torch.tensor(x_rom_nparr)
        self.decoder.eval()
        with torch.no_grad():
            x_decoded = self.decoder(x_rom_tensor)

        # Scale x_decoded into x_full
        x_decoded = self.scaler_x.inverse_transform(x_decoded)
        return x_decoded

    def decode_pyomo(self, x0, x1, x2):
        x0 = value(x0)
        x1 = value(x1)
        x2 = value(x2)

        x_rom_nparr = np.array([x0, x1, x2], dtype=np.float32).flatten().reshape(1, 3)
        x_rom_tensor = torch.tensor(x_rom_nparr)
        self.decoder.eval()
        with torch.no_grad():
            x_decoded = self.decoder(x_rom_tensor)

        # Scale x_decoded into x_full
        x_decoded = self.scaler_x.inverse_transform(x_decoded)

        # Return x_decoded as a numpy array, not pandas dataframe (to the basinhopper)
        # return x_decoded
        print(x_decoded)
        x5_decoded = np.array(x_decoded).flatten()[5]
        # if x5_decoded < 313:
        #     return pyomo.core.Constraint.Feasible
        # else:
        #     return pyomo.core.Constraint.Infeasible
        return Expression(rule=x5_decoded<=313)

def load_pickle(filename):
    with open(filename, "rb") as model:
        pickled_autoencoder = pickle.load(model)
    print("Pickle loaded: " + filename)
    return pickled_autoencoder


def sindy(ae_model, dataframe_fit, dataframe_score):
    # x_rom from autoencoder, returned as dataframe with shape (2400, 10)
    x_rom_fit = ae_model.encode(dataframe_fit).to_numpy()
    x_rom_score = ae_model.encode(dataframe_score).to_numpy()

    # Sindy needs to know the controller signals
    u0_fit = dataframe_fit["u0"].to_numpy().flatten().reshape(-1, 1)
    u1_fit = dataframe_fit["u1"].to_numpy().flatten().reshape(-1, 1)
    u0_score = dataframe_score["u0"].to_numpy().flatten().reshape(-1, 1)
    u1_score = dataframe_score["u1"].to_numpy().flatten().reshape(-1, 1)

    # Try to scale everything to the same scale
    u0_scaler = preprocessing.MinMaxScaler()
    u1_scaler = preprocessing.MinMaxScaler()
    x_rom_scaler = preprocessing.MinMaxScaler()
    u0_scaler.fit(u0_fit)
    u1_scaler.fit(u1_fit)
    x_rom_scaler.fit(x_rom_fit)

    u0_fit = u0_scaler.transform(u0_fit)
    u0_score = u0_scaler.transform(u0_score)
    u1_fit = u1_scaler.transform(u1_fit)
    u1_score = u1_scaler.transform(u1_score)
    x_rom_fit = x_rom_scaler.transform(x_rom_fit)
    x_rom_score = x_rom_scaler.transform(x_rom_score)

    # We need to split x_rom and u to into a list of 240 trajectories for sindy
    num_trajectories = 1680
    num_trajectories = 240
    u0_list_fit = np.split(u0_fit, num_trajectories)
    u1_list_fit = np.split(u1_fit, num_trajectories)
    u_list_fit = []
    for u0_fit, u1_fit in zip(u0_list_fit, u1_list_fit):
        u_list_fit.append(np.hstack((u0_fit.reshape(-1, 1), u1_fit.reshape(-1, 1))))
    x_rom_list_fit = np.split(x_rom_fit, num_trajectories, axis=0)

    num_trajectories = 240
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
    poly_library = pysindy.PolynomialLibrary(include_interaction=False, degree=2)
    fourier_library = pysindy.FourierLibrary(n_frequencies=6)
    identity_library = pysindy.IdentityLibrary()
    combined_library = poly_library + fourier_library + identity_library

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


    # Get x_rom initial values
    x_init = np.full(shape=(1, 20), fill_value=273)
    x_init_df_col = []
    for i in range(20):
        x_init_df_col.append("x{}".format(i))
    x_init_df = pd.DataFrame(x_init, columns=x_init_df_col)
    autoencoder = load_pickle("heatEq_autoencoder_3dim_lr001_batch100_epoch2000.pickle")
    x_rom_init = autoencoder.encode(x_init_df).to_numpy()
    x_rom_init_scaled = x_rom_scaler.transform(x_rom_init)
    print(x_rom_init_scaled)
    # Initial values for x_rom (scaled for Sindy):
    #      x0_rom     x1_rom        x2_rom
    #   0.2628937  0.6858409  0.44120657

    # Discover setpoint for x_rom
    x_rom_setpoints = discover_objectives(autoencoder)
    x_rom_setpoints_scaled = x_rom_scaler.transform(x_rom_setpoints)
    print(x_rom_setpoints_scaled)

    # Discovered setpoints for x_rom (scaled)
    # 0.70213366 0.211213   0.98931336

    # Decode
    x_final = np.array([0.6878071340000952, 0.22097627550550195, 1.04697611250378], dtype=np.float32).reshape(1, 3)
    x_final = x_rom_scaler.inverse_transform(x_final)
    x_final = torch.tensor(x_final)
    x_final = autoencoder.decode(x_final)
    print("Final state")
    print(x_final)

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
    data = pd.read_csv("data/autoencoder_training_data.csv")
    test_data = pd.read_csv("data/validation_dataset_3dim_wR_wPathCst.csv")
    autoencoder = Autoencoder(x_dim=20, x_rom_dim=3)
    autoencoder.fit(data, test_data)
    with open("heatEq_autoencoder_3dim_tanh.pickle", "wb") as model:
        pickle.dump(autoencoder, model)

    # data_score = pd.read_csv("../heatEq_3dims_wR_wPathCst/heatEq_240_trajectories_df.csv")
    # data_fit = pd.read_csv("../heatEq_3dims_wR_wPathCst/validation_dataset_3dim_wR_wPathCst.csv")
    # autoencoder = load_pickle("heatEq_autoencoder_3dim_lr001_batch100_epoch2000.pickle")
    # sindy(autoencoder, data_fit, data_score)

    autoencoder = load_pickle("heatEq_autoencoder_3dim_tanh.pickle")
    input_weight = autoencoder.decoder.input.weight.detach().cpu().numpy().T
    input_bias = autoencoder.decoder.input.bias.detach().cpu().numpy().T
    h1_weight = autoencoder.decoder.h1.weight.detach().cpu().numpy().T
    h1_bias = autoencoder.decoder.h1.bias.detach().cpu().numpy().T
    h2_weight = autoencoder.decoder.h2.weight.detach().cpu().numpy().T
    h2_bias = autoencoder.decoder.h2.bias.detach().cpu().numpy().T
    h3_weight = autoencoder.decoder.h3.weight.detach().cpu().numpy().T
    h3_bias = autoencoder.decoder.h3.bias.detach().cpu().numpy().T

    np.save("autoencoder_weights_biases_tanh/input_weight", input_weight)
    np.save("autoencoder_weights_biases_tanh/input_bias", input_bias)
    np.save("autoencoder_weights_biases_tanh/h1_weight", h1_weight)
    np.save("autoencoder_weights_biases_tanh/h1_bias", h1_bias)
    np.save("autoencoder_weights_biases_tanh/h2_weight", h2_weight)
    np.save("autoencoder_weights_biases_tanh/h2_bias", h2_bias)
    np.save("autoencoder_weights_biases_tanh/h3_weight", h3_weight)
    np.save("autoencoder_weights_biases_tanh/h3_bias", h3_bias)

    # W = [input_weight, h1_weight, h2_weight, h3_weight]
    # B = [input_bias, h1_bias, h2_bias, h3_bias]
    #
    # for i in range(4):
    #     print(W[i].shape)
    #     print(B[i].shape)
    #
    # elu = lambda Z: np.where(Z>0, Z, np.exp(Z)-1)
    #
    # y_pred = np.array([0.632633, -0.405015,  0.087859]).reshape(1, 3)
    # for i in range(4):
    #     print(i)
    #     if i == 3:
    #         y_pred = y_pred @ W[i] + B[i]
    #     else:
    #         y_pred = elu(y_pred @ W[i] + B[i])
    #
    # print(y_pred)
    # y_pred = np.array(y_pred).flatten().reshape(1, 20)
    #
    # print("Custom output")
    # print(autoencoder.scaler_x.inverse_transform(y_pred))
    #
    # print("Autoencoder output")
    # print(autoencoder.decode(np.array([0.632633, -0.405015,  0.087859])))
    #
    # x_init = np.full(shape=(1, 20), fill_value=273)
    # df_cols = []
    # for i in range(20):
    #     df_cols.append("x{}".format(i))
    # df = pd.DataFrame(x_init, columns=df_cols)
    # x_rom = autoencoder.encode(df)
    # print(x_rom)
    #
