import pickle

import numpy as np
import pandas as pd
import pysindy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, x_dim, x_rom_dim):
        super(Encoder, self).__init__()
        h1_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 2)  # max(3, 23//2) = 11
        h2_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 2)  # max(3, 23//2) = 11
        h3_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 4)  # max(3, 23//4) = 5

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
        h1_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 4)  # max(3, 23//4) = 5
        h2_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 2)  # max(3, 23//2) = 11
        h3_nodes = max(x_rom_dim, (x_dim + x_rom_dim) // 2)  # max(3, 23//2) = 11

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
    def __init__(self, x_dim, x_rom_dim):
        self.encoder = Encoder(x_dim, x_rom_dim)
        self.decoder = Decoder(x_dim, x_rom_dim)

        self.scaler_x = preprocessing.MinMaxScaler()

    def process_and_normalize_data(self, dataframe):
        x = []
        for i in range(20):
            if i != 5:
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
            if i != 5:
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

        mb_loader = torch.utils.data.DataLoader(x, batch_size=110, shuffle=False)

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
        x_rom_nparr = np.array(x_rom_nparr, dtype=np.float32).flatten().reshape(-1, 1)
        x_rom_tensor = torch.tensor(x_rom_nparr)
        self.decoder.eval()
        with torch.no_grad():
            x_decoded = self.decoder(x_rom_tensor)

        # Scale x_decoded into x_full
        x_decoded = self.scaler_x.inverse_transform(x_decoded)
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

    x5_fit = dataframe_fit["x5"].to_numpy().flatten().reshape(-1, 1)
    # x13_fit = dataframe_fit["x13"].to_numpy().flatten().reshape(-1, 1)
    x5_score = dataframe_score["x5"].to_numpy().flatten().reshape(-1, 1)
    # x13_score = dataframe_score["x13"].to_numpy().flatten().reshape(-1, 1)

    # Sindy needs to know the controller signals
    u0_fit = dataframe_fit["u0"].to_numpy().flatten().reshape(-1, 1)
    u1_fit = dataframe_fit["u1"].to_numpy().flatten().reshape(-1, 1)
    u0_score = dataframe_score["u0"].to_numpy().flatten().reshape(-1, 1)
    u1_score = dataframe_score["u1"].to_numpy().flatten().reshape(-1, 1)

    # Try to scale everything to the same scale
    u0_scaler = preprocessing.MinMaxScaler()
    u1_scaler = preprocessing.MinMaxScaler()
    x_rom_scaler = preprocessing.MinMaxScaler()
    x5_scaler = preprocessing.MinMaxScaler()
    # x13_scaler = preprocessing.MinMaxScaler()
    u0_scaler.fit(u0_fit)
    u1_scaler.fit(u1_fit)
    x_rom_scaler.fit(x_rom_fit)
    x5_scaler.fit(x5_fit)
    # x13_scaler.fit(x13_fit)

    u0_fit = u0_scaler.transform(u0_fit)
    u0_score = u0_scaler.transform(u0_score)
    u1_fit = u1_scaler.transform(u1_fit)
    u1_score = u1_scaler.transform(u1_score)
    x_rom_fit = x_rom_scaler.transform(x_rom_fit)
    x_rom_score = x_rom_scaler.transform(x_rom_score)

    x5_fit = x5_scaler.transform(x5_fit)
    # x13_fit = x13_scaler.transform(x13_fit)
    x5_score = x5_scaler.transform(x5_score)
    # x13_score = x13_scaler.transform(x13_score)

    x_rom_fit = np.hstack((x_rom_fit, x5_fit))
    # x_stacked_fit = np.hstack((x_rom_fit, x5_fit, x13_fit))
    x_rom_score = np.hstack((x_rom_score, x5_score))
    # x_stacked_score = np.hstack((x_rom_score, x5_score, x13_score))

    # We need to split x_rom and u to into a list of 240 trajectories for sindy
    # num_trajectories = 1680
    # num_trajectories = 240
    num_trajectories = 400
    u0_list_fit = np.split(u0_fit, num_trajectories)
    u1_list_fit = np.split(u1_fit, num_trajectories)
    u_list_fit = []
    for u0_fit, u1_fit in zip(u0_list_fit, u1_list_fit):
        u_list_fit.append(np.hstack((u0_fit.reshape(-1, 1), u1_fit.reshape(-1, 1))))
    x_rom_list_fit = np.split(x_rom_fit, num_trajectories, axis=0)
    # x_stacked_fit = np.split(x_stacked_fit, num_trajectories, axis=0)

    # num_trajectories = 240
    num_trajectories = 100
    u0_list_score = np.split(u0_score, num_trajectories)
    u1_list_score = np.split(u1_score, num_trajectories)
    u_list_score = []
    for u0_score, u1_score in zip(u0_list_score, u1_list_score):
        u_list_score.append(np.hstack((u0_score.reshape(-1, 1), u1_score.reshape(-1, 1))))
    x_rom_list_score = np.split(x_rom_score, num_trajectories, axis=0)
    # x_stacked_score = np.split(x_stacked_score, num_trajectories, axis=0)

    # print(u_list_fit)
    # print(u_list_score)

    # ----- SINDY FROM PYSINDY -----

    # ------ FIT WITH CONTROL OF XTILDE ----------
    # Get the polynomial feature library
    # include_interaction = False precludes terms like x0x1, x2x3
    poly_library = pysindy.PolynomialLibrary(include_interaction=False, degree=2, include_bias=True)
    fourier_library = pysindy.FourierLibrary(n_frequencies=3)
    identity_library = pysindy.IdentityLibrary()
    combined_library = poly_library + fourier_library + identity_library

    # Smooth our possibly noisy data (as it is generated with random spiky controls) (doesn't work)
    smoothed_fd = pysindy.SmoothedFiniteDifference(drop_endpoints=True)
    fd_drop_endpoints = pysindy.FiniteDifference(drop_endpoints=True)

    # Tell Sindy that the data is recorded at 0.1s intervals
    # sindy_model = pysindy.SINDy(t_default=0.1)
    sindy_model = pysindy.SINDy(t_default=0.1, feature_library=poly_library)
    # sindy_model = pysindy.SINDy(t_default=0.1, differentiation_method=smoothed_fd)

    # print(u_list_fit)
    # sindy_model.fit(x=x_rom, u=np.hstack((u0.reshape(-1, 1), u1.reshape(-1, 1))))
    sindy_model.fit(x=x_rom_list_fit, u=u_list_fit, multiple_trajectories=True)
    sindy_model.print()

    print("R2")
    score = sindy_model.score(x=x_rom_list_score, u=u_list_score, multiple_trajectories=True, metric=r2_score)
    print(score)
    score = sindy_model.score(x=x_rom_list_score, u=u_list_score, multiple_trajectories=True, metric=mean_squared_error)
    print("MSE")
    print(score)
    score = sindy_model.score(x=x_rom_list_score, u=u_list_score, multiple_trajectories=True,
                              metric=mean_absolute_error)
    print("MAE")
    print(score)

    # # ------ FIT FOR NATURAL SYSTEM INTERACTION  ----------
    # # Get the polynomial feature library
    # # include_interaction = False precludes terms like x0x1, x2x3
    # poly_library = pysindy.PolynomialLibrary(include_interaction=False, degree=2, include_bias=True)
    # fourier_library = pysindy.FourierLibrary(n_frequencies=3)
    # identity_library = pysindy.IdentityLibrary()
    # combined_library = poly_library + fourier_library + identity_library
    #
    # # Smooth our possibly noisy data (as it is generated with random spiky controls) (doesn't work)
    # smoothed_fd = pysindy.SmoothedFiniteDifference(drop_endpoints=True)
    # fd_drop_endpoints = pysindy.FiniteDifference(drop_endpoints=True)
    #
    # # Tell Sindy that the data is recorded at 0.1s intervals
    # # sindy_model = pysindy.SINDy(t_default=0.1)
    # sindy_model = pysindy.SINDy(t_default=0.1, feature_library=poly_library)
    # # sindy_model = pysindy.SINDy(t_default=0.1, differentiation_method=smoothed_fd)
    #
    # # sindy_model.fit(x=x_rom, u=np.hstack((u0.reshape(-1, 1), u1.reshape(-1, 1))))
    # sindy_model.fit(x=x_stacked_fit, multiple_trajectories=True)
    # sindy_model.print()
    #
    # print("R2")
    # score = sindy_model.score(x=x_stacked_score, multiple_trajectories=True, metric=r2_score)
    # print(score)
    # score = sindy_model.score(x=x_stacked_score, multiple_trajectories=True, metric=mean_squared_error)
    # print("MSE")
    # print(score)
    # score = sindy_model.score(x=x_stacked_score, multiple_trajectories=True,
    #                           metric=mean_absolute_error)
    # print("MAE")
    # print(score)

    # Get x_rom initial values
    x_init = np.full(shape=(1, 19), fill_value=273)
    x_init_df_col = []
    for i in range(20):
        if i != 5:
            x_init_df_col.append("x{}".format(i))
    x_init_df = pd.DataFrame(x_init, columns=x_init_df_col)
    autoencoder = load_pickle("heatEq_autoencoder_1dim_constraint.pickle")
    x_rom_init = autoencoder.encode(x_init_df).to_numpy()
    x_rom_init_scaled = x_rom_scaler.transform(x_rom_init)
    print("x_init - scaled for sindy dont scale again")
    print(x_rom_init_scaled)
    # Initial values for x_rom (scaled for Sindy):
    #      x0_rom     x1_rom        x2_rom
    #   0.2628937  0.6858409  0.44120657

    print("x5 init 273 K")
    print(x5_scaler.transform([[273]]))
    # print("x13 init 273 K")
    # print(x13_scaler.transform([[273]]))

    print("x5 sp 303 K")
    print(x5_scaler.transform([[303]]))
    # print("x13 sp 333 K")
    # print(x13_scaler.transform([[333]]))

    # Discover setpoint for x_rom
    x_rom_setpoints = discover_objectives(autoencoder)
    x_rom_setpoints_scaled = x_rom_scaler.transform(x_rom_setpoints)
    print("Setpoints - scaled for sindy dont scale again")
    print(x_rom_setpoints_scaled)

    # Discovered setpoints for x_rom (scaled)
    # 0.70213366 0.211213   0.98931336

    # # Decode
    # x_final = np.array([0.6878071340000952, 0.22097627550550195, 1.04697611250378], dtype=np.float32).reshape(1, 3)
    # x_final = x_rom_scaler.inverse_transform(x_final)
    # x_final = torch.tensor(x_final)
    # x_final = autoencoder.decode(x_final)
    # print("Final state")
    # print(x_final)

    return


def discover_objectives(ae_model):
    # Encode the final full state of the MPC controlled system
    x_full_setpoint_dict = {}

    x = [312.043309, 308.530539, 305.935527, 304.179534, 303.210444,
         302.999869, 303.541685, 304.851730, 306.968621, 309.955792,
         313.904979, 318.941577, 325.232511, 332.997712, 342.526917,
         354.204601, 368.547821, 386.265378, 408.354234, 436.265516]

    # x = [312.043309, 308.530539, 305.935527, 304.179534, 303.210444,
    #      302.999869, 303.541685, 304.851730, 306.968621, 309.955792,
    #      313.904979, 318.941577, 325.232511, 332.997712, 342.526917,
    #      354.204601, 368.547821, 386.265378, 408.354234, 436.265516]

    # x = [281.901332, 286.207153, 290.410114, 294.513652, 298.529802,
    #      302.456662, 306.284568, 309.997554, 313.567900, 316.947452,
    #      320.059310, 322.793336, 325.009599, 326.553958, 327.286519,
    #      327.112413, 325.983627, 323.826794, 320.424323, 315.694468]

    for i in range(20):
        x_full_setpoint_dict["x{}".format(i)] = x[i]

    x_full_setpoint_df = pd.DataFrame(x_full_setpoint_dict, index=[0])
    # This is our setpoint for the reduced model
    x_rom_setpoints = ae_model.encode(x_full_setpoint_df).to_numpy()

    print(x_rom_setpoints)
    return x_rom_setpoints


if __name__ == "__main__":
    # generate_data_for_svm()
    # run_svm()

    # data = pd.read_csv("data/mpc_data19.csv")
    # test_data = pd.read_csv("data/mpc_data_test9.csv")
    # autoencoder = Autoencoder(x_dim=19, x_rom_dim=1)
    # autoencoder.fit(data, test_data)
    # with open("heatEq_autoencoder_1dim_constraint.pickle", "wb") as model:
    #     pickle.dump(autoencoder, model)

    data_fit = pd.read_csv("data/mpc_data19.csv")
    data_score = pd.read_csv("data/mpc_data_test9.csv")
    autoencoder = load_pickle("heatEq_autoencoder_1dim_constraint.pickle")
    sindy(autoencoder, data_fit, data_score)

    # x_init = np.full(shape=(1, 20), fill_value=273)
    # df_cols = []
    # for i in range(20):
    #     df_cols.append("x{}".format(i))
    # df = pd.DataFrame(x_init, columns=df_cols)
    # x_rom = autoencoder.encode(df)
    # print("x_rom initial")
    # print(x_rom)
