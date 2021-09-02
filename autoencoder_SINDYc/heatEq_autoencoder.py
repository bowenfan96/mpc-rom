import pickle

import numpy as np
import pandas as pd
import pysindy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from pyomo.core import Expression
from pyomo.environ import value
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
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
    # num_trajectories = 1680
    # num_trajectories = 240
    num_trajectories = 400
    u0_list_fit = np.split(u0_fit, num_trajectories)
    u1_list_fit = np.split(u1_fit, num_trajectories)
    u_list_fit = []
    for u0_fit, u1_fit in zip(u0_list_fit, u1_list_fit):
        u_list_fit.append(np.hstack((u0_fit.reshape(-1, 1), u1_fit.reshape(-1, 1))))
    x_rom_list_fit = np.split(x_rom_fit, num_trajectories, axis=0)

    # num_trajectories = 240
    num_trajectories = 100
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
    poly_library = pysindy.PolynomialLibrary(include_interaction=True, degree=5, include_bias=True)
    fourier_library = pysindy.FourierLibrary(n_frequencies=3)
    identity_library = pysindy.IdentityLibrary()
    combined_library = poly_library + fourier_library + identity_library

    # Smooth our possibly noisy data (as it is generated with random spiky controls) (doesn't work)
    smoothed_fd = pysindy.SmoothedFiniteDifference(drop_endpoints=True)
    fd_drop_endpoints = pysindy.FiniteDifference(drop_endpoints=True)

    # Tell Sindy that the data is recorded at 0.1s intervals
    # sindy_model = pysindy.SINDy(t_default=0.1)
    sindy_model = pysindy.SINDy(t_default=0.1, feature_library=poly_library, differentiation_method=fd_drop_endpoints)
    # sindy_model = pysindy.SINDy(t_default=0.1, differentiation_method=smoothed_fd)

    # sindy_model.fit(x=x_rom, u=np.hstack((u0.reshape(-1, 1), u1.reshape(-1, 1))))
    sindy_model.fit(x=x_rom_list_fit, u=u_list_fit, multiple_trajectories=True)
    sindy_model.print()

    print("R2")
    score = sindy_model.score(x=x_rom_list_score, u=u_list_score, multiple_trajectories=True, metric=r2_score)
    print(score)
    score = sindy_model.score(x=x_rom_list_score, u=u_list_score, multiple_trajectories=True, metric=mean_squared_error)
    print("MSE")
    print(score)
    score = sindy_model.score(x=x_rom_list_score, u=u_list_score, multiple_trajectories=True, metric=mean_absolute_error)
    print("MAE")
    print(score)


    # Get x_rom initial values
    x_init = np.full(shape=(1, 20), fill_value=273)
    x_init_df_col = []
    for i in range(20):
        x_init_df_col.append("x{}".format(i))
    x_init_df = pd.DataFrame(x_init, columns=x_init_df_col)
    autoencoder = load_pickle("heatEq_autoencoder_1dim_elu_mpcData_setpoint.pickle")
    x_rom_init = autoencoder.encode(x_init_df).to_numpy()
    x_rom_init_scaled = x_rom_scaler.transform(x_rom_init)
    print("x_init - scaled for sindy dont scale again")
    print(x_rom_init_scaled)
    # Initial values for x_rom (scaled for Sindy):
    #      x0_rom     x1_rom        x2_rom
    #   0.2628937  0.6858409  0.44120657

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

def sindy_fom(dataframe_fit, dataframe_score):
    # x_rom from autoencoder, returned as dataframe with shape (2400, 10)
    x_fit = []
    for i in range(20):
        x_fit.append(dataframe_fit["x{}".format(i)].to_numpy())
    x_rom_fit = np.array(x_fit).transpose()
    x_score = []
    for i in range(20):
        x_score.append(dataframe_score["x{}".format(i)].to_numpy())
    x_rom_score = np.array(x_score).transpose()

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
    poly_library = pysindy.PolynomialLibrary(include_interaction=True, degree=5, include_bias=False)
    fourier_library = pysindy.FourierLibrary(n_frequencies=2)
    identity_library = pysindy.IdentityLibrary()
    combined_library = poly_library + fourier_library + identity_library

    # Smooth our possibly noisy data (as it is generated with random spiky controls) (doesn't work)
    smoothed_fd = pysindy.SmoothedFiniteDifference(drop_endpoints=True)
    fd_drop_endpoints = pysindy.FiniteDifference(drop_endpoints=True)

    # Tell Sindy that the data is recorded at 0.1s intervals
    # sindy_model = pysindy.SINDy(t_default=0.1)
    sindy_model = pysindy.SINDy(t_default=0.1, feature_library=poly_library, differentiation_method=fd_drop_endpoints)
    # sindy_model = pysindy.SINDy(t_default=0.1, differentiation_method=smoothed_fd)

    # sindy_model.fit(x=x_rom, u=np.hstack((u0.reshape(-1, 1), u1.reshape(-1, 1))))
    sindy_model.fit(x=x_rom_list_fit, u=u_list_fit, multiple_trajectories=True)
    sindy_model.print()

    print("R2")
    score = sindy_model.score(x=x_rom_list_score, u=u_list_score, multiple_trajectories=True, metric=r2_score)
    print(score)
    score = sindy_model.score(x=x_rom_list_score, u=u_list_score, multiple_trajectories=True, metric=mean_squared_error)
    print("MSE")
    print(score)
    score = sindy_model.score(x=x_rom_list_score, u=u_list_score, multiple_trajectories=True, metric=mean_absolute_error)
    print("MAE")
    print(score)


    # Get x_rom initial values
    x_init = np.full(shape=(1, 20), fill_value=273)
    x_init_df_col = []
    for i in range(20):
        x_init_df_col.append("x{}".format(i))
    x_init_df = pd.DataFrame(x_init, columns=x_init_df_col)
    autoencoder = load_pickle("heatEq_autoencoder_1dim_elu_mpcData_setpoint.pickle")
    x_rom_init = autoencoder.encode(x_init_df).to_numpy()
    x_rom_init_scaled = x_rom_scaler.transform(x_rom_init)
    print("x_init - scaled for sindy dont scale again")
    print(x_rom_init_scaled)
    # Initial values for x_rom (scaled for Sindy):
    #      x0_rom     x1_rom        x2_rom
    #   0.2628937  0.6858409  0.44120657

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


def generate_data_for_svm():
    autoencoder = load_pickle("heatEq_autoencoder_1dim_elu_mpcData_setpoint.pickle")

    num_samples = 3000

    # Rough calculation: 263 to 313 = 50, 263 to 343 = 80, 80*20 = 1600
    # Generate PASS data: x5 is less than 313
    pass_states = []
    for i in range(num_samples):
        # x0 to x19 can be anything in the range of 263 to 343 (from initial-10 to setpoint+10)
        x0_x4 = np.random.randint(low=263, high=343, size=(5, ))
        # x5 must be less than 313
        x5 = np.random.randint(low=263, high=313)
        x6_x19 = np.random.randint(low=263, high=343, size=(14,))
        # Get one pass state, append to list of passed state - label is 1
        state = np.hstack((x0_x4, x5, x6_x19)).flatten()
        pass_states.append(state)

    # Convert the passed states in the full state space to the reduced space
    pass_states = np.array(pass_states)
    df_cols = []
    for i in range(20):
        df_cols.append("x{}".format(i))
    pass_states_df = pd.DataFrame(pass_states, columns=df_cols)
    print(pass_states_df)

    rom_pass_states = autoencoder.encode(pass_states_df)

    # Create a vector of ones to label the pass state
    ones_vector = np.ones(shape=num_samples, dtype=int)
    ones_vector_df = pd.DataFrame(ones_vector, columns=["pass"])
    # hstack pass dataframe with label
    pass_df = pd.concat([rom_pass_states, ones_vector_df], axis=1)

    # Generate FAIL data: x5 is more than 313
    fail_states = []
    for i in range(num_samples):
        # x0 to x19 can be anything in the range of 263 to 343 (from initial-10 to setpoint+10)
        x0_x4 = np.random.randint(low=263, high=343, size=(5, ))
        # x5 fails if its more than 313
        x5 = np.random.randint(low=314, high=333)
        x6_x19 = np.random.randint(low=263, high=343, size=(14,))
        # Get one fail state, append to list of failed states - label is 0
        state = np.hstack((x0_x4, x5, x6_x19)).flatten()
        fail_states.append(state)

    # Convert the failed states in the full state space to the reduced space
    fail_states = np.array(fail_states)
    #
    fail_states_df = pd.DataFrame(fail_states, columns=df_cols)
    print(fail_states_df)

    rom_fail_states = autoencoder.encode(fail_states_df)

    # Create a vector of zeros to label the fail state
    zeros_vector = np.zeros(shape=num_samples, dtype=int)
    zeros_vector_df = pd.DataFrame(zeros_vector, columns=["pass"])
    # hstack pass dataframe with label
    fail_df = pd.concat([rom_fail_states, zeros_vector_df], axis=1)

    # vstack the SVM training data and save to csv
    svm_training_data = pd.concat([pass_df, fail_df], axis=0)
    svm_training_data.to_csv("svm_training_data.csv")


def run_svm():
    training_df = pd.read_csv("../heatEq_3dims_wR_wPathCst/svm_training_data.csv")
    x_y = training_df.to_numpy()
    X = x_y[:, 1:4]
    print(X)
    Y = x_y[:, -1]
    print(Y)
    model = svm.SVC(kernel='linear', verbose=1)
    clf = model.fit(X, Y)
    # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
    # Solve for w3 (z)
    z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]
    tmp = np.linspace(-1.5, 2.5, 40)
    x, y = np.meshgrid(tmp, tmp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], 'ob')
    ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'sr')
    ax.plot_surface(x, y, z(x, y), alpha=0.2)
    ax.set_xlim3d(-1, 2)
    ax.set_ylim3d(-1, 2)
    ax.set_zlim3d(-1, 2)
    ax.view_init(0, -60)
    plt.ion()
    plt.show()
    plt.savefig("svm_decision_boundary.svg", format="svg")




if __name__ == "__main__":
    # generate_data_for_svm()
    # run_svm()

    # data = pd.read_csv("data/mpc_data19.csv")
    # test_data = pd.read_csv("data/mpc_data_test9.csv")
    # autoencoder = Autoencoder(x_dim=20, x_rom_dim=1)
    # autoencoder.fit(data, test_data)
    # with open("heatEq_autoencoder_1dim_elu_mpcData_setpoint.pickle", "wb") as model:
    #     pickle.dump(autoencoder, model)

    data_fit = pd.read_csv("data/mpc_data19.csv")
    data_score = pd.read_csv("data/mpc_data_test9.csv")
    autoencoder = load_pickle("heatEq_autoencoder_1dim_elu_mpcData_setpoint.pickle")
    sindy(autoencoder, data_fit, data_score)
    # sindy_fom(data_fit, data_score)

    # x_init = np.full(shape=(1, 20), fill_value=273)
    # df_cols = []
    # for i in range(20):
    #     df_cols.append("x{}".format(i))
    # df = pd.DataFrame(x_init, columns=df_cols)
    # x_rom = autoencoder.encode(df)
    # print("x_rom initial")
    # print(x_rom)