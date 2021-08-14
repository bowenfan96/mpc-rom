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
        xe1 = F.leaky_relu(self.input(x_in))
        xe2 = F.leaky_relu(self.h1(xe1))
        xe3 = F.leaky_relu(self.h2(xe2))
        x_rom_out = (self.h3(xe3))
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


def load_pickle(filename="heatEq_autoencoder.pickle"):
    with open(filename, "rb") as model:
        pickled_autoencoder = pickle.load(model)
    print("Pickle loaded: " + filename)
    return pickled_autoencoder


def sindy(ae_model, dataframe):
    # x_rom from autoencoder, returned as dataframe with shape (2400, 10)
    x_rom = ae_model.encode(dataframe).to_numpy()

    # Sindy needs to know the controller signals
    u0 = dataframe["u0"].to_numpy().flatten()
    u1 = dataframe["u1"].to_numpy().flatten()

    # We need to split x_rom and u to into a list of 240 trajectories for sindy
    num_trajectories = 240
    u0_list = np.split(u0, num_trajectories)
    u1_list = np.split(u1, num_trajectories)
    u_list = []
    for u0, u1 in zip(u0_list, u1_list):
        u_list.append(np.hstack((u0.reshape(-1, 1), u1.reshape(-1, 1))))
    x_rom_list = np.split(x_rom, num_trajectories, axis=0)

    # ----- SINDY FROM PYSINDY -----
    # Get the polynomial feature library
    # include_interaction = False precludes terms like x0x1, x2x3
    poly_library = pysindy.PolynomialLibrary(include_interaction=False, degree=2)
    # Smooth our possibly noisy data (as it is generated with random spiky controls)
    smoothed_fd = pysindy.SmoothedFiniteDifference()
    # Tell Sindy that the data is recorded at 0.1s intervals
    sindy_model = pysindy.SINDy(t_default=0.1, feature_library=poly_library)
    # sindy_model = pysindy.SINDy(t_default=0.1, differentiation_method=smoothed_fd)
    sindy_model.fit(x=x_rom_list, u=u_list, multiple_trajectories=True, unbias=True)
    sindy_model.print()

    # ----- SINDY FROM DEEPTIME -----
    # Deeptime's sindy cannot recognize control terms, so we masquerade u0 and u1 as the last two x

    library = PolynomialFeatures(degree=1)
    optimizer = STLSQ(threshold=0.2)
    estimator = SINDy(
        library=library,
        optimizer=optimizer,
        input_features=["x", "y"]  # The feature names are just for printing
    )

    return


def discover_objectives(ae_model):
    # Perform a basinhopper search with powell to discover the model objective in the reduced space
    # Find reduced state that minimizes:
    # 0.995 * [(x_full_decoded_5 - 303) ** 2 + (x_full_decoded_13 - 333) ** 2] +
    # 0.005 * [(u0 - 273) ** 2 + (u1 - 273) ** 2]

    def basinhopper_helper(x_rom_bh, *arg_u_bh):
        # u_bh = arg_u_bh[0]
        # u0 = np.array(u_bh).flatten()[0]
        # u1 = np.array(u_bh).flatten()[1]

        # Load the autoencoder and call its decoder to get the predicted values of x5 and x13
        x_rom_bh = np.array(x_rom_bh, dtype=np.float32)
        x_rom_bh = torch.tensor(x_rom_bh)
        x_decoded = ae_model.decode(x_rom_bh)
        x_decoded = x_decoded.flatten()

        # Compute objective value for the setpoint part
        xd5 = x_decoded[5]
        xd13 = x_decoded[13]
        obj_val = (xd5 - 303) ** 2 + (xd13 - 333) ** 2
        return obj_val

    # Configure options for the local minimizer (Powell)
    gd_options = {}
    # gd_options["maxiter"] = 2
    gd_options["disp"] = True
    # gd_options["eps"] = 1

    # We choose Powell, which is gradientless, because our input to output mapping is not differentiable
    min_kwargs = {
        "method": 'Powell',
        "options": gd_options
    }

    # x_rom_0 to x_rom_4 initial guess
    x_rom_guess = np.full(shape=(5, ), fill_value=0)

    result = optimize.basinhopping(
        func=basinhopper_helper, x0=x_rom_guess, minimizer_kwargs=min_kwargs
    )

    # Return the x_rom values that give the closest x_full relative to the setpoint
    # This is our setpoint for the reduced model
    x_rom_setpoints = np.array(result["x"]).flatten()
    print(x_rom_setpoints)
    return x_rom_setpoints


if __name__ == "__main__":
    # data = pd.read_csv("heatEq_240_trajectories_df.csv")

    # autoencoder = Autoencoder(x_dim=20, x_rom_dim=5)
    # autoencoder.fit(data)
    # with open("heatEq_autoencoder_5dim.pickle", "wb") as model:
    #     pickle.dump(autoencoder, model)

    # data = pd.read_csv("heatEq_240_trajectories_df.csv")


    # Get x_rom initial values
    # x_init = np.full(shape=(1, 20), fill_value=273)
    # x_init_df_col = []
    # for i in range(20):
    #     x_init_df_col.append("x{}".format(i))
    # x_init_df = pd.DataFrame(x_init, columns=x_init_df_col)
    # autoencoder = load_pickle("heatEq_autoencoder_5dim.pickle")
    # print(autoencoder.encode(x_init_df))
    # Initial values for x_rom:
    #      x0_rom    x1_rom    x2_rom    x3_rom    x4_rom
    #   -0.203286 -0.271189  0.407007 -0.666588 -0.234218


    # data = pd.read_csv("R47 heatEq_240_trajectories_df.csv")
    # autoencoder = load_pickle("heatEq_autoencoder_5dim.pickle")
    # sindy(autoencoder, data)

    # Sindy fit for degree 1
    # x0' = -12.625 1 + 0.915 x0 + -24.623 x1 + -3.347 x3 + -11.467 x4
    # x1' = -67.528 1 + 3232.952 x0 + -7013.098 x1 + -3960.334 x2 + 1307.525 x3 + -5009.076 x4
    # x2' = -84.901 1 + 3067.923 x0 + -6685.955 x1 + -3747.449 x2 + 1233.639 x3 + -4761.072 x4
    # x3' = -17.457 1 + 1.582 x0 + -22.597 x1 + 15.782 x2 + -6.529 x3
    # x4' = 141.380 1 + -6370.022 x0 + 13831.149 x1 + 7799.579 x2 + -2573.599 x3 + 9873.393 x4

    # Sindy fit for degree 2
    # x0' = -86.337 1 + 10526.797 x0 + -22574.634 x1 + -12834.864 x2 + 4196.453 x3 + -15982.831 x4 + 416.963 x0^2 + -610.498 x1^2 + -36.191 x3^2 + 102.916 x4^2
    # x1' = -99.895 1 + 8153.836 x0 + -17602.472 x1 + -9778.537 x2 + 3252.057 x3 + -12239.332 x4 + 295.086 x0^2 + -732.165 x1^2 + -95.043 x2^2 + -16.073 x3^2 + 183.958 x4^2
    # x2' = -141.282 1 + 10815.599 x0 + -23367.518 x1 + -12881.589 x2 + 4292.769 x3 + -16117.669 x4 + 480.143 x0^2 + -1193.190 x1^2 + -157.342 x2^2 + -22.393 x3^2 + 302.014 x4^2
    # x3' = -5.091 1 + -839.014 x0 + 1772.372 x1 + 1189.361 x2 + -373.919 x3 + 1498.993 x4 + 141.254 x0^2 + -376.704 x1^2 + -58.241 x2^2 + 0.115 x3^2 + 101.679 x4^2
    # x4' = 216.162 1 + -17273.781 x0 + 37296.934 x1 + 20687.405 x2 + -6881.169 x3 + 25891.512 x4 + -659.484 x0^2 + 1622.126 x1^2 + 208.862 x2^2 + 34.354 x3^2 + -406.950 x4^2

    # Discover setpoint for x_rom
    autoencoder = load_pickle("heatEq_autoencoder_5dim.pickle")
    discover_objectives(autoencoder)

    # Discovered setpoints for x_rom
    # [1.10888901 1.65167426 0.38852846 0.09250596 5.45466693]
    # [0.04778089 -1.00491062  1.35108469  0.80050275  3.28888439]
    # [1.12985082  1.24368333 -0.46781554  0.04821881  4.01453943]
    # [1.68571239  0.88065856 -0.21620381 -0.23563355  2.53726015] > Using this
