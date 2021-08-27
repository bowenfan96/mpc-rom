import csv
import datetime
import pickle
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyomo.environ import *
from pyomo.dae import *
from pyomo.solvers import *

from heatEq_autoencoder import *


# THIN ROD WITH DIMENSIONALITY REDUCED FROM 20 TO 3


class SINDYc:
    def __init__(self, ae_model, dataframe_fit, dataframe_score):
        # x_rom from autoencoder, returned as dataframe with shape (2400, 10)
        x_rom_fit = ae_model.encode(dataframe_fit).to_numpy()
        x_rom_score = ae_model.encode(dataframe_score).to_numpy()

        # Sindy needs to know the controller signals
        u0_fit = dataframe_fit["u0"].to_numpy().flatten().reshape(-1, 1)
        u1_fit = dataframe_fit["u1"].to_numpy().flatten().reshape(-1, 1)
        u0_score = dataframe_score["u0"].to_numpy().flatten().reshape(-1, 1)
        u1_score = dataframe_score["u1"].to_numpy().flatten().reshape(-1, 1)

        # Try to scale everything to the same scale
        self.u0_scaler = preprocessing.MinMaxScaler()
        self.u1_scaler = preprocessing.MinMaxScaler()
        self.x_rom_scaler = preprocessing.MinMaxScaler()
        self.u0_scaler.fit(u0_fit)
        self.u1_scaler.fit(u1_fit)
        self.x_rom_scaler.fit(x_rom_fit)

        u0_fit = self.u0_scaler.transform(u0_fit)
        u0_score = self.u0_scaler.transform(u0_score)
        u1_fit = self.u1_scaler.transform(u1_fit)
        u1_score = self.u1_scaler.transform(u1_score)
        x_rom_fit = self.x_rom_scaler.transform(x_rom_fit)
        x_rom_score = self.x_rom_scaler.transform(x_rom_score)

        # We need to split x_rom and u to into a list of 240 trajectories for sindy
        # num_trajectories = 1680
        num_trajectories = 240
        u0_list_fit = np.split(u0_fit, num_trajectories)
        u1_list_fit = np.split(u1_fit, num_trajectories)
        self.u_list_fit = []
        for u0_fit, u1_fit in zip(u0_list_fit, u1_list_fit):
            self.u_list_fit.append(np.hstack((u0_fit.reshape(-1, 1), u1_fit.reshape(-1, 1))))
        self.x_rom_list_fit = np.split(x_rom_fit, num_trajectories, axis=0)

        num_trajectories = 240
        u0_list_score = np.split(u0_score, num_trajectories)
        u1_list_score = np.split(u1_score, num_trajectories)
        self.u_list_score = []
        for u0_score, u1_score in zip(u0_list_score, u1_list_score):
            self.u_list_score.append(np.hstack((u0_score.reshape(-1, 1), u1_score.reshape(-1, 1))))
        self.x_rom_list_score = np.split(x_rom_score, num_trajectories, axis=0)

        # Get scaling parameters for manual inverse transform
        # print("Sindy scaler data min")
        # print(self.x_rom_scaler.data_min_)
        # self.sindy_dataMin = np.array([0.4032871, -0.9662344, -1.677114]).flatten()
        # print("Sindy scaler data max")
        # print(self.x_rom_scaler.data_max_)
        # self.sindy_dataMax = np.array([0.94944113, -0.47337604, -0.6375582]).flatten()

        # Get autoencoder scaling parameters
        # print("Autoencoder scaler data min")
        # print(ae_model.scaler_x.data_min_)
        # self.autoencoder_dataMin = np.array([182.7997, 193.3699, 202.7192, 210.8632, 217.8229, 223.623,  228.2624, 231.7888, 234.2273, 235.5974, 235.9124, 235.1791, 233.3824, 230.4251, 226.341,  221.1054, 214.6942, 207.0872, 198.271,  186.1254]).flatten()
        # print("Autoencoder scaler data max")
        # print(ae_model.scaler_x.data_max_)
        # self.autoencoder_dataMax = np.array([437.6874, 410.5294, 389.3565, 372.7515, 359.7538, 349.6972, 342.1139, 336.6767, 333.1643, 331.4397, 331.4374, 333.1576, 336.6663, 342.1009, 349.6827, 359.7426, 372.7442, 389.352,  410.5268, 437.6862]).flatten()

        # Bounds for u0
        print("u0 bounds")
        print(self.u0_scaler.transform(np.array(173).reshape(-1, 1)))
        print(self.u0_scaler.transform(np.array(473).reshape(-1, 1)))
        # Bounds for u1
        print("u1 bounds")
        print(self.u1_scaler.transform(np.array(173).reshape(-1, 1)))
        print(self.u1_scaler.transform(np.array(473).reshape(-1, 1)))
        # 273
        print(self.u0_scaler.transform(np.array(273).reshape(-1, 1)))
        print(self.u1_scaler.transform(np.array(273).reshape(-1, 1)))

    def fit(self):
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

        sindy_model.fit(x=self.x_rom_list_fit, u=self.u_list_fit, multiple_trajectories=True)
        sindy_model.print()
        score = sindy_model.score(x=self.x_rom_list_score, u=self.u_list_score, multiple_trajectories=True)
        print(score)


class HeatEqSimulator:
    def __init__(self, duration=1):
        # Unique ID for savings csv and plots, based on model timestamp
        self.uid = datetime.datetime.now().strftime("%d-%H.%M.%S.%f")[:-3]
        print("Model uid", self.uid)

        # ----- SET UP SINDY -----
        data_fit = pd.read_csv("data/sindy_fit_data.csv")
        data_score = pd.read_csv("data/sindy_validation_data.csv")
        self.autoencoder = load_pickle("heatEq_autoencoder_3dim_elu_mse_0.000498.pickle")
        self.sindy = SINDYc(self.autoencoder, data_fit, data_score)
        self.sindy.fit()

        # ----- SET UP THE BASIC MODEL -----
        # Set up pyomo model structure
        self.model = ConcreteModel()
        self.model.time = ContinuousSet(bounds=(0, duration))

        # Initial state: the rod is 273 Kelvins throughout (all x_full = 273)
        # Initial values for x_rom - SCALED FOR SINDY
        x_init = [0.632633, -0.405015, 0.087859]
        x_init = self.sindy.x_rom_scaler.transform(np.array(x_init).reshape(1, 3)).flatten()

        self.model.x0 = Var(self.model.time)
        self.model.x0_dot = DerivativeVar(self.model.x0, wrt=self.model.time)
        self.model.x0[0].fix(x_init[0])
        self.model.x1 = Var(self.model.time)
        self.model.x1_dot = DerivativeVar(self.model.x1, wrt=self.model.time)
        self.model.x1[0].fix(x_init[1])
        self.model.x2 = Var(self.model.time)
        self.model.x2_dot = DerivativeVar(self.model.x2, wrt=self.model.time)
        self.model.x2[0].fix(x_init[2])

        # Set up controls
        self.model.u0 = Var(self.model.time, bounds=(0, 1))
        self.model.u1 = Var(self.model.time, bounds=(0, 1))

        # ODEs
        # Set up x0_dot = Ax + Bu
        def _ode_x0(m, _t):
            return m.x0_dot[_t] == 7.8731 + -5.435* self.model.x0[_t] + -7.725* self.model.x1[_t] + -6.196* self.model.x2[_t] + 1.014* self.model.u0[_t] + 2.843* self.model.u1[_t] + -0.161* self.model.x0[_t]**2 + 1.419* self.model.x1[_t]**2 + -1.618* self.model.x2[_t]**2 + -0.166* self.model.u0[_t]**2 + 0.013 *self.model.u1[_t]**2
            # return m.x0_dot[_t] == 7.8731 + -5.435* self.model.x0[_t] + -7.725* self.model.x1[_t] + -6.196* self.model.x2[_t] + 2.014* self.model.u0[_t] + 1.843* self.model.u1[_t] + -0.161* self.model.x0[_t]**2 + 1.419* self.model.x1[_t]**2 + -1.618* self.model.x2[_t]**2 + -0.166* self.model.u0[_t]**2 + 0.513 *self.model.u1[_t]**2
        self.model.x0_ode = Constraint(self.model.time, rule=_ode_x0)

        # Set up x1_dot to x18_dot = Ax only
        def _ode_x1(m, _t):
            return m.x1_dot[_t] == -10.6991 + 10.793* self.model.x0[_t] + 9.294* self.model.x1[_t] + 6.198* self.model.x2[_t] + 2.789* self.model.u0[_t] + -3.086* self.model.u1[_t] + -2.686* self.model.x0[_t]**2 + -0.930* self.model.x1[_t]**2 + 1.490* self.model.x2[_t]**2 + -0.999* self.model.u0[_t]**2 + 0.079 *self.model.u1[_t]**2
            # return m.x1_dot[_t] == -10.6991 + 10.793* self.model.x0[_t] + 9.294* self.model.x1[_t] + 6.198* self.model.x2[_t] + 1.789* self.model.u0[_t] + -3.086* self.model.u1[_t] + -2.686* self.model.x0[_t]**2 + -0.930* self.model.x1[_t]**2 + 1.490* self.model.x2[_t]**2 + -0.999* self.model.u0[_t]**2 + -1.779 *self.model.u1[_t]**2
        self.model.x1_ode = Constraint(self.model.time, rule=_ode_x1)

        def _ode_x2(m, _t):
            return m.x2_dot[_t] == -0.6151 + 0.955* self.model.x0[_t] + 0.906* self.model.x1[_t] + 2.317* self.model.x2[_t] + -6.927* self.model.u0[_t] + 1.063* self.model.x0[_t]**2 + -0.930* self.model.x1[_t]**2 + 0.264* self.model.x2[_t]**2 + 1.581* self.model.u0[_t]**2 + 0.677 *self.model.u1[_t]**2
            # return m.x2_dot[_t] == -0.6151 + 0.955* self.model.x0[_t] + 0.906* self.model.x1[_t] + 2.317* self.model.x2[_t] + -4.927* self.model.u0[_t] + 1.063* self.model.x0[_t]**2 + -0.930* self.model.x1[_t]**2 + 0.264* self.model.x2[_t]**2 + 1.581* self.model.u0[_t]**2 + 0.677 *self.model.u1[_t]**2
        self.model.x2_ode = Constraint(self.model.time, rule=_ode_x2)

        # Lagrangian cost
        self.model.L = Var(self.model.time)
        self.model.L_dot = DerivativeVar(self.model.L, wrt=self.model.time)
        self.model.L[0].fix(0)

        # ----- OBJECTIVE AND COST FUNCTION -----
        # Objective:
        # We want to heat element 6 (x[5]) at the 1/3 position to 30 C, 303 K
        # And element 14 (x[13]) at the 2/3 position to 60 C, 333 K
        # We would like to minimize the controller costs too, in terms of how much heating or cooling is applied

        # Define weights for setpoint and controller objectives
        setpoint_weight = 0.995
        controller_weight = 1 - setpoint_weight

        x_rom_setpoints = np.array([0.70213366, 0.211213, 0.98931336]).flatten()

        # Lagrangian cost
        def _Lagrangian(m, _t):
            return m.L_dot[_t] \
                   == setpoint_weight * ((m.x0[_t] - x_rom_setpoints[0]) ** 2
                                         + (m.x1[_t] - x_rom_setpoints[1]) ** 2
                                         + (m.x2[_t] - x_rom_setpoints[2]) ** 2) \
                   + controller_weight * ((m.u0[_t] - 0.33340985) ** 2 + (m.u1[_t] - 0.33314682) ** 2)
        self.model.L_integral = Constraint(self.model.time, rule=_Lagrangian)

        # Objective function is to minimize the Lagrangian cost integral
        def _objective(m):
            return m.L[m.time.last()] - m.L[0]
        self.model.objective = Objective(rule=_objective, sense=minimize)

        self.autoencoder = load_pickle("heatEq_autoencoder_3dim_elu_mse_0.000498.pickle")

        # # Load extracted weights
        # w1 = np.load("autoencoder_weights_biases/input_weight.npy")
        # w2 = np.load("autoencoder_weights_biases/h1_weight.npy")
        # w3 = np.load("autoencoder_weights_biases/h2_weight.npy")
        # w4 = np.load("autoencoder_weights_biases/h3_weight.npy")
        # # Load extracted biases
        # b1 = np.load("autoencoder_weights_biases/input_bias.npy")
        # b2 = np.load("autoencoder_weights_biases/h1_bias.npy")
        # b3 = np.load("autoencoder_weights_biases/h2_bias.npy")
        # b4 = np.load("autoencoder_weights_biases/h3_bias.npy")
        #
        # W = [w1, w2, w3, w4]
        # B = [b1, b2, b3, b4]
        #
        # # Constraint for x5 <= 313 K
        # def _constraint_x5(m, _t):
        #     # Array of pyomo model variables
        #     x_hat = np.array([m.x0[_t], m.x1[_t], m.x2[_t]]).reshape(1, 3)
        #
        #     # Forward pass
        #     x_hat = x_hat @ W[0] + B[0]
        #     for row in range(x_hat.shape[0]):
        #         for col in range(x_hat.shape[1]):
        #             x_hat[row, col] = tanh(x_hat[row, col])
        #
        #     x_hat = x_hat @ W[1] + B[1]
        #     for row in range(x_hat.shape[0]):
        #         for col in range(x_hat.shape[1]):
        #             x_hat[row, col] = tanh(x_hat[row, col])
        #
        #     x_hat = x_hat @ W[2] + B[2]
        #     for row in range(x_hat.shape[0]):
        #         for col in range(x_hat.shape[1]):
        #             x_hat[row, col] = tanh(x_hat[row, col])
        #
        #     x_hat = x_hat @ W[3] + B[3]
        #
        #     x_hat = np.array(x_hat).flatten().reshape(1, 20)
        #     # x_hat = self.autoencoder.scaler_x.inverse_transform(x_hat)
        #     x_hat = x_hat.flatten()
        #     return x_hat[5] <= 313

            # sindy scaler_x.inverse_transform
            # return m.x0[_t] <= 313
            # print(self.autoencoder.decode(np.hstack((value(m.x0[_t]), value(m.x1[_t]), value(m.x2[_t])))))
            # print(self.autoencoder.decode(np.hstack((value(m.x0[_t]), value(m.x1[_t]), value(m.x2[_t])))) <= 313)
            # return self.autoencoder.decode_pyomo(m.x0[_t], m.x1[_t], m.x2[_t])


        # self.model.constraint_x5 = Constraint(self.model.time, rule=_constraint_x5)

        # ----- DISCRETIZE THE MODEL INTO FINITE ELEMENTS -----
        # We need to discretize before adding ODEs in matrix form
        # We fix finite elements at 10, collocation points at 4, controls to be piecewise linear
        discretizer = TransformationFactory("dae.collocation")
        discretizer.apply_to(self.model, nfe=10, ncp=4, scheme="LAGRANGE-RADAU")

        # Make controls piecewise linear
        discretizer.reduce_collocation_points(self.model, var=self.model.u0, ncp=1, contset=self.model.time)
        discretizer.reduce_collocation_points(self.model, var=self.model.u1, ncp=1, contset=self.model.time)

        print("Done init")

        return

    def mpc_control(self):
        mpc_solver = SolverFactory("ipopt", tee=True)
        # mpc_solver.options['max_iter'] = 10000
        mpc_results = mpc_solver.solve(self.model)
        self.model.display()

        return mpc_results

    def parse_mpc_results(self):
        # Each t, x0, x1, x2, etc, U, L, instantaneous cost, cost to go, should be a column
        # Label them and return a pandas dataframe
        t = []
        # x is a list of lists
        x = []
        u0 = []
        u1 = []
        L = []
        inst_cost = []
        ctg = []

        # Record data at the intervals of finite elements only (0.1s), do not include collocation points
        timesteps = [timestep / 10 for timestep in range(11)]
        for time in self.model.time:
            if time in timesteps:
                t.append(time)
                u0.append(value(self.model.u0[time]))
                u1.append(value(self.model.u1[time]))
                L.append(value(self.model.L[time]))

                # Get all the x values
                temp_x = []
                temp_x.append(value(self.model.x0[time]))
                temp_x.append(value(self.model.x1[time]))
                temp_x.append(value(self.model.x2[time]))
                x.append(temp_x)

        # Make sure all 11 time steps are recorded; this was problematic due to Pyomo's float indexing
        assert len(t) == 11

        # Scale back everything
        u0 = self.sindy.u0_scaler.inverse_transform(np.array(u0).reshape(-1, 1)).flatten()
        u1 = self.sindy.u1_scaler.inverse_transform(np.array(u1).reshape(-1, 1)).flatten()
        x = self.sindy.x_rom_scaler.inverse_transform(np.array(x))
        x = self.autoencoder.decode(x)

        print(x)

        for time in range(len(t)):
            # Instantaneous cost is L[t1] - L[t0]
            if time == 0:
                inst_cost.append(0)
            else:
                inst_cost.append(L[time] - L[time-1])

        # Calculate cost to go
        for time in range(len(t)):
            ctg.append(inst_cost[time])
        # Sum backwards from tf
        for time in reversed(range(len(t) - 1)):
            ctg[time] += ctg[time + 1]

        x = np.array(x)

        df_data = {"t": t}
        for x_idx in range(20):
            df_data["x{}".format(x_idx)] = x[:, x_idx]
        df_data["u0"] = u0
        df_data["u1"] = u1
        df_data["L"] = L
        df_data["inst_cost"] = inst_cost
        df_data["ctg"] = ctg

        mpc_results_df = pd.DataFrame(df_data)
        mpc_results_df_dropped_t0 = mpc_results_df.drop(index=0)

        return mpc_results_df, mpc_results_df_dropped_t0

    def plot(self, dataframe, num_rounds=0, num_run_in_round=0):
        t = dataframe["t"]
        ctg = dataframe["ctg"]

        # Plot x[5] and x[13], the elements whose temperatures we are trying to control
        x5 = dataframe["x5"]
        x13 = dataframe["x13"]
        u0 = dataframe["u0"]
        u1 = dataframe["u1"]

        if "path_diff" in dataframe.columns:
            cst = dataframe["path_diff"]
            if cst.max() <= 0:
                cst_status = "Pass"
            else:
                cst_status = "Fail"
        else:
            cst_status = "None"

        # Check that the cost to go is equal to the Lagrangian cost integral
        assert np.isclose(ctg.iloc[0], dataframe["L"].iloc[-1], atol=0.01)
        total_cost = round(ctg.iloc[0], 3)

        fig, axs = plt.subplots(3, constrained_layout=True)
        fig.set_size_inches(5, 10)

        axs[0].plot(t, x5, label="$x_5$")
        axs[0].legend()

        axs[1].plot(t, x13, label="$x_{13}$")
        axs[1].legend()

        axs[2].step(t, u0, label="$u_0$")
        axs[2].step(t, u1, label="$u_1$")
        axs[2].legend()

        fig.suptitle("Control policy and system state after {} rounds of training \n "
                     "Run {}: Cost = {}, Constraint = {}"
                     .format(num_rounds, num_run_in_round, total_cost, cst_status))
        plt.xlabel("Time")

        # Save plot with autogenerated filename
        # svg_filename = results_folder + "Round {} Run {} Cost {} Constraint {}"\
        #     .format(num_rounds, num_run_in_round, total_cost, cst_status) + ".svg"
        # plt.savefig(fname=svg_filename, format="svg")

        plt.show()
        # plt.close()
        return


def load_pickle(filename):
    with open(filename, "rb") as model:
        pickled_nn_model = pickle.load(model)
    print("Pickle loaded: " + filename)
    return pickled_nn_model


if __name__ == "__main__":
    heatEq_system = HeatEqSimulator()
    print(heatEq_system.mpc_control())
    main_res, _ = heatEq_system.parse_mpc_results()
    heatEq_system.plot(main_res)
