import datetime

from matplotlib import pyplot as plt
from pyomo.dae import *
from pyomo.environ import *

from heatEq_autoencoder import *


# THIN ROD WITH DIMENSIONALITY REDUCED FROM 20 TO 1


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

        x5_fit = dataframe_fit["x5"].to_numpy().flatten().reshape(-1, 1)
        x13_fit = dataframe_fit["x13"].to_numpy().flatten().reshape(-1, 1)
        x5_score = dataframe_score["x5"].to_numpy().flatten().reshape(-1, 1)
        x13_score = dataframe_score["x13"].to_numpy().flatten().reshape(-1, 1)

        # Try to scale everything to the same scale
        self.u0_scaler = preprocessing.MinMaxScaler()
        self.u1_scaler = preprocessing.MinMaxScaler()
        self.x_rom_scaler = preprocessing.MinMaxScaler()
        self.x5_scaler = preprocessing.MinMaxScaler()
        self.x13_scaler = preprocessing.MinMaxScaler()
        self.u0_scaler.fit(u0_fit)
        self.u1_scaler.fit(u1_fit)
        self.x_rom_scaler.fit(x_rom_fit)
        self.x5_scaler.fit(x5_fit)
        self.x13_scaler.fit(x13_fit)

        u0_fit = self.u0_scaler.transform(u0_fit)
        u0_score = self.u0_scaler.transform(u0_score)
        u1_fit = self.u1_scaler.transform(u1_fit)
        u1_score = self.u1_scaler.transform(u1_score)
        x_rom_fit = self.x_rom_scaler.transform(x_rom_fit)
        x_rom_score = self.x_rom_scaler.transform(x_rom_score)

        x5_fit = self.x5_scaler.transform(x5_fit)
        x13_fit = self.x13_scaler.transform(x13_fit)
        x5_score = self.x5_scaler.transform(x5_score)
        x13_score = self.x13_scaler.transform(x13_score)

        x_rom_fit = np.hstack((x_rom_fit, x5_fit, x13_fit))
        x_rom_score = np.hstack((x_rom_score, x5_score, x13_score))

        # We need to split x_rom and u to into a list of 240 trajectories for sindy
        # num_trajectories = 1680
        num_trajectories = 240
        num_trajectories = 400
        u0_list_fit = np.split(u0_fit, num_trajectories)
        u1_list_fit = np.split(u1_fit, num_trajectories)
        self.u_list_fit = []
        for u0_fit, u1_fit in zip(u0_list_fit, u1_list_fit):
            self.u_list_fit.append(np.hstack((u0_fit.reshape(-1, 1), u1_fit.reshape(-1, 1))))
        self.x_rom_list_fit = np.split(x_rom_fit, num_trajectories, axis=0)

        num_trajectories = 240
        num_trajectories = 100
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

        # ----- SINDY FROM PYSINDY -----
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
        sindy_model = pysindy.SINDy(t_default=0.1, feature_library=poly_library,
                                    differentiation_method=fd_drop_endpoints)
        # sindy_model = pysindy.SINDy(t_default=0.1, differentiation_method=smoothed_fd)

        # sindy_model.fit(x=x_rom, u=np.hstack((u0.reshape(-1, 1), u1.reshape(-1, 1))))
        sindy_model.fit(x=self.x_rom_list_fit, u=self.u_list_fit, multiple_trajectories=True)
        sindy_model.print()

        print("R2")
        score = sindy_model.score(x=self.x_rom_list_score, u=self.u_list_score, multiple_trajectories=True,
                                  metric=r2_score)
        print(score)
        score = sindy_model.score(x=self.x_rom_list_score, u=self.u_list_score, multiple_trajectories=True,
                                  metric=mean_squared_error)
        print("MSE")
        print(score)
        score = sindy_model.score(x=self.x_rom_list_score, u=self.u_list_score, multiple_trajectories=True,
                                  metric=mean_absolute_error)
        print("MAE")
        print(score)

        # Get x_rom initial values
        x_init = np.full(shape=(1, 20), fill_value=273)
        x_init_df_col = []
        for i in range(20):
            x_init_df_col.append("x{}".format(i))
        x_init_df = pd.DataFrame(x_init, columns=x_init_df_col)
        x_rom_init = ae_model.encode(x_init_df).to_numpy()
        x_rom_init_scaled = self.x_rom_scaler.transform(x_rom_init)
        print("x_init - scaled for sindy dont scale again")
        print(x_rom_init_scaled)

        # Initial values for x_rom (scaled for Sindy):
        #      x0_rom     x1_rom        x2_rom
        #   0.2628937  0.6858409  0.44120657

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

        # Discover setpoint for x_rom
        x_rom_setpoints = discover_objectives(ae_model)
        x_rom_setpoints_scaled = self.x_rom_scaler.transform(x_rom_setpoints)
        print("Setpoints - scaled for sindy dont scale again")
        print(x_rom_setpoints_scaled)

        print("setpoint check")
        sp_check = self.x_rom_scaler.inverse_transform(x_rom_setpoints_scaled)
        sp_check = ae_model.decode(sp_check)
        print(sp_check)

    def fit(self):
        # ----- SINDY FROM PYSINDY -----
        # Get the polynomial feature library
        # include_interaction = False precludes terms like x0x1, x2x3
        poly_library = pysindy.PolynomialLibrary(include_interaction=False, degree=2)
        fourier_library = pysindy.FourierLibrary(n_frequencies=3)
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
        data_fit = pd.read_csv("data/mpc_data19.csv")
        data_score = pd.read_csv("data/mpc_data_test9.csv")
        self.autoencoder = load_pickle("heatEq_autoencoder_1dim_constraint_no_x13.pickle")
        self.sindy = SINDYc(self.autoencoder, data_fit, data_score)
        self.sindy.fit()

        # ----- SET UP THE BASIC MODEL -----
        # Set up pyomo model structure
        self.model = ConcreteModel()
        self.model.time = ContinuousSet(bounds=(0, duration))

        # Initial state: the rod is 273 Kelvins throughout (all x_full = 273)
        # Initial values for x_rom - SCALED FOR SINDY
        x_init = [0.8421907, 0.49981415]
        # x_init = self.sindy.x_rom_scaler.transform(np.array(x_init).reshape(1, 3)).flatten()

        self.model.x0 = Var(self.model.time, initialize=x_init[0])
        self.model.x0_dot = DerivativeVar(self.model.x0, wrt=self.model.time)
        self.model.x0[0].fix(x_init[0])
        self.model.x5 = Var(self.model.time, initialize=x_init[1])
        self.model.x5_dot = DerivativeVar(self.model.x5, wrt=self.model.time)
        self.model.x5[0].fix(x_init[1])
        # self.model.x13 = Var(self.model.time, initialize=x_init[2])
        # self.model.x13_dot = DerivativeVar(self.model.x13, wrt=self.model.time)
        # self.model.x13[0].fix(x_init[2])

        # Set up controls
        self.model.u0 = Var(self.model.time, bounds=(-1, 1), initialize=-0.30186862)
        self.model.u1 = Var(self.model.time, bounds=(-1, 1), initialize=-0.33333335)

        # ODEs
        # Set up x0_dot = Ax + Bu
        def _ode_x0(m, _t):
            # return m.x0[_t] == 450901.9451 + -6.932 * m.x0[_t] + -3.261 * m.x5[_t] + 3.739 * m.x13[_t] + 0.701 * m.u0[
            #     _t] + -450902.483 * m.u1[_t]
            # return m.x0[_t] == -6.815 * m.x0[_t] + -3.198 * m.x5[_t] + 3.796 * m.x13[_t] + 0.691 * m.u0[_t] + -0.654 * \
            #        m.u1[_t]
            # return m.x0[_t] == -20.352 * m.x0[_t] + -3.238 * m.x5[_t] + 9.327 * m.x13[_t] + -1.546 * m.u0[
            #     _t] + -469466.906 * m.u1[_t] + 22.088 * m.x0[_t] ** 2 + -7.434 * m.x13[_t] ** 2 + 0.773 * m.u0[
            #            _t] ** 2 + 469468.335 * m.u1[_t] ** 2
            # return m.x0[_t] == 2.3841 + -9.251 * m.x0[_t] + 0.407 * m.u0[_t] + -2.358 * m.u1[_t]
            # return m.x0[_t] == 1.9351 + -2.279 * m.x0[_t] + -0.417 * m.x5[_t] + -2.386 * m.x13[_t] + -1.538 * m.u0[
            #     _t] + -0.653 * m.u1[_t] + -6.425 * m.x0[_t] ** 2 + -0.007 * m.x5[_t] ** 2 + 3.793 * m.x13[
            #            _t] ** 2 + 1.919 * m.u0[_t] ** 2 + -2.198 * m.u1[_t] ** 2
            # return m.x0[_t] == 7.0611 + -33.122 * m.x0[_t] + -1.467 * m.x5[_t] + -2.334 * m.x13[_t] + -0.633 * m.u0[_t] + 471.925 * m.u1[
            #     _t] + 42.327 * m.x0[_t] ** 2 + 4.274 * m.x5[_t] ** 2 + 5.857 * m.x13[_t] ** 2 + 1.143 * m.u0[
            #     _t] ** 2 + -939.626 * m.u1[_t] ** 2 + -23.181 * m.x0[_t] ** 3 + -4.619 * m.x5[_t] ** 3 + -7.496 * m.x13[
            #     _t] ** 3 + -0.339 * m.u0[_t] ** 3 + 466.451 * m.u1[_t] ** 3
            return m.x0_dot[_t] == -1.0781 + -11.079 * m.x0[_t] + -0.489 * m.x5[_t] + -2.235 * m.u0[_t] + 9.458 * m.u1[
                _t] + 5.421 * m.x0[_t] ** 2 + -0.382 * m.x5[_t] ** 2 + 1.740 * m.u0[_t] ** 2 + -7.394 * m.u1[_t] ** 2

        self.model.x0_ode = Constraint(self.model.time, rule=_ode_x0)

        # def _ode_x0_1(m, _t):
        #     return m.x0[_t] == -0.6301 + -6.045 * m.x0[_t] + -1.854 * m.x5[_t] + 2.427 * m.x13[_t]
        #
        # self.model.x0_ode_1 = Constraint(self.model.time, rule=_ode_x0_1)

        def _ode_x5(m, _t):
            # return m.x5[_t] == -1993884.7751 + 46.891 * m.x0[_t] + 7.335 * m.x5[_t] + 28.264 * m.x13[_t] + -0.803 * \
            #        m.u0[_t] + 1993849.322 * m.u1[_t]
            # return m.x5[_t] == 46.374 * m.x0[_t] + 7.060 * m.x5[_t] + 28.011 * m.x13[_t] + -0.761 * m.u0[_t] + -34.936 * \
            #        m.u1[_t]
            # return m.x5[_t] == 83.379 * m.x0[_t] + 274.531 * m.x5[_t] + 39.527 * m.x13[_t] + -0.631 * m.u0[
            #     _t] + 1062089.170 * m.u1[_t] + -49.330 * m.x0[_t] ** 2 + -134.317 * m.x5[_t] ** 2 + 10.787 * m.u0[
            #            _t] ** 2 + -1062268.944 * m.u1[_t] ** 2
            # return m.x5[_t] == 11.2391 + 6.828 * m.x0[_t] + -23.043 * m.x5[_t] + -0.940 * m.x13[_t] + 3.834 * m.u0[
            #     _t] + -4.199 * m.u1[_t] + -6.102 * m.x0[_t] ** 2 + 9.771 * m.x5[_t] ** 2 + 3.752 * m.x13[
            #            _t] ** 2 + -1.134 * m.u0[_t] ** 2 + 3.291 * m.u1[_t] ** 2
            # return m.x5[_t] == -22.2111 + 235.439 * m.x0[_t] + -14.838 * m.x5[_t] + 3.346 * m.x13[_t] + -7.400 * m.u0[
            #     _t] + 440.783 * m.u1[_t] + -382.624 * m.x0[_t] ** 2 + -23.791 * m.x5[_t] ** 2 + -28.655 * m.x13[
            #            _t] ** 2 + 15.882 * m.u0[_t] ** 2 + -981.132 * m.u1[_t] ** 2 + 182.571 * m.x0[_t] ** 3 + 36.605 * \
            #        m.x5[_t] ** 3 + 64.684 * m.x13[_t] ** 3 + -4.763 * m.u0[_t] ** 3 + 525.622 * m.u1[_t] ** 3

            # return m.x5[_t] == 13.7821 + -2.589 * m.x0[_t] + -12.604 * m.x5[_t] + -1.486 * m.x13[_t]
            return m.x5_dot[_t] == 18.0571 + -3.176 * m.x0[_t] + -22.595 * m.x5[_t] + 4.502 * m.u0[_t] + -8.747 * m.u1[
                _t] + -3.103 * m.x0[_t] ** 2 + 8.711 * m.x5[_t] ** 2 + -0.612 * m.u0[_t] ** 2 + 4.411 * m.u1[_t] ** 2

        self.model.x5_ode = Constraint(self.model.time, rule=_ode_x5)

        # def _ode_x13(m, _t):
        #     # return m.x13[_t] == -578551.0601 + 12.556 * m.x0[_t] + 5.977 * m.x5[_t] + -3.291 * m.x13[_t] + -3.559 * \
        #     #        m.u0[_t] + 578548.582 * m.u1[_t]
        #     # return m.x13[_t] == 12.406 * m.x0[_t] + 5.897 * m.x5[_t] + -3.365 * m.x13[_t] + -3.547 * m.u0[_t] + -2.327 * \
        #     #        m.u1[_t]
        #
        #     # return m.x13[_t] == -13.691 * m.x0[_t] + 64.078 * m.x5[_t] + 76.132 * m.x13[_t] + -4.707 * m.u0[
        #     #     _t] + -1118829.158 * m.u1[_t] + 71.648 * m.x0[_t] ** 2 + -29.466 * m.x5[_t] ** 2 + -51.899 * m.x13[
        #     #            _t] ** 2 + 2.724 * m.u0[_t] ** 2 + 1118770.586 * m.u1[_t] ** 2
        #     # return m.x13[_t] == 14.9331 + -13.246 * m.x0[_t] + -0.045 * m.x5[_t] + -16.837 * m.x13[_t] + -2.412 * m.u0[
        #     #     _t] + 0.287 * m.u1[_t] + 3.466 * m.x0[_t] ** 2 + 0.648 * m.x5[_t] ** 2 + -1.224 * m.x13[
        #     #            _t] ** 2 + 0.503 * m.u0[_t] ** 2 + 2.388 * m.u1[_t] ** 2
        #     return m.x13[_t] == 4.8081 + 39.179 * m.x0[_t] + 1.478 * m.x5[_t] + -19.619 * m.x13[_t] + -2.166 * m.u0[
        #         _t] + -195.141 * m.u1[_t] + -74.690 * m.x0[_t] ** 2 + -5.749 * m.x5[_t] ** 2 + 5.290 * m.x13[
        #                _t] ** 2 + -1.545 * m.u0[_t] ** 2 + 379.426 * m.u1[_t] ** 2 + 36.106 * m.x0[_t] ** 3 + 7.024 * \
        #            m.x5[_t] ** 3 + 6.887 * m.x13[_t] ** 3 + 2.775 * m.u0[_t] ** 3 + -184.281 * m.u1[_t] ** 3
        #     # return m.x13[_t] == 16.7261 + -12.293 * m.x0[_t] + -1.037 * m.x5[_t] + -15.636 * m.x13[_t]
        #
        # self.model.x13_ode = Constraint(self.model.time, rule=_ode_x13)

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
        setpoint_weight = 1
        controller_weight = 1 - setpoint_weight

        # Lagrangian cost
        def _Lagrangian(m, _t):
            return m.L_dot[_t] \
                   == setpoint_weight * ((m.x0[_t] - 0.0010463) ** 2)
            # + (m.x13[_t] - 1) ** 2)
            # + controller_weight * ((m.u0[_t] - -0.30186862) ** 2 + (m.u1[_t] - -0.33333335) ** 2)

        self.model.L_integral = Constraint(self.model.time, rule=_Lagrangian)

        # Constraint for the element at the 1/3 position: path constraint for additional challenge
        # 5t^2 + 10t + 293
        def _constraint_x5(m, _t):
            return m.x5[_t] <= (5 * _t ** 2) + (10 * _t) + 293

        self.model.constraint_x5 = Constraint(self.model.time, rule=_constraint_x5)

        # Objective function is to minimize the Lagrangian cost integral
        def _objective(m):
            return m.L[m.time.last()] - m.L[0]

        self.model.objective = Objective(rule=_objective, sense=minimize)

        self.autoencoder = load_pickle("heatEq_autoencoder_1dim_constraint_no_x13.pickle")

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
        # mpc_solver.options['DualReductions'] = 0
        # mpc_solver.options['InfUnbdInfo'] = 1

        mpc_results = mpc_solver.solve(self.model)
        self.model.display()

        print(mpc_solver.options['FarkasProof'])
        print(mpc_solver.options['FarkasDual'])

        return mpc_results

    def simulate_system_sindy_controls(self):
        timesteps = [timestep / 10 for timestep in range(11)]

        u0_nn = np.full(shape=(11,), fill_value=-0.30186862)
        u1_nn = np.full(shape=(11,), fill_value=-0.33333335)

        self.model.var_input = Suffix(direction=Suffix.LOCAL)
        # Create a dictionary of piecewise linear controller actions
        u0_nn_profile = {timesteps[i]: u0_nn[i] for i in range(len(timesteps))}
        u1_nn_profile = {timesteps[i]: u1_nn[i] for i in range(len(timesteps))}

        # Update the control sequence to Pyomo
        self.model.var_input[self.model.u0] = u0_nn_profile
        self.model.var_input[self.model.u1] = u1_nn_profile

        sim = Simulator(self.model, package="casadi")
        tsim, profiles = sim.simulate(numpoints=11, varying_inputs=self.model.var_input)

        print("PROFILES")
        print(profiles)

        # For some reason both tsim and profiles contain duplicates
        # Use pandas to drop the duplicates first
        # profiles columns: x0, x1, ..., x19, L
        temp_dict = {"t": tsim}
        for j in range(3):
            temp_dict["x{}".format(j)] = profiles[:, j]
        temp_dict["L"] = profiles[:, 3]

        deduplicate_df = pd.DataFrame(temp_dict)
        deduplicate_df = deduplicate_df.round(4)
        deduplicate_df.drop_duplicates(ignore_index=True, inplace=True)

        # Make dataframe from the simulator results
        t = deduplicate_df["t"]
        x = []
        for j in range(3):
            x.append(deduplicate_df["x{}".format(j)])

        # Note: at this point, x is a list of 20 pandas series, each series has 11 rows
        # Check duplicates were removed correctly
        assert len(t) == 11

        # Scale back everything
        u0 = self.sindy.u0_scaler.inverse_transform(np.array(u0_nn).reshape(-1, 1)).flatten()
        u1 = self.sindy.u1_scaler.inverse_transform(np.array(u1_nn).reshape(-1, 1)).flatten()
        x_temp = self.sindy.x_rom_scaler.inverse_transform(np.array(x).reshape(-1, 3))
        x_temp = self.autoencoder.decode(x_temp)

        print(x)

        # Make dataframe from the final simulator results
        t = deduplicate_df["t"]
        x = []
        for i in range(20):
            x.append(x_temp[:, i])
        L = deduplicate_df["L"]
        # u0 = u0_nn
        # u1 = u1_nn
        inst_cost = []
        ctg = []

        for time in range(len(t)):
            # Instantaneous cost is L[t1] - L[t0]
            if time == 10:
                inst_cost.append(0)
            else:
                inst_cost.append(L[time + 1] - L[time])

        # Calculate cost to go
        for time in range(len(t)):
            ctg.append(inst_cost[time])
        # Sum backwards from tf
        for time in reversed(range(len(t) - 1)):
            ctg[time] += ctg[time + 1]

        # Calculate path violations
        path = [x[5][int(time * 10)] - 313 for time in t]
        path_violation = []
        for p in path:
            if max(path) > 0:
                path_violation.append(max(path))
            else:
                path_violation.append(p)

        temp_dict = {"t": t}
        for i in range(20):
            temp_dict["x{}".format(i)] = x[i]
        temp_dict["u0"] = u0
        temp_dict["u1"] = u1
        temp_dict["L"] = L
        temp_dict["inst_cost"] = inst_cost
        temp_dict["ctg"] = ctg
        temp_dict["path_diff"] = path_violation

        print(temp_dict)

        sindy_sim_results_df = pd.DataFrame(temp_dict)
        sindy_sim_results_df_dropped_tf = sindy_sim_results_df.drop(index=10)

        return sindy_sim_results_df, sindy_sim_results_df_dropped_tf

    def parse_mpc_results(self):
        # Each t, x0, x1, x2, etc, U, L, instantaneous cost, cost to go, should be a column
        # Label them and return a pandas dataframe
        t = []
        u0 = []
        u1 = []
        x5 = []
        x13 = []
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
                x5.append(value(self.model.x5[time]))
                # x13.append(value(self.model.x13[time]))
                L.append(value(self.model.L[time]))

        # Make sure all 11 time steps are recorded; this was problematic due to Pyomo's float indexing
        assert len(t) == 11

        # Scale back everything
        u0 = self.sindy.u0_scaler.inverse_transform(np.array(u0).reshape(-1, 1)).flatten()
        u1 = self.sindy.u1_scaler.inverse_transform(np.array(u1).reshape(-1, 1)).flatten()
        x5 = self.sindy.x5_scaler.inverse_transform(np.array(x5).reshape(-1, 1)).flatten()
        # x13 = self.sindy.x13_scaler.inverse_transform(np.array(x13).reshape(-1, 1)).flatten()

        for time in range(len(t)):
            # Instantaneous cost is L[t1] - L[t0]
            if time == 0:
                inst_cost.append(0)
            else:
                inst_cost.append(L[time] - L[time - 1])

        # Calculate cost to go
        for time in range(len(t)):
            ctg.append(inst_cost[time])
        # Sum backwards from tf
        for time in reversed(range(len(t) - 1)):
            ctg[time] += ctg[time + 1]

        df_data = {"t": t}
        df_data["x5"] = x5
        # df_data["x13"] = x13
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
        # x13 = dataframe["x13"]
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
        axs[0].plot(t, np.full(shape=(t.size,), fill_value=303), "--", label="Setpoint for $x_5$")
        axs[0].legend()

        # axs[1].plot(t, x13, label="$x_{13}$")
        # axs[1].plot(t, np.full(shape=(t.size, ), fill_value=333), "--", label="Setpoint for $x_{13}$")
        # axs[1].legend()

        axs[2].step(t, u0, label="$u_0$")
        axs[2].step(t, u1, label="$u_1$")
        axs[2].legend()

        fig.suptitle("Sindy MPC performance as seen in the ROM system\n "
                     "Reduced from 20 to 1 state using autoencoder\n"
                     "with additional 2 unreduced constraints"
                     .format(num_rounds, num_run_in_round, total_cost, cst_status))
        plt.xlabel("Time")

        # Save plot with autogenerated filename
        # svg_filename = results_folder + "Round {} Run {} Cost {} Constraint {}"\
        #     .format(num_rounds, num_run_in_round, total_cost, cst_status) + ".svg"
        plt.savefig(fname="Sindy.svg", format="svg")

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
    main_res["u0"].to_csv("u0.csv")
    main_res["u1"].to_csv("u1.csv")
    heatEq_system.plot(main_res)

    # heatEq_system = HeatEqSimulator()
    # main_res, _ = heatEq_system.simulate_system_sindy_controls()
    # heatEq_system.plot(main_res)
