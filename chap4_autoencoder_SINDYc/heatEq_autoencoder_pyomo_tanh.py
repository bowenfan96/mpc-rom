import datetime

from pyomo.dae import *
from pyomo.environ import *
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from heatEq_autoencoder_tanh import *


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
        num_trajectories = 400
        u0_list_fit = np.split(u0_fit, num_trajectories)
        u1_list_fit = np.split(u1_fit, num_trajectories)
        self.u_list_fit = []
        for u0_fit, u1_fit in zip(u0_list_fit, u1_list_fit):
            self.u_list_fit.append(np.hstack((u0_fit.reshape(-1, 1), u1_fit.reshape(-1, 1))))
        self.x_rom_list_fit = np.split(x_rom_fit, num_trajectories, axis=0)

        num_trajectories = 100
        u0_list_score = np.split(u0_score, num_trajectories)
        u1_list_score = np.split(u1_score, num_trajectories)
        self.u_list_score = []
        for u0_score, u1_score in zip(u0_list_score, u1_list_score):
            self.u_list_score.append(np.hstack((u0_score.reshape(-1, 1), u1_score.reshape(-1, 1))))
        self.x_rom_list_score = np.split(x_rom_score, num_trajectories, axis=0)

        # Get scaling parameters for manual inverse transform
        print("Sindy scaler data min")
        print(self.x_rom_scaler.data_min_)
        self.sindy_dataMin = np.array([-1.2632675]).flatten()
        print("Sindy scaler data max")
        print(self.x_rom_scaler.data_max_)
        self.sindy_dataMax = np.array([1.2731746]).flatten()
        print("Sindy scale")
        print(self.x_rom_scaler.scale_)
        print("Sindy self.min_")
        print(self.x_rom_scaler.min_)

        # Get autoencoder scaling parameters
        print("Autoencoder scaler data min")
        print(ae_model.scaler_x.data_min_)
        self.autoencoder_dataMin = np.array([243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243,
                                             243, 243, 243, 243, 243, 243]).flatten()
        print("Autoencoder scaler data max")
        print(ae_model.scaler_x.data_max_)
        self.autoencoder_dataMax = np.array([400.92786, 366.24084, 338.71667, 317.02072, 303.4187, 303.0223, 303.5431
                                                , 304.84406, 306.95587, 309.94135, 313.89136, 318.93024, 325.22433,
                                             332.9931
                                                , 342.5486, 354.29337, 368.7209, 386.54282, 408.7622,
                                             436.84055]).flatten()
        print("Autoencoder scale")
        print(ae_model.scaler_x.scale_)
        print("Autoencoder self.min_")
        print(ae_model.scaler_x.min_)

        # Bounds for u0
        print("u0 bounds")
        print(self.u0_scaler.transform(np.array(173).reshape(-1, 1)))
        print(self.u0_scaler.transform(np.array(473).reshape(-1, 1)))
        # Bounds for u1
        print("u1 bounds")
        print(self.u1_scaler.transform(np.array(173).reshape(-1, 1)))
        print(self.u1_scaler.transform(np.array(473).reshape(-1, 1)))

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
        # score = sindy_model.score(x=self.x_rom_list_score, u=self.u_list_score, multiple_trajectories=True)
        # print(score)
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


class HeatEqSimulator:
    def __init__(self, duration=1):
        # Unique ID for savings csv and plots, based on model timestamp
        self.uid = datetime.datetime.now().strftime("%d-%H.%M.%S.%f")[:-3]
        print("Model uid", self.uid)

        # ----- SET UP SINDY -----
        data_fit = pd.read_csv("data/mpc_data19.csv")
        data_score = pd.read_csv("data/mpc_data_test9.csv")
        self.autoencoder = load_pickle("heatEq_autoencoder_3dim_tanh.pickle")
        self.sindy = SINDYc(self.autoencoder, data_fit, data_score)
        self.sindy.fit()

        # ----- SET UP THE BASIC MODEL -----
        # Set up pyomo model structure
        self.model = ConcreteModel()
        self.model.time = ContinuousSet(bounds=(0, duration))

        # Initial state: the rod is 273 Kelvins throughout (all x_full = 273)
        # Initial values for x_rom - SCALED FOR SINDY
        x_init = [0.20573092]
        x_init = self.sindy.x_rom_scaler.transform(np.array(x_init).reshape(1, 1)).flatten()

        self.model.x0 = Var(self.model.time)
        self.model.x0_dot = DerivativeVar(self.model.x0, wrt=self.model.time)
        self.model.x0[0].fix(x_init[0])
        # self.model.x1 = Var(self.model.time)
        # self.model.x1_dot = DerivativeVar(self.model.x1, wrt=self.model.time)
        # self.model.x1[0].fix(x_init[1])
        # self.model.x2 = Var(self.model.time)
        # self.model.x2_dot = DerivativeVar(self.model.x2, wrt=self.model.time)
        # self.model.x2[0].fix(x_init[2])

        # Set up controls
        # self.model.u0 = Var(self.model.time, bounds=(-1.10092131, 1.29623675))
        self.model.u0 = Var(self.model.time, bounds=(-1, 1.84))
        # self.model.u0 = Var(self.model.time)
        # self.model.u1 = Var(self.model.time, bounds=(-1, 1))
        self.model.u1 = Var(self.model.time, bounds=(-1, 1.84))

        # self.model.u1 = Var(self.model.time)

        # ODEs
        # Set up x0_dot = Ax + Bu

        def _ode_x0(m, _t):
            # return m.x0[_t] == 8.0091 + -5.595 * m.x0[_t] + 1.518 * m.u0[_t] + -2.463 *m.u1[_t]
            return m.x0_dot[_t] == 8.4461 + -13.747 * m.x0[_t] + 1.167 * m.u0[_t] + -4.188 * m.u1[_t] + 4.583 * m.x0[
                _t] ** 2 + -0.168 * m.u0[_t] ** 2 + 4.900 * m.u1[_t] ** 2

        self.model.x0_ode = Constraint(self.model.time, rule=_ode_x0)

        # Set up x1_dot to x18_dot = Ax only
        # def _ode_x1(m, _t):
        #     return m.x1_dot[_t] == -0.7591 + 1.243 * m.x0[_t] + 0.619 * m.x1[_t] + 0.349 * m.x2[_t] + -2.548  * m.u0[_t]  + 1.129  * m.u1[_t]
        # self.model.x1_ode = Constraint(self.model.time, rule=_ode_x1)

        # def _ode_x2(m, _t):
        #     return m.x2_dot[_t] == 2.1331 + -3.858 * m.x0[_t] + 1.087 * m.x1[_t] + -3.432 * m.x2[_t] + -0.829  * m.u0[_t]  + 2.771  * m.u1[_t]
        # self.model.x2_ode = Constraint(self.model.time, rule=_ode_x2)

        # Lagrangian cost
        self.model.L = Var(self.model.time)
        self.model.L_dot = DerivativeVar(self.model.L, wrt=self.model.time)
        self.model.L[0].fix(0)

        # ----- OBJECTIVE AND COST FUNCTION -----
        # Objective:
        # We want to heat element 6 (x[5]) at the 1/3 position to 30 C, 303 K
        # And element 14 (x[13]) at the 2/3 position to 60 C, 333 K
        # We would like to minimize the controller costs too, in terms of how much heating or cooling is applied

        # Load extracted weights
        w1 = np.load("autoencoder_weights_biases_tanh/input_weight.npy")
        w2 = np.load("autoencoder_weights_biases_tanh/h1_weight.npy")
        w3 = np.load("autoencoder_weights_biases_tanh/h2_weight.npy")
        w4 = np.load("autoencoder_weights_biases_tanh/h3_weight.npy")
        # Load extracted biases
        b1 = np.load("autoencoder_weights_biases_tanh/input_bias.npy")
        b2 = np.load("autoencoder_weights_biases_tanh/h1_bias.npy")
        b3 = np.load("autoencoder_weights_biases_tanh/h2_bias.npy")
        b4 = np.load("autoencoder_weights_biases_tanh/h3_bias.npy")

        W = [w1, w2, w3, w4]
        B = [b1, b2, b3, b4]

        self.sindy_dataMin = np.array([-1.2632675]).flatten()
        self.sindy_dataMax = np.array([1.2731746]).flatten()
        self.ae_dataMin = np.array([243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243,
                                    243, 243, 243, 243, 243, 243]).flatten()
        self.ae_dataMax = np.array([400.92786, 366.24084, 338.71667, 317.02072, 303.4187, 303.0223, 303.5431
                                       , 304.84406, 306.95587, 309.94135, 313.89136, 318.93024, 325.22433, 332.9931
                                       , 342.5486, 354.29337, 368.7209, 386.54282, 408.7622, 436.84055]).flatten()

        self.ae_scale = [0.00633201, 0.00811419, 0.0104475, 0.01350973, 0.01655117, 0.01666047
            , 0.01651716, 0.0161697, 0.01563578, 0.01493845, 0.01410609, 0.01316998
            , 0.01216185, 0.01111196, 0.01004534, 0.00898526, 0.00795413, 0.00696656
            , 0.00603274, 0.00515888]
        self.sindy_scale = [0.39425302]

        self.ae_min = [-1.5386773, -1.9717488, -2.5387425, -3.2828646, -4.0219336, -4.048495
            , -4.01367, -3.9292378, -3.7994947, -3.6300437, -3.4277804, -3.2003062
            , -2.9553297, -2.700207, -2.4410183, -2.1834185, -1.9328531, -1.6928747
            , -1.4659554, -1.2536076]

        self.sindy_min = [0.49804702]

        # Define weights for setpoint and controller objectives
        setpoint_weight = 0.995
        controller_weight = 1 - setpoint_weight

        # Lagrangian cost
        # def _Lagrangian(m, _t):
        #     # Array of pyomo model variables
        #     x_hat = np.array([m.x0[_t], m.x1[_t], m.x2[_t]]).reshape(1, 3)
        #
        #     # print(x_hat)
        #
        #     def _sindy_inverse_transform(x, idx):
        #         x = (x - 1) * (self.sindy_dataMax[idx] - self.sindy_dataMin[idx]) + self.sindy_dataMin[idx]
        #         return x
        #     for idx in range(x_hat.shape[0]):
        #         # x_hat[idx] = _sindy_inverse_transform(x_hat[idx], idx)
        #         x_hat[idx] = (x_hat[idx] - 1) * (self.sindy_dataMax[idx] - self.sindy_dataMin[idx]) + self.sindy_dataMin[idx]
        #
        #     # print(x_hat)
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
        #     # print(x_hat)
        #
        #     x_hat = np.array(x_hat).flatten()
        #
        #     def _ae_inverse_transform(x, idx):
        #         x = (x - 1) * (self.ae_dataMax[idx] - self.ae_dataMin[idx]) + self.ae_dataMin[idx]
        #         return x
        #     for idx in range(x_hat.shape[0]):
        #         x_hat[idx] = (x_hat[idx] - 1) * (self.ae_dataMax[idx] - self.ae_dataMin[idx]) + self.ae_dataMin[idx]
        #         # x_hat[idx] = _ae_inverse_transform(x_hat[idx], idx)
        #
        #     # x_hat = self.sindy.x_rom_scaler.inverse_transform(x_hat)
        #     # x_hat = self.autoencoder.scaler_x.inverse_transform(x_hat)
        #     x_hat = x_hat.flatten()
        #
        #     x5 = x_hat[5]
        #
        #     # print(x5)
        #
        #     x13 = x_hat[13]
        #
        #     # print(x13)
        #
        #     # Scale u0 and u1 into Sindy space for cost
        #     u0_273 = self.sindy.u0_scaler.transform(np.array(273).reshape(-1, 1)).flatten()
        #     u1_273 = self.sindy.u1_scaler.transform(np.array(273).reshape(-1, 1)).flatten()
        #
        #     return m.L_dot[_t] == setpoint_weight * ((x5 - 303) ** 2 + (x13 - 333) ** 2) \
        #            + controller_weight * ((m.u0[_t] - u0_273[0]) ** 2 + (m.u1[_t] - u1_273[0]) ** 2)
        #
        # self.model.L_integral = Constraint(self.model.time, rule=_Lagrangian)

        x_rom_setpoints = np.array([0.99993205]).reshape(-1, 1).flatten()

        # x_rom_setpoints = self.sindy.x_rom_scaler.transform(x_rom_setpoints).flatten()

        # Lagrangian cost
        def _Lagrangian(m, _t):
            return m.L_dot[_t] \
                   == setpoint_weight * ((m.x0[_t] - x_rom_setpoints[0]) ** 2)
            # + controller_weight * ((m.u0[_t] - 0.333) ** 2 + (m.u1[_t] - 0.333) ** 2)

        self.model.L_integral = Constraint(self.model.time, rule=_Lagrangian)

        # Objective function is to minimize the Lagrangian cost integral
        def _objective(m):
            return m.L[m.time.last()] - m.L[0]

        self.model.objective = Objective(rule=_objective, sense=minimize)

        constraint_counter = 0

        # Constraint for x5 <= 313 K
        def _constraint_x5(m, _t):
            # Array of pyomo model variables
            x_hat = np.array([m.x0[_t]]).reshape(1, 1)

            def _sindy_inverse_transform(x, idx):
                x -= self.sindy_min[idx]
                x /= self.sindy_scale[idx]
                # x = (x - 1) * (self.sindy_dataMax[idx] - self.sindy_dataMin[idx]) + self.sindy_dataMin[idx]
                return x
            for idx in range(x_hat.shape[0]):
                x_hat[idx] = _sindy_inverse_transform(x_hat[idx], idx)

            # Forward pass
            x_hat = x_hat @ W[0] + B[0]
            for row in range(x_hat.shape[0]):
                for col in range(x_hat.shape[1]):
                    x_hat[row, col] = tanh(x_hat[row, col])

            x_hat = x_hat @ W[1] + B[1]
            for row in range(x_hat.shape[0]):
                for col in range(x_hat.shape[1]):
                    x_hat[row, col] = tanh(x_hat[row, col])

            x_hat = x_hat @ W[2] + B[2]
            for row in range(x_hat.shape[0]):
                for col in range(x_hat.shape[1]):
                    x_hat[row, col] = tanh(x_hat[row, col])

            x_hat = x_hat @ W[3] + B[3]

            x_hat = np.array(x_hat).flatten()

            def _ae_inverse_transform(x, idx):
                # x = (x - 1) * (self.ae_dataMax[idx] - self.ae_dataMin[idx]) + self.ae_dataMin[idx]
                x -= self.ae_min[idx]
                x /= self.ae_scale[idx]
                return x

            for idx in range(x_hat.shape[0]):
                x_hat[idx] = _ae_inverse_transform(x_hat[idx], idx)

            # x_hat = self.sindy.x_rom_scaler.inverse_transform(x_hat)
            # x_hat = self.autoencoder.scaler_x.inverse_transform(x_hat)
            x_hat = x_hat.flatten()
            return x_hat[5] <= (5 * _t ** 2) + (10 * _t) + 293
            # return x_hat[13] <= 140

            # sindy scaler_x.inverse_transform
            # return m.x0[_t] <= 313
            # print(self.autoencoder.decode(np.hstack((value(m.x0[_t]), value(m.x1[_t]), value(m.x2[_t])))))
            # print(self.autoencoder.decode(np.hstack((value(m.x0[_t]), value(m.x1[_t]), value(m.x2[_t])))) <= 313)
            # return self.autoencoder.decode_pyomo(m.x0[_t], m.x1[_t], m.x2[_t])

        self.model.constraint_x5 = Constraint(self.model.time, rule=_constraint_x5)

        # ----- DISCRETIZE THE MODEL INTO FINITE ELEMENTS -----
        # We need to discretize before adding ODEs in matrix form
        # We fix finite elements at 10, collocation points at 4, controls to be piecewise linear
        discretizer = TransformationFactory("dae.collocation")
        discretizer.apply_to(self.model, nfe=10, ncp=4, scheme="LAGRANGE-RADAU")

        # Make controls piecewise linear
        discretizer.reduce_collocation_points(self.model, var=self.model.u0, ncp=1, contset=self.model.time)
        discretizer.reduce_collocation_points(self.model, var=self.model.u1, ncp=1, contset=self.model.time)

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
                x.append(temp_x)

        # Make sure all 11 time steps are recorded; this was problematic due to Pyomo's float indexing
        assert len(t) == 11

        # Scale back everything
        u0 = self.sindy.u0_scaler.inverse_transform(np.array(u0).reshape(-1, 1)).flatten()
        u1 = self.sindy.u1_scaler.inverse_transform(np.array(u1).reshape(-1, 1)).flatten()
        x = self.sindy.x_rom_scaler.inverse_transform(np.array(x).reshape(-1, 1))
        x = self.autoencoder.decode(x)

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

        x5_path_cst = []
        # 5t^2 + 10t + 293
        for ts in t:
            x5_path_cst.append((5 * ts ** 2) + (10 * ts) + 293)

        axs[0].plot(t, x5, label="$x_5$")
        axs[0].plot(t, x5_path_cst, label="Path constraint for $x_5$")
        axs[0].plot(t, np.full(shape=(t.size,), fill_value=303), "--", label="Setpoint for $x_5$")
        axs[0].legend()

        axs[1].plot(t, x13, label="$x_{13}$")
        axs[1].plot(t, np.full(shape=(t.size,), fill_value=333), "--", label="Setpoint for $x_{13}$")
        axs[1].legend()

        axs[2].step(t, u0, label="$u_0$")
        axs[2].step(t, u1, label="$u_1$")
        axs[2].legend()

        fig.suptitle("Sindy MPC performance as seen in the ROM system\n "
                     "Reduced to 1 state, constrained using integrated decoder"
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
    main_res["u0"].to_csv("u0.csv")
    main_res["u1"].to_csv("u1.csv")
    heatEq_system.plot(main_res)
