import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyomo.dae import *
from pyomo.environ import *

results_folder = ""


class HeatEqSimulator:
    def __init__(self, duration=1, N=20):
        # Unique ID for savings csv and plots, based on model timestamp
        self.uid = datetime.datetime.now().strftime("%d-%H.%M.%S.%f")[:-3]
        print("Model uid", self.uid)

        # ----- GENERATE THE MODEL MATRICES -----
        # Apply the method of lines on the heat equation to generate the A matrix
        # Length of the rod = 1 m
        # Number of segments = number of discretization points - 1 (as 2 ends take up 2 points)
        length = 1
        num_segments = N - 1
        # Thermal diffusivity alpha
        alpha = 0.3
        segment_length = length / num_segments
        # Constant
        c = alpha / (segment_length ** 2)

        # Generate A matrix
        A_mat = np.zeros(shape=(N, N))
        for row in range(A_mat.shape[0]):
            for col in range(A_mat.shape[1]):
                if row == col:
                    A_mat[row][col] = -2
                elif abs(row - col) == 1:
                    A_mat[row][col] = 1
                else:
                    A_mat[row][col] = 0
        # Multiply constant to all elements in A
        self.A = c * A_mat

        # Generate B matrix
        # Two sources of heat at each end of the rod
        num_heaters = 2
        B_mat = np.zeros(shape=(N, num_heaters))
        # First heater on the left
        B_mat[0][0] = 1
        # Second heater on the right
        B_mat[N - 1][num_heaters - 1] = 1
        # Multiply constant to all elements in B
        self.B = c * B_mat

        # ----- SET UP THE BASIC MODEL -----
        # Set up pyomo model structure
        self.model = ConcreteModel()
        self.model.time = ContinuousSet(bounds=(0, duration))

        # Initial state: the rod is 273 Kelvins throughout
        # Change this array if random initial states are desired
        x_init = np.full(shape=(N,), fill_value=273)
        # x_init = np.random.randint(low=263, high=283, size=N)

        # NOTE: Pyomo can simulate via scipy/casadi only if:
        # 1. model.u is indexed only by time, so Bu using matrix multiplication is not possible
        # 2. model contains if statements, so the ode cannot have conditions
        # After trying many things, this (silly) method seems to be the only way
        # if we want pyomo to simulate with random controls

        # Set up all the finite elements
        self.model.x0 = Var(self.model.time)
        self.model.x0_dot = DerivativeVar(self.model.x0, wrt=self.model.time)
        self.model.x0[0].fix(x_init[0])
        self.model.x1 = Var(self.model.time)
        self.model.x1_dot = DerivativeVar(self.model.x1, wrt=self.model.time)
        self.model.x1[0].fix(x_init[1])
        self.model.x2 = Var(self.model.time)
        self.model.x2_dot = DerivativeVar(self.model.x2, wrt=self.model.time)
        self.model.x2[0].fix(x_init[2])
        self.model.x3 = Var(self.model.time)
        self.model.x3_dot = DerivativeVar(self.model.x3, wrt=self.model.time)
        self.model.x3[0].fix(x_init[3])
        self.model.x4 = Var(self.model.time)
        self.model.x4_dot = DerivativeVar(self.model.x4, wrt=self.model.time)
        self.model.x4[0].fix(x_init[4])
        self.model.x5 = Var(self.model.time)
        self.model.x5_dot = DerivativeVar(self.model.x5, wrt=self.model.time)
        self.model.x5[0].fix(x_init[5])
        self.model.x6 = Var(self.model.time)
        self.model.x6_dot = DerivativeVar(self.model.x6, wrt=self.model.time)
        self.model.x6[0].fix(x_init[6])
        self.model.x7 = Var(self.model.time)
        self.model.x7_dot = DerivativeVar(self.model.x7, wrt=self.model.time)
        self.model.x7[0].fix(x_init[7])
        self.model.x8 = Var(self.model.time)
        self.model.x8_dot = DerivativeVar(self.model.x8, wrt=self.model.time)
        self.model.x8[0].fix(x_init[8])
        self.model.x9 = Var(self.model.time)
        self.model.x9_dot = DerivativeVar(self.model.x9, wrt=self.model.time)
        self.model.x9[0].fix(x_init[9])
        self.model.x10 = Var(self.model.time)
        self.model.x10_dot = DerivativeVar(self.model.x10, wrt=self.model.time)
        self.model.x10[0].fix(x_init[10])
        self.model.x11 = Var(self.model.time)
        self.model.x11_dot = DerivativeVar(self.model.x11, wrt=self.model.time)
        self.model.x11[0].fix(x_init[11])
        self.model.x12 = Var(self.model.time)
        self.model.x12_dot = DerivativeVar(self.model.x12, wrt=self.model.time)
        self.model.x12[0].fix(x_init[12])
        self.model.x13 = Var(self.model.time)
        self.model.x13_dot = DerivativeVar(self.model.x13, wrt=self.model.time)
        self.model.x13[0].fix(x_init[13])
        self.model.x14 = Var(self.model.time)
        self.model.x14_dot = DerivativeVar(self.model.x14, wrt=self.model.time)
        self.model.x14[0].fix(x_init[14])
        self.model.x15 = Var(self.model.time)
        self.model.x15_dot = DerivativeVar(self.model.x15, wrt=self.model.time)
        self.model.x15[0].fix(x_init[15])
        self.model.x16 = Var(self.model.time)
        self.model.x16_dot = DerivativeVar(self.model.x16, wrt=self.model.time)
        self.model.x16[0].fix(x_init[16])
        self.model.x17 = Var(self.model.time)
        self.model.x17_dot = DerivativeVar(self.model.x17, wrt=self.model.time)
        self.model.x17[0].fix(x_init[17])
        self.model.x18 = Var(self.model.time)
        self.model.x18_dot = DerivativeVar(self.model.x18, wrt=self.model.time)
        self.model.x18[0].fix(x_init[18])
        self.model.x19 = Var(self.model.time)
        self.model.x19_dot = DerivativeVar(self.model.x19, wrt=self.model.time)
        self.model.x19[0].fix(x_init[19])

        # Set up controls
        self.model.u0 = Var(self.model.time, bounds=(173, 373))
        self.model.u1 = Var(self.model.time, bounds=(173, 373))

        # ODEs
        # Set up x0_dot = Ax + Bu
        def _ode_x0(m, _t):
            return m.x0_dot[_t] == self.A[0][0] * m.x0[_t] + self.A[0][1] * m.x1[_t] + self.B[0][0] * m.u0[_t]

        self.model.x0_ode = Constraint(self.model.time, rule=_ode_x0)

        # Set up x1_dot to x18_dot = Ax only
        def _ode_x1(m, _t):
            return m.x1_dot[_t] == self.A[1][1 - 1] * m.x0[_t] + self.A[1][1] * m.x1[_t] + self.A[1][1 + 1] * m.x2[_t]

        self.model.x1_ode = Constraint(self.model.time, rule=_ode_x1)

        def _ode_x2(m, _t):
            return m.x2_dot[_t] == self.A[2][2 - 1] * m.x1[_t] + self.A[2][2] * m.x2[_t] + self.A[2][2 + 1] * m.x3[_t]

        self.model.x2_ode = Constraint(self.model.time, rule=_ode_x2)

        def _ode_x3(m, _t):
            return m.x3_dot[_t] == self.A[3][3 - 1] * m.x2[_t] + self.A[3][3] * m.x3[_t] + self.A[3][3 + 1] * m.x4[_t]

        self.model.x3_ode = Constraint(self.model.time, rule=_ode_x3)

        def _ode_x4(m, _t):
            return m.x4_dot[_t] == self.A[4][4 - 1] * m.x3[_t] + self.A[4][4] * m.x4[_t] + self.A[4][4 + 1] * m.x5[_t]

        self.model.x4_ode = Constraint(self.model.time, rule=_ode_x4)

        def _ode_x5(m, _t):
            return m.x5_dot[_t] == self.A[5][5 - 1] * m.x4[_t] + self.A[5][5] * m.x5[_t] + self.A[5][5 + 1] * m.x6[_t]

        self.model.x5_ode = Constraint(self.model.time, rule=_ode_x5)

        def _ode_x6(m, _t):
            return m.x6_dot[_t] == self.A[6][6 - 1] * m.x5[_t] + self.A[6][6] * m.x6[_t] + self.A[6][6 + 1] * m.x7[_t]

        self.model.x6_ode = Constraint(self.model.time, rule=_ode_x6)

        def _ode_x7(m, _t):
            return m.x7_dot[_t] == self.A[7][7 - 1] * m.x6[_t] + self.A[7][7] * m.x7[_t] + self.A[7][7 + 1] * m.x8[_t]

        self.model.x7_ode = Constraint(self.model.time, rule=_ode_x7)

        def _ode_x8(m, _t):
            return m.x8_dot[_t] == self.A[8][8 - 1] * m.x7[_t] + self.A[8][8] * m.x8[_t] + self.A[8][8 + 1] * m.x9[_t]

        self.model.x8_ode = Constraint(self.model.time, rule=_ode_x8)

        def _ode_x9(m, _t):
            return m.x9_dot[_t] == self.A[9][9 - 1] * m.x8[_t] + self.A[9][9] * m.x9[_t] + self.A[9][9 + 1] * m.x10[_t]

        self.model.x9_ode = Constraint(self.model.time, rule=_ode_x9)

        def _ode_x10(m, _t):
            return m.x10_dot[_t] == self.A[10][10 - 1] * m.x9[_t] + self.A[10][10] * m.x10[_t] + self.A[10][10 + 1] * \
                   m.x11[_t]

        self.model.x10_ode = Constraint(self.model.time, rule=_ode_x10)

        def _ode_x11(m, _t):
            return m.x11_dot[_t] == self.A[11][11 - 1] * m.x10[_t] + self.A[11][11] * m.x11[_t] + self.A[11][11 + 1] * \
                   m.x12[_t]

        self.model.x11_ode = Constraint(self.model.time, rule=_ode_x11)

        def _ode_x12(m, _t):
            return m.x12_dot[_t] == self.A[12][12 - 1] * m.x11[_t] + self.A[12][12] * m.x12[_t] + self.A[12][12 + 1] * \
                   m.x13[_t]

        self.model.x12_ode = Constraint(self.model.time, rule=_ode_x12)

        def _ode_x13(m, _t):
            return m.x13_dot[_t] == self.A[13][13 - 1] * m.x12[_t] + self.A[13][13] * m.x13[_t] + self.A[13][13 + 1] * \
                   m.x14[_t]

        self.model.x13_ode = Constraint(self.model.time, rule=_ode_x13)

        def _ode_x14(m, _t):
            return m.x14_dot[_t] == self.A[14][14 - 1] * m.x13[_t] + self.A[14][14] * m.x14[_t] + self.A[14][14 + 1] * \
                   m.x15[_t]

        self.model.x14_ode = Constraint(self.model.time, rule=_ode_x14)

        def _ode_x15(m, _t):
            return m.x15_dot[_t] == self.A[15][15 - 1] * m.x14[_t] + self.A[15][15] * m.x15[_t] + self.A[15][15 + 1] * \
                   m.x16[_t]

        self.model.x15_ode = Constraint(self.model.time, rule=_ode_x15)

        def _ode_x16(m, _t):
            return m.x16_dot[_t] == self.A[16][16 - 1] * m.x15[_t] + self.A[16][16] * m.x16[_t] + self.A[16][16 + 1] * \
                   m.x17[_t]

        self.model.x16_ode = Constraint(self.model.time, rule=_ode_x16)

        def _ode_x17(m, _t):
            return m.x17_dot[_t] == self.A[17][17 - 1] * m.x16[_t] + self.A[17][17] * m.x17[_t] + self.A[17][17 + 1] * \
                   m.x18[_t]

        self.model.x17_ode = Constraint(self.model.time, rule=_ode_x17)

        def _ode_x18(m, _t):
            return m.x18_dot[_t] == self.A[18][18 - 1] * m.x17[_t] + self.A[18][18] * m.x18[_t] + self.A[18][18 + 1] * \
                   m.x19[_t]

        self.model.x18_ode = Constraint(self.model.time, rule=_ode_x18)

        # Set up x19_dot = Ax + Bu
        def _ode_x19(m, _t):
            return m.x19_dot[_t] == self.A[19][19] * m.x19[_t] + self.A[19][18] * m.x18[_t] + self.B[19][1] * m.u1[_t]

        self.model.x19_ode = Constraint(self.model.time, rule=_ode_x19)

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

        # Lagrangian cost
        def _Lagrangian(m, _t):
            return m.L_dot[_t] \
                   == setpoint_weight * ((m.x5[_t] - 303) ** 2 + (m.x13[_t] - 333) ** 2) \
                   + controller_weight * ((m.u0[_t] - 273) ** 2 + (m.u1[_t] - 273) ** 2)

        self.model.L_integral = Constraint(self.model.time, rule=_Lagrangian)

        # Objective function is to minimize the Lagrangian cost integral
        def _objective(m):
            return m.L[m.time.last()] - m.L[0]

        self.model.objective = Objective(rule=_objective, sense=minimize)

        # Constraint for the element at the 1/3 position: temperature must not exceed 313 K (10 K above setpoint)
        def _constraint_x5(m, _t):
            return m.x5[_t] <= 313

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
        mpc_results = mpc_solver.solve(self.model)

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
                temp_x.append(value(self.model.x3[time]))
                temp_x.append(value(self.model.x4[time]))
                temp_x.append(value(self.model.x5[time]))
                temp_x.append(value(self.model.x6[time]))
                temp_x.append(value(self.model.x7[time]))
                temp_x.append(value(self.model.x8[time]))
                temp_x.append(value(self.model.x9[time]))
                temp_x.append(value(self.model.x10[time]))
                temp_x.append(value(self.model.x11[time]))
                temp_x.append(value(self.model.x12[time]))
                temp_x.append(value(self.model.x13[time]))
                temp_x.append(value(self.model.x14[time]))
                temp_x.append(value(self.model.x15[time]))
                temp_x.append(value(self.model.x16[time]))
                temp_x.append(value(self.model.x17[time]))
                temp_x.append(value(self.model.x18[time]))
                temp_x.append(value(self.model.x19[time]))
                x.append(temp_x)

        # Make sure all 11 time steps are recorded; this was problematic due to Pyomo's float indexing
        assert len(t) == 11

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

        x = np.array(x)

        df_data = {"t": t}
        for x_idx in range(self.A.shape[0]):
            df_data["x{}".format(x_idx)] = x[:, x_idx]
        df_data["u0"] = u0
        df_data["u1"] = u1
        df_data["L"] = L
        df_data["inst_cost"] = inst_cost
        df_data["ctg"] = ctg

        mpc_results_df = pd.DataFrame(df_data)
        mpc_results_df_dropped_t0 = mpc_results_df.drop(index=0)

        return mpc_results_df, mpc_results_df_dropped_t0

    def simulate_system_rng_controls(self):
        timesteps = [timestep / 10 for timestep in range(11)]
        u0_rng = [
                  373,
                  360.2492412,
                  296.1008372,
                  294.5995899,
                  286.5524599,
                  283.5063943,
                  281.7249347,
                  281.0282384,
                  279.6974883,
                  275.8649638,
        275.8649638]
        u1_rng = [
                  373,
                  373,
                  373,
                  373,
                  373,
                  373,
                  373,
                  372.7735169,
                  362.835462,
                  296.7658734,
        296.7658734]

        # Create a dictionary of piecewise linear controller actions
        u0_rng_profile = {timesteps[i]: u0_rng[i] for i in range(len(timesteps))}
        u1_rng_profile = {timesteps[i]: u1_rng[i] for i in range(len(timesteps))}

        self.model.var_input = Suffix(direction=Suffix.LOCAL)
        self.model.var_input[self.model.u0] = u0_rng_profile
        self.model.var_input[self.model.u1] = u1_rng_profile

        sim = Simulator(self.model, package="casadi")
        tsim, profiles = sim.simulate(numpoints=11, varying_inputs=self.model.var_input)

        # profiles are given as 2D array: each row is a time instance, while the columns are:
        # x0, x1 ... x19, L

        # For some reason both tsim and profiles contain duplicates
        # Use pandas to drop the duplicates first
        # profiles columns: x0, x1, ..., x19, L
        temp_dict = {"t": tsim}
        for i in range(20):
            temp_dict["x{}".format(i)] = profiles[:, i]
        temp_dict["L"] = profiles[:, 20]

        deduplicate_df = pd.DataFrame(temp_dict)
        deduplicate_df = deduplicate_df.round(4)
        deduplicate_df.drop_duplicates(ignore_index=True, inplace=True)

        # Make dataframe from the simulator results
        t = deduplicate_df["t"]
        x = []
        for i in range(20):
            x.append(deduplicate_df["x{}".format(i)])
        L = deduplicate_df["L"]
        u0 = u0_rng
        u1 = u1_rng
        inst_cost = []
        ctg = []

        # Note: at this point, x is a list of 20 pandas series, each series has 11 rows
        # Check duplicates were removed correctly
        assert len(t) == 11

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

        rng_sim_results_df = pd.DataFrame(temp_dict)
        rng_sim_results_df_dropped_tf = rng_sim_results_df.drop(index=10)

        return rng_sim_results_df, rng_sim_results_df_dropped_tf


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
        axs[0].plot(t, np.full(shape=(t.size,), fill_value=313), label="Constraint for $x_5$")
        axs[0].plot(t, np.full(shape=(t.size,), fill_value=303), "--", label="Setpoint for $x_5$")
        axs[0].legend()

        axs[1].plot(t, x13, label="$x_{13}$")
        axs[1].plot(t, np.full(shape=(t.size,), fill_value=333), "--", label="Setpoint for $x_{13}$")
        axs[1].legend()

        axs[2].step(t, u0, label="$u_0$", where="post")
        axs[2].step(t, u1, label="$u_1$", where="post")
        axs[2].legend()

        # fig.suptitle("Control policy and system state after {} rounds of training \n "
        #              "Run {}: Cost = {}, Constraint = {}"
        #              .format(num_rounds, num_run_in_round, total_cost, cst_status))
        # plt.xlabel("Time")

        fig.suptitle("BPOD MPC Controls: Cost achieved = {}\n"
                     "FOM system, simulated with BPOD controls".format(total_cost))
        plt.xlabel("Time")

        # Save plot with autogenerated filename
        svg_filename = results_folder + "svgs/" + "Round {} Run {} Cost {} Constraint {}" \
            .format(num_rounds, num_run_in_round, total_cost, cst_status) + ".svg"
        # plt.savefig(fname=svg_filename, format="svg")
        plt.savefig(fname="BPOD_controls_on_FOM.svg", format="svg")

        plt.show()
        # plt.close()
        return


if __name__ == "__main__":
    # generate_trajectories(save_csv=True)

    # main_simple_sys = HeatEqSimulator()
    # main_nn_model = load_pickle("simple_nn_controller.pickle")
    # main_res, _ = main_simple_sys.simulate_system_nn_controls(main_nn_model)
    # main_simple_sys.plot(main_res)
    # print(main_res)

    # replay("heatEq_240_trajectories_df.csv")

    # heatEq_system = HeatEqSimulator()
    # main_res, _ = heatEq_system.simulate_system_sindy_controls()
    # heatEq_system.plot(main_res)
    # pd.set_option('display.max_columns', None)
    # print(main_res)
    # main_res.to_csv("heatEq_mpc_trajectory.csv")

    heatEq_system = HeatEqSimulator()
    heatEq_system.mpc_control()
    main_res, _ = heatEq_system.parse_mpc_results()
    # main_res.to_csv("FOM_results.csv")
    # heatEq_system.plot(main_res)
    #
    # heatEq_system = HeatEqSimulator()
    # main_res, _ = heatEq_system.simulate_system_rng_controls()
    # print(main_res)
    # heatEq_system.plot(main_res)
