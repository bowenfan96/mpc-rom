import csv
import datetime
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyomo.environ import *
from pyomo.dae import *
from pyomo.solvers import *

from heatEq_nn_controller import *

results_folder = "heatEq_replay_results/vertex05-moreExpl/"


# THIN ROD WITH DIMENSIONALITY REDUCED FROM 20 TO 5


class HeatEqSimulator:
    def __init__(self, duration=1):
        # Unique ID for savings csv and plots, based on model timestamp
        self.uid = datetime.datetime.now().strftime("%d-%H.%M.%S.%f")[:-3]
        print("Model uid", self.uid)

        # ----- SET UP THE BASIC MODEL -----
        # Set up pyomo model structure
        self.model = ConcreteModel()
        self.model.time = ContinuousSet(bounds=(0, duration))

        # Initial state: the rod is 273 Kelvins throughout (all x_full = 273)
        # Initial values for x_rom:
        #      x0_rom    x1_rom    x2_rom    x3_rom    x4_rom
        #   -0.203286 -0.271189  0.407007 -0.666588 -0.234218
        x_init = [0.2628937,  0.6858409,  0.44120657]

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

        # Set up controls
        self.model.u0 = Var(self.model.time)
        self.model.u1 = Var(self.model.time)

        # x0' = 7769149.7131 + -151255.558 x0 + -87800.330 x1 + -140319.799 x2 + 46491.535 x3 + -37807.701 x4 + -1.117 u0 + -1.100 u1
        # x1' = 1122573.8321 + -21845.813 x0 + -12695.379 x1 + -20260.219 x2 + 6715.461 x3 + -5444.430 x4 + -0.139 u0 + -0.129 u1
        # x2' = -5640898.6741 + 109818.855 x0 + 63750.531 x1 + 101878.377 x2 + -33754.750 x3 + 27445.793 x4 + 0.794 u0 + 0.783 u1
        # x3' = 12490213.9981 + -243148.831 x0 + -141174.186 x1 + -225553.599 x2 + 74739.735 x3 + -60743.890 x4 + -1.783 u0 + -1.739 u1
        # x4' = 2605548.1721 + -50715.154 x0 + -29456.870 x1 + -47041.362 x2 + 15588.982 x3 + -12656.697 x4 + -0.351 u0 + -0.343 u1

        # ODEs
        # Set up x0_dot = Ax + Bu
        def _ode_x0(m, _t):
            return m.x0_dot[_t] == 7.8731 + -5.435* self.model.x0[_t] + -7.725* self.model.x1[_t] + -6.196* self.model.x2[_t] + 2.014* self.model.u0[_t] + 1.843* self.model.u1[_t] + -0.161* self.model.x0[_t]**2 + 1.419* self.model.x1[_t]**2 + -1.618* self.model.x2[_t]**2 + -0.166* self.model.u0[_t]**2 + 0.513 *self.model.u1[_t]**2
        self.model.x0_ode = Constraint(self.model.time, rule=_ode_x0)

        # Set up x1_dot to x18_dot = Ax only
        def _ode_x1(m, _t):
            return m.x1_dot[_t] == -10.6991 + 10.793* self.model.x0[_t] + 9.294* self.model.x1[_t] + 6.198* self.model.x2[_t] + 1.789* self.model.u0[_t] + -3.086* self.model.u1[_t] + -2.686* self.model.x0[_t]**2 + -0.930* self.model.x1[_t]**2 + 1.490* self.model.x2[_t]**2 + -0.999* self.model.u0[_t]**2 + -1.779 *self.model.u1[_t]**2
        self.model.x1_ode = Constraint(self.model.time, rule=_ode_x1)

        def _ode_x2(m, _t):
            return m.x2_dot[_t] == -0.6151 + 0.955* self.model.x0[_t] + 0.906* self.model.x1[_t] + 2.317* self.model.x2[_t] + -4.927* self.model.u0[_t] + 1.063* self.model.x0[_t]**2 + -0.930* self.model.x1[_t]**2 + 0.264* self.model.x2[_t]**2 + 1.581* self.model.u0[_t]**2 + 0.677 *self.model.u1[_t]**2
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

        # Lagrangian cost
        def _Lagrangian(m, _t):
            return m.L_dot[_t] \
                   == setpoint_weight * ((m.x0[_t] - 0.70213366) ** 2
                                         + (m.x1[_t] - 0.211213) ** 2
                                         + (m.x2[_t] - 0.98931336) ** 2)
                   # + controller_weight * ((m.u0[_t] - 273) ** 2 + (m.u1[_t] - 273) ** 2)
        self.model.L_integral = Constraint(self.model.time, rule=_Lagrangian)

        # Objective function is to minimize the Lagrangian cost integral
        def _objective(m):
            return m.L[m.time.last()] - m.L[0]
        self.model.objective = Objective(rule=_objective, sense=minimize)

        # # Constraint for the element at the 1/3 position: temperature must not exceed 313 K (10 K above setpoint)
        # def _constraint_x5(m, _t):
        #     return m.x5[_t] <= 313
        # self.model.constraint_x5 = Constraint(self.model.time, rule=_constraint_x5)

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
        mpc_solver.options['max_iter'] = 10000
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
        for x_idx in range(3):
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
        x5 = dataframe["x0"]
        x13 = dataframe["x1"]
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
        svg_filename = results_folder + "Round {} Run {} Cost {} Constraint {}"\
            .format(num_rounds, num_run_in_round, total_cost, cst_status) + ".svg"
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
    # generate_trajectories(save_csv=True)

    # main_simple_sys = HeatEqSimulator()
    # main_nn_model = load_pickle("simple_nn_controller.pickle")
    # main_res, _ = main_simple_sys.simulate_system_nn_controls(main_nn_model)
    # main_simple_sys.plot(main_res)
    # print(main_res)

    # replay("heatEq_240_trajectories_df.csv")

    heatEq_system = HeatEqSimulator()
    print(heatEq_system.mpc_control())
    main_res, _ = heatEq_system.parse_mpc_results()
    heatEq_system.plot(main_res)
