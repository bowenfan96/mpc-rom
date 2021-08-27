import csv
import datetime
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyomo.environ import *
from pyomo.dae import *
from pyomo.solvers import *

results_folder = ""


class HeatEqSimulator:
    def __init__(self, duration=1, N=5):
        # Unique ID for savings csv and plots, based on model timestamp
        self.uid = datetime.datetime.now().strftime("%d-%H.%M.%S.%f")[:-3]
        print("Model uid", self.uid)

        # ----- REDUCED MODEL MATRICES -----
        self.A = [[-1.7890, 0.3302, 3.2920, -3.7327, 1.6222],
                  [0.8249, -3.5471, 4.7861, 10.9044, -5.9803],
                  [-3.1413, -4.6014, -20.8645, 2.2005, 2.5368],
                  [3.9364, -11.0162, 7.1610, -32.0809, 46.6012],
                  [1.9797, -5.8669, 0.3289, -47.4296, -99.0670]]
        self.A = np.array(self.A)

        self.B = [[-11.3132, -8.1587],
                  [9.8776, -7.3334],
                  [-3.2400, -14.3493],
                  [17.8005, -0.7036],
                  [11.3212, -2.3636]]
        self.B = np.array(self.B)

        self.C = [[-0.1091, 0.0949, 0.0355, -0.1777, 0.1098],
                  [-0.0869, -0.0783, 0.1428, 0.0120, -0.0351]]
        self.C = np.array(self.C)

        # ----- SET UP THE BASIC MODEL -----
        # Set up pyomo model structure
        self.model = ConcreteModel()
        self.model.time = ContinuousSet(bounds=(0, duration))

        # Initial state: the rod is 273 Kelvins throughout
        # Change this array if random initial states are desired
        x_init = np.full(shape=(N,), fill_value=273)

        self.model.I = RangeSet(0, self.A.shape[1] - 1)
        self.model.x = Var(self.model.I, self.model.time)
        self.model.x_dot = DerivativeVar(self.model.x, wrt=self.model.time, initialize=0)

        self.model.J = RangeSet(0, self.B.shape[1] - 1)
        self.model.u = Var(self.model.J, self.model.time, initialize=0, bounds=(173, 373))

        # FOR Y = CX
        self.model.K = RangeSet(0, self.C.shape[0] - 1)
        self.model.y = Var(self.model.K, self.model.time)

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
        # setpoint_weight = 1
        controller_weight = 1 - setpoint_weight

        # Lagrangian cost
        # def _Lagrangian(m, _t):
        #     return m.L_dot[_t] \
        #            == setpoint_weight * ((m.y[0, _t] - 273) ** 2 + (m.y[1, _t] - 273) ** 2)

        def _Lagrangian(m, _t):
            return m.L_dot[_t] \
                   == setpoint_weight * ((m.y[0, _t] - 303) ** 2 + (m.y[1, _t] - 333) ** 2) \
                   + controller_weight * ((m.u[0, _t] - 273) ** 2 + (m.u[1, _t] - 273) ** 2)

        self.model.L_integral = Constraint(self.model.time, rule=_Lagrangian)

        # Objective function is to minimize the Lagrangian cost integral
        def _objective(m):
            return m.L[m.time.last()] - m.L[0]

        self.model.objective = Objective(rule=_objective, sense=minimize)

        # # Constraint for the element at the 1/3 position: temperature must not exceed 313 K (10 K above setpoint)
        # def _constraint_x5(m, _t):
        #     return m.y[0, _t] <= 313
        #
        # self.model.constraint_x5 = Constraint(self.model.time, rule=_constraint_x5)

        # Define derivative variables
        def ode_Ax(m, i, t):
            return sum((m.x[j, t] * self.A[i][j]) for j in range(self.A.shape[1]))

        def ode_Bu(m, i, t):
            return sum((m.u[j, t] * self.B[i][j]) for j in range(self.B.shape[1]))

        def ode_Cx(m, k, t):
            return sum((m.x[j, t] * self.C[k][j]) for j in range(self.C.shape[1]))

        self.model.ode = ConstraintList()
        self.model.ycx = ConstraintList()

        # Edit finite element step size here
        self.discretizer = TransformationFactory('dae.collocation')
        self.discretizer.apply_to(self.model, wrt=self.model.time, nfe=10,
                                  scheme='LAGRANGE-RADAU', ncp=4)
        # Force u to be piecewise linear
        self.discretizer.reduce_collocation_points(self.model, var=self.model.u, ncp=1, contset=self.model.time)

        # FOR GENERAL X_DOT = AX + BU
        for time in self.model.time:
            for i in range(self.A.shape[0]):
                self.model.ode.add(
                    self.model.x_dot[i, time] == ode_Ax(self.model, i, time) + ode_Bu(self.model, i, time)
                )

        # Fix variables based on initial values
        self.model.x[0, 0].fix(-2587.204565)
        self.model.x[1, 0].fix(-244.729339)
        self.model.x[2, 0].fix(207.1720569)
        self.model.x[3, 0].fix(-34.3094579)
        self.model.x[4, 0].fix(4.6393217)

        # FOR Y = CX + DU, IF Y IS THE OBJECTIVE TO BE CONTROLLED
        for time in self.model.time:
            for k in range(self.B.shape[1]):
                self.model.ycx.add(
                    self.model.y[k, time] == ode_Cx(self.model, k, time)
                )
        return

    def mpc_control(self):
        mpc_solver = SolverFactory("ipopt", tee=True)
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
                u0.append(value(self.model.u[0, time]))
                u1.append(value(self.model.u[1, time]))
                L.append(value(self.model.L[time]))

                # Get all the x values
                temp_x = []
                temp_x.append(value(self.model.y[0, time]))
                temp_x.append(value(self.model.y[1, time]))
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
        for x_idx in range(self.B.shape[1]):
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
        axs[0].plot(t, np.full(shape=(t.size,), fill_value=313), label="Constraint for $x_5$")
        axs[0].plot(t, np.full(shape=(t.size,), fill_value=303), "--", label="Setpoint for $x_5$")
        axs[0].legend()

        axs[1].plot(t, x13, label="$x_{13}$")
        axs[1].plot(t, np.full(shape=(t.size,), fill_value=333), "--", label="Setpoint for $x_{13}$")
        axs[1].legend()

        axs[2].step(t, u0, label="$u_0$")
        axs[2].step(t, u1, label="$u_1$")
        axs[2].legend()

        # fig.suptitle("Control policy and system state after {} rounds of training \n "
        #              "Run {}: Cost = {}, Constraint = {}"
        #              .format(num_rounds, num_run_in_round, total_cost, cst_status))
        # plt.xlabel("Time")

        # fig.suptitle("MPC BPOD Controller: Cost achieved = {}\n"
        #              "BPOD model reduction from 20 to 5 states".format(total_cost))

        fig.suptitle("ROM system as seen by the BPOD Controller\n"
                     "Some inaccuracies in output dynamics v. FOM system".format(total_cost))

        plt.xlabel("Time")

        plt.savefig(fname="BPOD_ROM_system.svg", format="svg")

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
    # main_res.to_csv("BPOD_results.csv")
    heatEq_system.plot(main_res)
