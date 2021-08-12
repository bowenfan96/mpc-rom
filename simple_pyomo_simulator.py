# Citation: The system model is taken from https://colab.research.google.com/drive/17KJn7tVyQ3nXlGSGEJ0z6DcpRnRd0VRp
import csv
import datetime
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyomo.environ import *
from pyomo.dae import *
from pyomo.solvers import *

from simple_nn_controller import *

results_folder = "simple_replay_results/ray11-2/"


class SimpleSimulator:
    def __init__(self, duration=1, x0_init=0, x1_init=-1):
        # Unique ID for savings csv and plots, based on model timestamp
        self.uid = datetime.datetime.now().strftime("%d-%H.%M.%S.%f")[:-3]
        print("Model uid", self.uid)

        # ----- SET UP THE BASIC MODEL -----
        # Set up pyomo model structure
        self.model = ConcreteModel()
        self.model.time = ContinuousSet(bounds=(0, duration))

        # State variables
        self.model.x0 = Var(self.model.time)
        self.model.x1 = Var(self.model.time)
        self.model.x0_dot = DerivativeVar(self.model.x0, wrt=self.model.time)
        self.model.x1_dot = DerivativeVar(self.model.x1, wrt=self.model.time)
        # Initial state
        self.model.x0[0].fix(x0_init)
        self.model.x1[0].fix(x1_init)

        # Controls
        self.model.u = Var(self.model.time, bounds=(-20, 20))

        # Lagrangian cost
        self.model.L = Var(self.model.time)
        self.model.L_dot = DerivativeVar(self.model.L, wrt=self.model.time)
        self.model.L[0].fix(0)

        # ODEs
        def _ode_x0(m, t):
            return m.x0_dot[t] == m.x1[t]
        self.model.ode_x0 = Constraint(self.model.time, rule=_ode_x0)

        def _ode_x1(m, t):
            return m.x1_dot[t] == -m.x1[t] + m.u[t]
        self.model.ode_x1 = Constraint(self.model.time, rule=_ode_x1)

        # Path constraint for x1
        def _path_constraint_x1(m, t):
            return m.x1[t] + 0.5 - 8 * (t - 0.5) ** 2 <= 0
        self.model.constraint_x1 = Constraint(self.model.time, rule=_path_constraint_x1)

        # Lagrangian cost
        def _Lagrangian(m, t):
            return m.L_dot[t] == (m.x0[t] ** 2) + (m.x1[t] ** 2) + (5E-3 * m.u[t] ** 2)
        self.model.L_integral = Constraint(self.model.time, rule=_Lagrangian)

        # Objective function is to minimize the Lagrangian cost integral
        def _objective(m):
            return m.L[m.time.last()] - m.L[0]
        self.model.objective = Objective(rule=_objective, sense=minimize)

        # ----- DISCRETIZE THE MODEL INTO FINITE ELEMENTS -----
        # We fix finite elements at 10, collocation points at 4, controls to be piecewise linear
        discretizer = TransformationFactory("dae.collocation")
        discretizer.apply_to(self.model, nfe=10, ncp=4, scheme="LAGRANGE-RADAU")

        # Make controls piecewise linear
        discretizer.reduce_collocation_points(self.model, var=self.model.u, ncp=1, contset=self.model.time)

        return

    def mpc_control(self):
        mpc_solver = SolverFactory("ipopt")
        mpc_results = mpc_solver.solve(self.model)

        return mpc_results

    def parse_mpc_results(self):
        # Each t, X0, X1, U, L, instantaneous cost, cost to go, should be a column
        # Label them and return a pandas dataframe
        t = []
        x0 = []
        x1 = []
        u = []
        L = []
        inst_cost = []
        ctg = []

        # Record data at the intervals of finite elements only (0.1s), do not include collocation points
        timesteps = [timestep / 10 for timestep in range(11)]
        for time in self.model.time:
            if time in timesteps:
                t.append(time)
                x0.append(value(self.model.x0[time]))
                x1.append(value(self.model.x1[time]))
                u.append(value(self.model.u[time]))
                L.append(value(self.model.L[time]))

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

        mpc_results_df = pd.DataFrame(
            {"t": t, "x0": x0, "x1": x1, "u": u, "L": L, "inst_cost": inst_cost, "ctg": ctg}
        )
        mpc_results_df_dropped_t0 = mpc_results_df.drop(index=0)

        return mpc_results_df, mpc_results_df_dropped_t0

    def simulate_system_rng_controls(self):
        timesteps = [timestep / 10 for timestep in range(11)]
        u_rng = np.random.uniform(low=-20, high=20, size=11)

        # Create a dictionary of piecewise linear controller actions
        u_rng_profile = {timesteps[i]: u_rng[i] for i in range(len(timesteps))}

        self.model.var_input = Suffix(direction=Suffix.LOCAL)
        self.model.var_input[self.model.u] = u_rng_profile

        sim = Simulator(self.model, package="casadi")
        tsim, profiles = sim.simulate(numpoints=11, varying_inputs=self.model.var_input)

        # For some reason both tsim and profiles contain duplicates
        # Use pandas to drop the duplicates first
        # profiles columns: x0, x1, L
        deduplicate_df = pd.DataFrame(
            {"t": tsim, "x0": profiles[:, 0], "x1": profiles[:, 1], "L": profiles[:, 2]}
        )
        deduplicate_df = deduplicate_df.round(10)
        deduplicate_df.drop_duplicates(ignore_index=True, inplace=True)

        # Make dataframe from the simulator results
        t = deduplicate_df["t"]
        x0 = deduplicate_df["x0"]
        x1 = deduplicate_df["x1"]
        L = deduplicate_df["L"]
        u = u_rng
        inst_cost = []
        ctg = []

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
        path = [x1[int(time * 10)] + 0.5 - 8 * (time - 0.5) ** 2 for time in t]
        path_violation = []
        for p in path:
            if max(path) > 0:
                path_violation.append(max(path))
            else:
                path_violation.append(p)

        rng_sim_results_df = pd.DataFrame(
            {"t": t, "x0": x0, "x1": x1, "u": u, "L": L,
             "inst_cost": inst_cost, "ctg": ctg, "path_diff": path_violation}
        )
        rng_sim_results_df_dropped_tf = rng_sim_results_df.drop(index=10)

        return rng_sim_results_df, rng_sim_results_df_dropped_tf

    def simulate_system_nn_controls(self, nn_model):
        timesteps = [timestep / 10 for timestep in range(11)]
        u_nn = np.zeros(11)
        # Initial x values to be passed to the neural net at the first loop
        current_x = [0, -1]

        self.model.var_input = Suffix(direction=Suffix.LOCAL)

        # Pyomo does not support simulating step by step, so we need to run 11 simulation loops
        # At loop i, we get state x_i and discard subsequent states
        # We call the neural net to predict the optimal u for x_i, then fix u time i
        for i in range(11):
            # Fetch optimal action, u, by calling the neural net with current_x
            u_opt = nn_model.get_u_opt(current_x)

            # Replace the control sequence of the current timestep with u_opt
            u_nn[i] = u_opt

            # Create a dictionary of piecewise linear controller actions
            u_nn_profile = {timesteps[i]: u_nn[i] for i in range(len(timesteps))}

            # Update the control sequence to Pyomo
            self.model.var_input[self.model.u] = u_nn_profile

            sim = Simulator(self.model, package="casadi")
            tsim, profiles = sim.simulate(numpoints=11, varying_inputs=self.model.var_input)

            # For some reason both tsim and profiles contain duplicates
            # Use pandas to drop the duplicates first
            # profiles columns: x0, x1, L
            deduplicate_df = pd.DataFrame(
                {"t": tsim, "x0": profiles[:, 0], "x1": profiles[:, 1], "L": profiles[:, 2]}
            )
            deduplicate_df = deduplicate_df.round(10)
            deduplicate_df.drop_duplicates(ignore_index=True, inplace=True)

            # Make dataframe from the simulator results
            t = deduplicate_df["t"]
            x0 = deduplicate_df["x0"]
            x1 = deduplicate_df["x1"]

            # Check duplicates were removed correctly
            assert len(t) == 11

            # Update current_x to the next state output by the simulator
            if i < 10:
                current_x[0] = x0[i+1]
                current_x[1] = x1[i+1]

        # Make dataframe from the final simulator results
        t = deduplicate_df["t"]
        x0 = deduplicate_df["x0"]
        x1 = deduplicate_df["x1"]
        L = deduplicate_df["L"]
        u = u_nn
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
        path = [x1[int(time * 10)] + 0.5 - 8 * (time - 0.5) ** 2 for time in t]
        path_violation = []
        for p in path:
            if max(path) > 0:
                path_violation.append(max(path))
            else:
                path_violation.append(p)

        nn_sim_results_df = pd.DataFrame(
            {"t": t, "x0": x0, "x1": x1, "u": u, "L": L,
             "inst_cost": inst_cost, "ctg": ctg, "path_diff": path_violation}
        )
        nn_sim_results_df_dropped_tf = nn_sim_results_df.drop(index=10)

        return nn_sim_results_df, nn_sim_results_df_dropped_tf

    @staticmethod
    def plot(dataframe, num_rounds=0, num_run_in_round=0):
        t = dataframe["t"]
        x0 = dataframe["x0"]
        x1 = dataframe["x1"]
        u = dataframe["u"]
        ctg = dataframe["ctg"]
        cst = dataframe["path_diff"]

        if cst.max() <= 0:
            cst_status = "Pass"
        else:
            cst_status = "Fail"

        # Check that the cost to go is equal to the Lagrangian cost integral
        # assert np.isclose(ctg[0], dataframe["L"].iloc[-1], atol=0.01)
        total_cost = round(ctg[0], 3)

        fig, axs = plt.subplots(3, constrained_layout=True)
        fig.set_size_inches(5, 10)

        axs[0].plot(t, x0, label="$x_0$")
        axs[0].legend()

        axs[1].plot(t, x1, label="$x_1$")
        axs[1].plot(t, -0.5 + 8 * (np.array(t) - 0.5) ** 2, label="Path constraint for $x_1$")
        axs[1].legend()

        axs[2].step(t, u, label="Neural net controller action, u")
        axs[2].legend()

        fig.suptitle("Control policy and system state after {} rounds of training \n "
                     "Run {}: Cost = {}, Constraint = {}"
                     .format(num_rounds, num_run_in_round, total_cost, cst_status))
        plt.xlabel("Time")

        # Save plot with autogenerated filename
        svg_filename = results_folder + "Round {} Run {} Cost {} Constraint {}"\
            .format(num_rounds, num_run_in_round, total_cost, cst_status) + ".svg"
        plt.savefig(fname=svg_filename, format="svg")

        # plt.show()
        plt.close()

        return


def generate_trajectories(save_csv=False):
    df_cols = ["t", "x0", "x1", "u", "L", "inst_cost", "ctg", "path_diff"]
    # 40 trajectories which obeyed the path constraint
    obey_path_df = pd.DataFrame(columns=df_cols)
    # 20 trajectories which violated the path constraint
    violate_path_df = pd.DataFrame(columns=df_cols)
    simple_60_trajectories_df = pd.DataFrame(columns=df_cols)

    num_samples = 0
    num_good = 0
    num_bad = 0

    while num_samples < 120:

        while num_good < 2:
            simple_sys = SimpleSimulator()
            _, trajectory = simple_sys.simulate_system_rng_controls()
            if trajectory["path_diff"].max() <= 0:
                simple_60_trajectories_df = pd.concat([simple_60_trajectories_df, trajectory])
                obey_path_df = pd.concat([obey_path_df, trajectory])
                num_good += 1
                num_samples += 1

        while num_bad < 1:
            simple_sys = SimpleSimulator()
            _, trajectory = simple_sys.simulate_system_rng_controls()
            if trajectory["path_diff"].max() > 0:
                simple_60_trajectories_df = pd.concat([simple_60_trajectories_df, trajectory])
                violate_path_df = pd.concat([violate_path_df, trajectory])
                num_bad += 1
                num_samples += 1

        # Reset
        num_good = 0
        num_bad = 0

        print("Samples: ", num_samples)

    simple_60_trajectories_df.to_csv("simple_120_trajectories_df.csv")
    obey_path_df.to_csv("obey_path_df.csv")
    violate_path_df.to_csv("violate_path_df.csv")


def load_pickle(filename):
    with open(filename, "rb") as model:
        pickled_nn_model = pickle.load(model)
    print("Pickle loaded: " + filename)
    return pickled_nn_model


def replay(trajectory_df_filename, buffer_capacity=240):
    # Use this to keep track where to push out old data
    forgotten_trajectories_count = 0
    pickle_filename = "simple_nn_controller_120.pickle"
    og_trajectory_df_filename = trajectory_df_filename

    best_cost_after_n_rounds = {}

    for rp_round in range(150):
        trajectory_df = pd.read_csv(results_folder + trajectory_df_filename, sep=",")
        nn_model = load_pickle(pickle_filename)
        run_trajectories = []

        best_cost_in_round = np.inf

        for run in range(6):
            simple_sys = SimpleSimulator()
            df_1s, df_point9s = simple_sys.simulate_system_nn_controls(nn_model)

            # Store the best result of this run if it passes constraints
            run_cost = df_1s["ctg"][0]
            run_constraint = df_1s["path_diff"].max()
            if run_cost < best_cost_in_round and run_constraint <= 0:
                best_cost_in_round = run_cost

            simple_sys.plot(df_1s, num_rounds=rp_round+1, num_run_in_round=run+1)
            run_trajectories.append(df_point9s)

        # Decide whether to push out old memories
        if trajectory_df.shape[0] >= buffer_capacity * 10:
            # Get replace the 6 oldest trajectories with new data (60 rows at a time)
            forgotten_trajectories_count = forgotten_trajectories_count % buffer_capacity
            row_slice_start = forgotten_trajectories_count * 10
            row_slice_end = row_slice_start + 60

            df_temp_concat = pd.DataFrame(columns=trajectory_df.columns.tolist())
            for df_temp in run_trajectories:
                df_temp_concat = pd.concat([df_temp_concat, df_temp])

            trajectory_df.iloc[row_slice_start:row_slice_end] = df_temp_concat.iloc[0:60]
            print(trajectory_df)
            forgotten_trajectories_count += 6

        else:
            for df_temp in run_trajectories:
                trajectory_df = pd.concat([trajectory_df, df_temp])

        trajectory_df_filename = "R{} ".format(rp_round+1) + og_trajectory_df_filename
        trajectory_df.to_csv(results_folder + trajectory_df_filename)
        pickle_filename = train_and_pickle(rp_round, results_folder + trajectory_df_filename)

        # If best cost in round is better than current running best cost, add it to dictionary
        if len(best_cost_after_n_rounds) == 0:
            best_cost_after_n_rounds[rp_round] = best_cost_in_round
        else:
            best_key = min(best_cost_after_n_rounds, key=best_cost_after_n_rounds.get)
            if best_cost_in_round < best_cost_after_n_rounds[best_key]:
                best_cost_after_n_rounds[rp_round] = best_cost_in_round
            else:
                best_cost_after_n_rounds[rp_round] = best_cost_after_n_rounds[best_key]

        # Plot best cost against rounds
        # Unindent it to plot once, plotting every round and savings just in case anything fails
        plt.plot(*zip(*sorted(best_cost_after_n_rounds.items())))
        plt.title("Best cost obtained after each round")
        plt.xlabel("Number of rounds")
        plt.ylabel("Best cost obtained")
        plot_filename = results_folder + "best_cost_plot.svg"
        plt.savefig(fname=plot_filename, format="svg")
        # plt.show()
        plt.close()

        # Save the best cost against rounds as a csv
        best_cost_csv_filename = results_folder + "best_cost_plot.csv"
        with open(best_cost_csv_filename, "w") as csv_file:
            writer = csv.writer(csv_file)
            for k, v in best_cost_after_n_rounds.items():
                writer.writerow([k, v])

    return


if __name__ == "__main__":
    # generate_trajectories(save_csv=True)

    # main_simple_sys = SimpleSimulator()
    # main_nn_model = load_pickle("simple_nn_controller.pickle")
    # main_res, _ = main_simple_sys.simulate_system_nn_controls(main_nn_model)
    # main_simple_sys.plot(main_res)
    # print(main_res)

    replay("simple_120_trajectories_df.csv")
