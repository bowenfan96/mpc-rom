# Import Pyomo classes
from pyomo.environ import *
from pyomo.dae import *
from pyomo.solvers import *

# Import system file with real dynamics
import system

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class MPC:
    def __init__(self, xi_csv, a_csv, b_csv, duration):
        """
        x_dot = Ax + Bu
        :param xi_csv: Initial system state
        :param a_csv: Matrix A as a csv file
        :param b_csv: Matrix B as a csv file
        :param duration: Number of time steps
        """

        self.xi_csv = xi_csv
        self.a_csv = a_csv
        self.b_csv = b_csv
        self.duration = duration-1

        self.A = np.genfromtxt(a_csv, delimiter=',')
        self.B = np.genfromtxt(b_csv, delimiter=',')
        # A should be a square matrix
        assert self.A.ndim == 2 and self.A.shape[0] == self.A.shape[1]
        # A and B should have same number of rows
        assert self.B.shape[0] == self.A.shape[0]

        self.x = np.genfromtxt(xi_csv, delimiter=',')
        assert self.x.ndim == 1
        assert self.x.shape[0] == self.A.shape[0]

        # Initialize pyomo model
        self.model = ConcreteModel()

        self.model.time = ContinuousSet(bounds=(0, self.duration))

        self.model.I = RangeSet(0, self.A.shape[1]-1)
        self.model.J = RangeSet(0, self.B.shape[1]-1)
        self.model.x = Var(self.model.I, self.model.time, initialize=0)
        self.model.x_dot = DerivativeVar(self.model.x, wrt=self.model.time, initialize=0)

        self.model.u = Var(self.model.J, self.model.time, initialize=0)

        self.discretizer = TransformationFactory('dae.collocation')
        self.discretizer.apply_to(self.model, wrt=self.model.time, scheme='LAGRANGE-RADAU')

        # Define derivative variables
        def ode_Ax(m, i, t):
            return sum((m.x[j, t] * self.A[i][j]) for j in range(self.A.shape[1]))

        def ode_Bu(m, i, t):
            return sum((m.u[j, t] * self.B[i][j]) for j in range(self.B.shape[1]))

        self.model.ode = ConstraintList()
        for time in self.model.time:
            for i in range(self.A.shape[0]):
                self.model.ode.add(
                    self.model.x_dot[i, time] == ode_Ax(self.model, i, time) + ode_Bu(self.model, i, time)
                )
                # Fix variables based on initial values
                self.model.x[i, 0].fix(self.x[i])

        # Objective: Bring the entire system to zero
        # sum(abs(value(self.model.x[i, time])) for i in self.model.I)
        def obj_rule(m):
            return sum(abs(m.x[j]) for j in m.I * m.time)

        self.model.obj = Objective(
            rule=obj_rule,
            sense=minimize
        )

    def solve(self, sim_sys=True):
        opt = SolverFactory('ipopt', tee=True)
        results = None

        mpc_state = []
        sys_state = []
        mpc_action = []
        obj = []

        if sim_sys:
            sys = system.System(self.xi_csv, self.a_csv, self.b_csv)

            for time in self.model.time:
                # MPC solver
                print("Solving...")
                results = opt.solve(self.model)
                mpc_state.append(list(value(self.model.x[:, time])))

                # Send these controls to system
                controls = list(value(self.model.u[:, time]))

                mpc_action.append(controls)

                self.x = np.array(sys.simulate(duration=1, controls=controls))
                self.x = self.x.flatten()

                # self.model.display()
                print(self.x)
                sys_state.append(self.x)

                for i in self.model.I:
                    print("Time: {}, x_{}: {}".format(time, i, self.x[i]))
                    self.model.x[i, time].fix(self.x[i])

        else:
            results = opt.solve(self.model)
            self.model.display()
            for time in self.model.time:
                mpc_state.append(list(value(self.model.x[:, time])))
                mpc_action.append(list(value(self.model.u[:, time])))

                # sum(abs(m.x[j]) for j in m.I * m.time)
                print("Cost function")
                print(sum(abs(value(self.model.x[i, time])) for i in self.model.I))

                obj.append(sum(abs(value(self.model.x[i, time])) for i in self.model.I))

        # Output error if solution cannot be found
        print(results.solver.status)

        # Turn lists into numpy arrays
        mpc_state = np.array(mpc_state)
        mpc_action = np.array(mpc_action)
        sys_state = np.array(sys_state)
        obj = np.array(obj)

        print(obj)

        return mpc_state, sys_state, mpc_action, obj

    def plot(self, mpc_state, sys_state, mpc_action):
        for i in range(len(mpc_state[0])):
            plt.plot(mpc_state[:, i], label='mpc_x{}'.format(i))
            # plt.plot(sys_state[:, i], label='sys_x{}'.format(i))
        # for j in range(len(mpc_action[0])):
        #     plt.plot(mpc_action[:, j], label='u{}'.format(j))

        plt.xlabel("Time")
        plt.xticks(range(mpc_state.shape[0]))
        plt.legend()
        plt.savefig("mpc_plot.svg", format="svg")
        plt.show()


if __name__ == "__main__":
    mpc = MPC("xi.csv", "A.csv", "B.csv", 5)
    mpc_state, sys_state, mpc_action, obj = mpc.solve(sim_sys=False)
    mpc.plot(mpc_state, sys_state, mpc_action)

    np.savetxt("mpc_state.csv", mpc_state, delimiter=",")
    np.savetxt("sys_state.csv", sys_state, delimiter=",")

    data_export = []
    for time in mpc.model.time:
        data_export.append(
            np.hstack(
                # value(mpc.model.x_dot[:, time])
                (value(mpc.model.x[:, time]), value(mpc.model.u[:, time]))
            )
        )
    data_export = np.array(data_export)
    print("OBJ")
    print(obj)
    data_export = np.column_stack((data_export, obj))

    print(data_export)
    print(mpc.A.shape[1])

    df_col_names = []
    df_col_names.extend("x_{}".format(i) for i in range(mpc.A.shape[1]))
    df_col_names.extend("u_{}".format(i) for i in range(mpc.B.shape[1]))
    # df_col_names.extend("xdot_{}".format(i) for i in range(mpc.A.shape[1]))
    df_col_names.extend(["obj"])

    print(df_col_names)

    df_export = pd.DataFrame(data_export, columns=df_col_names)

    print(df_export)

    df_export.to_csv("df_export.csv")
