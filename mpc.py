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
        self.duration = duration

        self.A = np.genfromtxt(a_csv, delimiter=',')
        self.B = np.genfromtxt(b_csv, delimiter=',')
        # A should be a square matrix
        assert self.A.ndim == 2 and self.A.shape[0] == self.A.shape[1]
        # A and B should have same number of rows
        assert self.B.shape[0] == self.A.shape[0]

        self.x = np.genfromtxt(xi_csv, delimiter=',')
        assert self.x.shape[0] == self.A.shape[0]
        self.x.flatten()

        # Initialize pyomo model
        self.model = ConcreteModel()

        self.model.time = ContinuousSet(bounds=(0, duration))

        self.model.I = RangeSet(0, self.A.shape[1]-1)
        self.model.J = RangeSet(0, self.B.shape[1]-1)
        self.model.x = Var(self.model.I, self.model.time, initialize=0)
        self.model.x_dot = DerivativeVar(self.model.x, wrt=self.model.time, initialize=0)

        self.model.u = Var(self.model.I, self.model.time, initialize=0)

        self.discretizer = TransformationFactory('dae.finite_difference')
        self.discretizer.apply_to(self.model, nfe=int(duration), wrt=self.model.time, scheme='BACKWARD')

        self.model.display()

        # Define derivative variables
        def ode_Ax(i):
            return sum((self.model.x[i, time] * self.A[i][j]) for j in range(self.A.shape[1]))

        def ode_Bu(i):
            return sum((self.model.u[i, time] * self.B[i][j]) for j in range(self.B.shape[1]))

        self.model.ode = ConstraintList()
        for time in self.model.time:
            for i in range(self.A.shape[0]):
                print(time, i)
                self.model.ode.add(
                    self.model.x_dot[i, time] == ode_Ax(i) + ode_Bu(i)
                )
                # Fix variables based on initial values
                self.model.x[i, 0].fix(self.x[i])

        # Objective: Bring to zero
        self.model.obj = Objective(
            expr=summation(self.model.x),
            sense=minimize
        )

    def solve(self):
        opt = SolverFactory('glpk', tee=True)

        mpc_state = []
        sys_state = []
        mpc_action = []

        sys = system.System(self.xi_csv, self.a_csv, self.b_csv)

        for time in self.model.time:
            # MPC solver
            opt.solve(self.model)
            mpc_state.append(self.model.x[:, time])

            self.x = sys.simulate(1)
            self.x.flatten()
            print(self.x)
            sys_state.append(self.x)

            for i in self.model.I:
                print(i)
                print(time)
                self.model.x[i, time].fix(self.x[0][i])

        return mpc_state, sys_state, mpc_action

    @staticmethod
    def plot(mpc_state, sys_state, mpc_action):
        for i in range(mpc_state.shape[1]):
            plt.plot(mpc_state[:, i], label='x{}'.format(i))

        plt.xlabel("Time")
        plt.legend()
        plt.savefig("mpc_plot.svg", format="svg")
        plt.show()


if __name__ == "__main__":
    mpc = MPC("xi.csv", "A.csv", "B.csv", 100)
    mpc_state, sys_state, mpc_action = mpc.solve()
    mpc.plot(mpc_state, sys_state, mpc_action)






# for time in model.time:
#     opt.solve(model)
#
#     current_state = [value(model.x1[time]), value(model.x2[time]), value(model.x3[time])]
#     controls = [value(model.u[time])]
#     new_state = system.step_system(current_state, time, controls, 1)
#
#     mpc_x1_plot.append(value(model.x1[time]))
#     mpc_x2_plot.append(value(model.x2[time]))
#     mpc_x3_plot.append(value(model.x3[time]))
#     mpc_u_plot.append(value(model.u[time]))
#
#     if time == 0:
#         sys_x1_plot.append(value(model.x1[time]))
#         sys_x2_plot.append(value(model.x2[time]))
#         sys_x3_plot.append(value(model.x3[time]))
#
#     elif time < len(model.time):
#         sys_x1_plot.append(new_state[0])
#         sys_x2_plot.append(new_state[1])
#         sys_x3_plot.append(new_state[2])
#
#     # Record system and controller state and fix new state from simulated system
#     # step = 1
#     # if time < len(model.time):
#     #     model.x1[time + step].fix(new_state[0])
#     #     model.x2[time + step].fix(new_state[1])
#     #     model.x3[time + step].fix(new_state[2])
#
#     print('Time: ', time)
#     print("MPC")
#     print("x1: ", value(model.x1[time]), "\tx2: ", value(model.x2[time]),
#           "\tx3: ", value(model.x3[time]), "\tu: ", value(model.u[time]))
#     print("System")
#     print("x1: ", sys_x1_plot[int(time)], "\tx2: ", sys_x2_plot[int(time)], "\tx3", sys_x3_plot[int(time)])
#
# plt.plot(model.time, mpc_x1_plot, label='mpc_x1')
# plt.plot(model.time, mpc_x2_plot, label='mpc_x2')
# plt.plot(model.time, mpc_x3_plot, label='mpc_x3')
# plt.plot(model.time, mpc_u_plot, label='mpc_u')
#
# plt.plot(model.time, sys_x1_plot, label='sys_x1')
# plt.plot(model.time, sys_x2_plot, label='sys_x2')
# plt.plot(model.time, sys_x3_plot, label='sys_x3')
#
# plt.xlabel("Time")
# plt.legend()
# plt.savefig("plot.svg", format="svg")
# plt.show()
#
# system_data = pd.DataFrame(zip(sys_x1_plot, sys_x2_plot, sys_x3_plot))
# system_data.to_csv("data.csv")
