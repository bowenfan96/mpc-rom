# Import Pyomo classes
from pyomo.environ import *
from pyomo.dae import *
from pyomo.solvers import *

# Import system file with real dynamics
import system

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import time

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
        assert self.x.shape[0] == self.A.shape[0]
        self.x.flatten()

        # Initialize pyomo model
        self.model = ConcreteModel()

        self.model.time = ContinuousSet(bounds=(0, self.duration))

        self.model.I = RangeSet(0, self.A.shape[1]-1)
        self.model.J = RangeSet(0, self.B.shape[1]-1)
        self.model.x = Var(self.model.I, self.model.time, initialize=0)
        self.model.x_dot = DerivativeVar(self.model.x, wrt=self.model.time, initialize=0)

        self.model.u = Var(self.model.J, self.model.time, initialize=0)

        self.discretizer = TransformationFactory('dae.finite_difference')
        self.discretizer.apply_to(self.model, nfe=int(self.duration), wrt=self.model.time, scheme='BACKWARD')

        # Define derivative variables
        def ode_Ax(m, i, t):
            return sum((m.x[j, t] * self.A[i][j]) for j in range(self.A.shape[1]))

        def ode_Bu(m, i, t):
            return sum((m.u[j, t] * self.B[i][j]) for j in range(self.B.shape[1]))

        self.model.ode = ConstraintList()
        for time in self.model.time:
            for i in range(self.A.shape[0]):
                print(time, i)
                self.model.ode.add(
                    self.model.x_dot[i, time] == ode_Ax(self.model, i, time) + ode_Bu(self.model, i, time)
                )
                # Fix variables based on initial values
                self.model.x[i, 0].fix(self.x[i])

        self.model.ode.display()

        # Objective: Bring to zero
        self.model.obj = Objective(
            expr=sum((self.model.x[0, t] - 100)**2 for t in self.model.time),
            sense=minimize
        )

    def solve(self):
        opt = SolverFactory('ipopt', tee=True)

        mpc_state = []
        sys_state = []
        mpc_action = []

        sys = system.System(self.xi_csv, self.a_csv, self.b_csv)

        for time in self.model.time:
            # MPC solver
            opt.solve(self.model)
            mpc_state.append(list(value(self.model.x[:, time])))

            # Send these controls to system
            controls = list(value(self.model.u[:, time]))

            mpc_action.append(controls)

            self.model.display()

            self.x = sys.simulate(duration=1, controls=controls)
            self.x.flatten()
            sys_state.append(self.x)

            for i in self.model.I:
                print("Time: {}, x_i: {}".format(time, i))
                self.model.x[i, time].fix(self.x[0][i])

        # Turn lists into numpy arrays
        mpc_state = np.array(mpc_state)

        mpc_action = np.array(mpc_action)

        return mpc_state, sys_state, mpc_action

    @staticmethod
    def plot(mpc_state, sys_state, mpc_action):
        for i in range(len(mpc_state[0])):
            plt.plot(mpc_state[:, i], label='x{}'.format(i))
            plt.plot(mpc_action[:, i], label='u{}'.format(i))

        plt.xlabel("Time")
        plt.legend()
        plt.savefig("mpc_plot.svg", format="svg")
        plt.show()


if __name__ == "__main__":
    mpc = MPC("xi.csv", "A.csv", "B.csv", 5)
    mpc_state, sys_state, mpc_action = mpc.solve()
    mpc.plot(mpc_state, sys_state, mpc_action)
