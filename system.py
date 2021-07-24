import time

import numpy as np
import pandas as pd
import scipy.integrate
import matplotlib.pyplot as plt
import casadi


class System:
    def __init__(self, xi_csv, a_csv, b_csv, c_csv=None, d_csv=None):
        """
        Initialize a linear system model using A, B, C, D matrices:
        x_dot = Ax + Bu
        y = Cx + Du
        D can be empty
        """
        self.A = np.genfromtxt(a_csv, delimiter=',')
        self.B = np.genfromtxt(b_csv, delimiter=',')
        self.C = np.genfromtxt(c_csv, delimiter=',') if c_csv is not None else None
        self.D = np.genfromtxt(d_csv, delimiter=',') if d_csv is not None else None

        # Initialize system variables x, x_dot, u, y
        # A should be a square matrix
        assert self.A.ndim == 2 and self.A.shape[0] == self.A.shape[1]
        # A and B should have same number of rows
        assert self.B.shape[0] == self.A.shape[0]

        self.x = np.genfromtxt(xi_csv, delimiter=',')
        assert self.x.shape[0] == self.A.shape[0]
        self.x_dot = np.zeros(self.x.shape[0])

        # Same number of controls as the number of columns of B
        self.u = np.zeros(self.B.shape[1])

        if self.C is not None:
            # Same number of outputs as the number of rows of C
            # assert self.C.ndim == 2 and self.x.shape[0] == self.C.shape[1]
            self.y = np.zeros(self.C.shape[0])

        if self.D is not None:
            # assert self.D.ndim == 2 and self.x.shape[0] == self.D.shape[0]
            assert self.u.shape[0] == self.D.shape[1]

    def step_casadi(self, controls=None):
        """
        Simulate the system using Casadi through 1 time step
        :return: New state of the system as a numpy array
        """

        print(self.x.size)

        x = casadi.MX.sym('x', self.x.size)
        A = casadi.DM(self.A)
        B = casadi.DM(self.B)

        # Controls should be a numpy array with same size as number of controllers
        if controls is not None:
            u = controls
        else:
            u = np.zeros(B.shape[1])

        rhs = casadi.plus(casadi.mtimes(A, x), casadi.mtimes(B, u))

        ode = {}
        ode['x'] = x
        ode['ode'] = rhs
        F = casadi.integrator('F', 'cvodes', ode, {'tf': 5})
        res = F(x0=self.x)
        print(res["xf"])

        return self.x

    def step_scipy(self, controls=None):
        """
        Simulate the system using scipy integrator
        :return:
        """
        # Controls should be a numpy array with same size as number of controllers
        u = (controls,) if controls is not None else None

        def model(t, x, u=None):
            if u is not None:
                x_dot = np.add(np.matmul(self.A, x), np.matmul(self.B, u))
                # print(x_dot)
                # time.sleep(1)
                return x_dot
            else:
                x_dot = np.matmul(self.A, x)
                print("Hi")
                # print(x_dot)
                # time.sleep(1)
                return x_dot

        # Model function signature is fun(t, y)
        sys = scipy.integrate.solve_ivp(
            fun=model, t_span=(0, 1), t_eval=[1], y0=self.x, method='RK45', args=u,
            # Force the step size to be 1
            max_step=1, first_step=1, atol=1e99, rtol=1e99
        )
        self.x = np.transpose(sys.y).flatten()
        # print(self.x)
        # time.sleep(2)

        return self.x

    def simulate(self, duration, integrator="casadi", controls=None):
        """
        Simulate the system and plot
        :param duration: Duration (number of time steps)
        :param integrator: "scipy" or "casadi"
        :param controls: Control signals
        :return: Simulated system state over the duration (sst)
        """
        # System state through time
        sst = []
        # sst.append(self.x)
        for time in range(duration):
            if integrator == "scipy":
                self.x = self.step_scipy(controls if controls is not None else None)
            elif integrator == "casadi":
                self.x = self.step_casadi(controls if controls is not None else None)
            sst.append(self.x)

        sst = np.array(sst)
        np.savetxt('sst.csv', sst, delimiter=',')

        return sst

    @staticmethod
    def plot(sst):
        assert sst.ndim == 2
        for i in range(sst.shape[1]):
            plt.plot(sst[:, i], label='x{}'.format(i))

        plt.xlabel("Time")
        plt.legend()
        plt.savefig("sys_plot.svg", format="svg")
        plt.show()


if __name__ == "__main__":
    system = System("xi.csv", "A.csv", "B.csv", "C.csv")
    results = system.simulate(20)
    system.plot(results)
