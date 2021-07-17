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
            assert self.C.ndim == 2 and self.x.shape[0] == self.C.shape[1]
            self.y = np.zeros(self.C.shape[0])

        if self.D is not None:
            assert self.D.ndim == 2 and self.x.shape[0] == self.D.shape[0]
            assert self.u.shape[0] == self.D.shape[1]

    def step_casadi(self, controls=None):
        """
        Simulate the system using Casadi through 1 time step
        :return: New state of the system as a numpy array
        """

        x = casadi.MX.sym('x', self.x.size)

        # TODO: fix symbolic generator
        A = casadi.MX(self.A)

        rhs = A*x
        print(rhs)
        print(x.shape)
        print(rhs.shape)

        ode = {'x': x, 'ode': rhs}
        F = casadi.integrator('F', 'cvodes', ode, {'tf': 5})

        res = F(x0=self.x)

        print(res['xf'])

    def step_scipy(self, controls=None):
        """
        Simulate the system using scipy integrator
        :return:
        """
        def model(t, xu):
            if controls is not None:
                print(xu)
                x = xu[0]
                u = xu[1]
                print(np.matmul(self.A, xu))
                print(np.matmul(self.B, xu))
                x_dot = np.add(np.matmul(self.A, x), np.matmul(self.B, u))
                return x_dot
            else:
                x = xu
                x_dot = np.matmul(self.A, x)
                return x_dot

        sys = scipy.integrate.ode(model)
        sys.set_integrator('lsoda')
        sys.set_initial_value(self.x, t=0)

        self.x = np.array(sys.integrate(sys.t + 1))
        return self.x

    def simulate(self, duration, integrator="scipy", controls=None):
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
    # simulate_and_get_data(10)

    system = System("xi.csv", "A.csv", "B.csv", "C.csv")
    system.plot(system.simulate(2))











def step_system(current_state, time, controls, step_len=1):
    u = controls[0]

    def model(t, state):
        x1 = state[0]
        x2 = state[1]
        x3 = state[2]
        x4 = state[3]

        dx1dt = x2
        dx2dt = -x1
        dx3dt = 0.9*x2 + 0.1*x1 + 0.03*x3
        dx4dt = 0.9*x2 + 0.03*x4

        return [dx1dt, dx2dt, dx3dt, dx4dt]

    system = scipy.integrate.ode(model)
    system.set_integrator('lsoda')
    system.set_initial_value(current_state, time)
    new_state = np.array(system.integrate(system.t + step_len))
    return new_state


def simulate_and_get_data(duration):
    x1_i = 1
    x2_i = 0
    x3_i = 0
    x4_i = 0

    current_state = [x1_i, x2_i, x3_i, x4_i]
    controls = [0]

    sys_x1_plot = []
    sys_x2_plot = []
    sys_x3_plot = []
    sys_x4_plot = []

    for time in range(duration):
        sys_x1_plot.append(current_state[0])
        sys_x2_plot.append(current_state[1])
        sys_x3_plot.append(current_state[2])
        sys_x4_plot.append(current_state[3])

        current_state = step_system(current_state, duration, controls, step_len=1)

    plt.plot(range(duration), sys_x1_plot, label='sys_x1')
    plt.plot(range(duration), sys_x2_plot, label='sys_x2')
    plt.plot(range(duration), sys_x3_plot, label='sys_x3')
    plt.plot(range(duration), sys_x4_plot, label='sys_x4')

    plt.xlabel("Time")
    plt.legend()
    plt.savefig("plot.svg", format="svg")
    plt.show()
