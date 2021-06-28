import numpy as np
import scipy.integrate

import matplotlib.pyplot as plt

# dx1dt = -2x1 + x2 + x3 + u
# dx2dt = x1 - x2
# dx3dt = x1 - x3 - x1^2


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


if __name__ == "__main__":
    simulate_and_get_data(100)