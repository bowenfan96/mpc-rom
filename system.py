import numpy as np
import scipy.integrate

# dx1dt = -2x1 + x2 + x3 + u
# dx2dt = x1 - x2
# dx3dt = x1 - x3 - x1^2

def step_system(current_state, time, controls, step_len):
    u = controls[0]

    def model(t, state):
        x1 = state[0]
        x2 = state[1]
        x3 = state[2]

        dx1dt = -2*x1 + x2 + x3 + u
        dx2dt = x1 - x2
        dx3dt = -x3 + 0.25*x1**2

        return [dx1dt, dx2dt, dx3dt]

    system = scipy.integrate.ode(model)
    system.set_integrator('lsoda')
    system.set_initial_value(current_state, time)
    new_state = np.array(system.integrate(system.t + step_len))
    return new_state
