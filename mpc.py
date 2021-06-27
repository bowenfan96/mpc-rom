# Import Pyomo classes
from pyomo.environ import *
from pyomo.dae import *
from pyomo.solvers import *

# Import system file with real dynamics
import system

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# Define a concrete model with hardcoded dynamics
model = ConcreteModel()
# Define the set of continuous time
model.time = ContinuousSet(bounds=(0, 100))
assert model.time.last() == 100

# Define system variables
model.x1 = Var(model.time, initialize=10)
model.x2 = Var(model.time, initialize=0)
model.x3 = Var(model.time, initialize=0)
model.u = Var(model.time, initialize=0)

# Define derivative variables
model.dx1dt = DerivativeVar(model.x1, wrt=model.time, initialize=0)
model.dx2dt = DerivativeVar(model.x2, wrt=model.time, initialize=0)
model.dx3dt = DerivativeVar(model.x3, wrt=model.time, initialize=0)

discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(model, nfe=model.time.last(), wrt=model.time, scheme='BACKWARD')

# ODEs and constraints
# Flawed model used in MPC
# dx1dt = -1.5x1 + 0.9x2 + x3 + 1.2u
# dx2dt = 1.2x1 - 0.8x2
# dx3dt = -0.8x3 + 0.20x1^2

# True model used in system file
# dx1dt = -2x1 + x2 + x3 + u
# dx2dt = x1 - x2
# dx3dt = -x3 + 0.3x1^2

# def _setPoint(m, t):
#     # if t > 20:
#     #     return m.x1[t] == 0.02*t**2 - 0.5*t
#     if t == 100:
#         return m.x1[t] == 10
#     else:
#         return Constraint.Skip
model.x1[0].fix(10)
model.x2[0].fix(0)
model.x3[0].fix(0)
model.u[0].fix(0)

def _ode1(m, t):
    # For testing to check that the control is working (next line only)
    # return m.dx1dt[t] == 0.05 * m.u[t]

    return m.dx1dt[t] == 2*m.x1[t] + m.x2[t] + m.x3[t] + m.u[t]

def _ode2(m, t):
    return m.dx2dt[t] == m.x1[t] - m.x2[t]

def _ode3(m, t):
    return m.dx3dt[t] == -m.x3[t] - 0.2*m.x1[t]**2

# model.c1 = Constraint(model.time, rule=_setPoint)
model.c2 = Constraint(model.time, rule=_ode1)
model.c3 = Constraint(model.time, rule=_ode2)
model.c4 = Constraint(model.time, rule=_ode3)

# Controller constraint
def _u_constraint(m, t):
    return abs(m.u[t]) <= 0.5
model.c5 = Constraint(model.time, rule=_u_constraint)

# Define objective function
# Objective: Least control cost
# model.obj = Objective(
#     expr=sum((0.5 * model.u[t] ** 2) for t in model.time),
#     sense=minimize
# )

# Objective: Least error from setpoint
model.obj = Objective(
    expr=sum((model.x1[int(t)] - 0)**2 for t in model.time),
    sense=minimize
)

opt = SolverFactory('ipopt', tee=True)
opt.options['max_iter'] = 1000

mpc_x1_plot = []
mpc_x2_plot = []
mpc_x3_plot = []
mpc_u_plot = []

sys_x1_plot = []
sys_x2_plot = []
sys_x3_plot = []

sys_dx1dt_plot = []
sys_dx2dt_plot = []
sys_dx3dt_plot = []

for time in model.time:
    opt.solve(model)

    current_state = [value(model.x1[time]), value(model.x2[time]), value(model.x3[time])]
    controls = [value(model.u[time])]
    new_state = system.step_system(current_state, time, controls, 1)

    mpc_x1_plot.append(value(model.x1[time]))
    mpc_x2_plot.append(value(model.x2[time]))
    mpc_x3_plot.append(value(model.x3[time]))
    mpc_u_plot.append(value(model.u[time]))

    if time == 0:
        sys_x1_plot.append(value(model.x1[time]))
        sys_x2_plot.append(value(model.x2[time]))
        sys_x3_plot.append(value(model.x3[time]))

    elif time < len(model.time):
        sys_x1_plot.append(new_state[0])
        sys_x2_plot.append(new_state[1])
        sys_x3_plot.append(new_state[2])

    # Record system and controller state and fix new state from simulated system
    # step = 1
    # if time < len(model.time):
    #     model.x1[time + step].fix(new_state[0])
    #     model.x2[time + step].fix(new_state[1])
    #     model.x3[time + step].fix(new_state[2])

    print('Time: ', time)
    print("MPC")
    print("x1: ", value(model.x1[time]), "\tx2: ", value(model.x2[time]),
          "\tx3: ", value(model.x3[time]), "\tu: ", value(model.u[time]))
    print("System")
    print("x1: ", sys_x1_plot[int(time)], "\tx2: ", sys_x2_plot[int(time)], "\tx3", sys_x3_plot[int(time)])

plt.plot(model.time, mpc_x1_plot, label='mpc_x1')
plt.plot(model.time, mpc_x2_plot, label='mpc_x2')
plt.plot(model.time, mpc_x3_plot, label='mpc_x3')
plt.plot(model.time, mpc_u_plot, label='mpc_u')

plt.plot(model.time, sys_x1_plot, label='sys_x1')
plt.plot(model.time, sys_x2_plot, label='sys_x2')
plt.plot(model.time, sys_x3_plot, label='sys_x3')

plt.xlabel("Time")
plt.legend()
plt.savefig("plot.svg", format="svg")
plt.show()

system_data = pd.DataFrame(zip(sys_x1_plot, sys_x2_plot, sys_x3_plot))
system_data.to_csv("data.csv")
