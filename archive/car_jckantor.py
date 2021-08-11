import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import *
from pyomo.dae import *

# parameters
ar_max = 2.8
av_max = 2.8
phi_max = 0.7
v_max  = 30
v_min = -4

# time and length scales
L = 5

# create a model object
m = ConcreteModel()

# define the independent variable
m.tf = Var(domain=NonNegativeReals)
m.t = ContinuousSet(bounds=(0, 1))

# define control inputs
m.av = Var(m.t)
m.phi = Var(m.t, bounds=(-phi_max, phi_max))

# define the dependent variables
m.x = Var(m.t)
m.y = Var(m.t)
m.a = Var(m.t)
m.v = Var(m.t)

# define derivatives
m.x_dot = DerivativeVar(m.x)
m.y_dot = DerivativeVar(m.y)
m.a_dot = DerivativeVar(m.a)
m.v_dot = DerivativeVar(m.v)

# define the differential equation as constrainta
m.ode_x = Constraint(m.t, rule=lambda m, t: m.x_dot[t] == m.v[t]*cos(m.a[t]))
m.ode_y = Constraint(m.t, rule=lambda m, t: m.y_dot[t] == m.v[t]*sin(m.a[t]))
m.ode_a = Constraint(m.t, rule=lambda m, t: m.a_dot[t] == m.v[t]*tan(m.phi[t])/L)
m.ode_v = Constraint(m.t, rule=lambda m, t: m.v_dot[t] == m.av[t])

# path constraints
m.path_x1 = Constraint(m.t, rule=lambda m, t: m.x[t] >= 0)
m.path_y1 = Constraint(m.t, rule=lambda m, t: m.y[t] >= 0)
m.path_v1 = Constraint(m.t, rule=lambda m, t: m.v[t] <= m.tf*v_max/L)
m.path_v2 = Constraint(m.t, rule=lambda m, t: m.v[t] >= m.tf*v_min/L)
m.path_a1 = Constraint(m.t, rule=lambda m, t: m.av[t] <= m.tf**2*av_max/L)
m.path_a2 = Constraint(m.t, rule=lambda m, t: m.av[t] >= -m.tf**2*av_max/L)
m.path_a3 = Constraint(m.t, rule=lambda m, t: m.v[t]**2*sin(m.phi[t]) <= m.tf**2*ar_max/L)
m.path_a4 = Constraint(m.t, rule=lambda m, t: m.v[t]**2*sin(m.phi[t]) >= -m.tf**2*ar_max/L)

# initial conditions
m.pc = ConstraintList()
m.pc.add(m.x[0]==0)
m.pc.add(m.y[0]==0)
m.pc.add(m.a[0]==0)
m.pc.add(m.v[0]==0)

# final conditions
m.pc.add(m.x[1]==0)
m.pc.add(m.y[1]==4)
m.pc.add(m.a[1]==0)
m.pc.add(m.v[1]==0)

# final conditions on the control inputs
m.pc.add(m.av[1]==0)
m.pc.add(m.phi[1]==0)

# define the optimization objective
m.integral = Integral(m.t, wrt=m.t, rule=lambda m, t: m.av[t]**2 + (m.v[t]**2*sin(m.phi[t]))**2)
m.obj = Objective(expr= m.tf + L**2*m.integral/m.tf**3)

# transform and solve
TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=30)
SolverFactory('ipopt').solve(m).write()
m.display()


def plot_results(t, x, y, a, v, av, phi):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    ax[0].plot(t, av, t, v ** 2 * np.sin(phi) / L)
    ax[0].legend(['Acceleration', 'Lateral Acceleration'])

    ax[1].plot(t, phi, t, a)
    ax[1].legend(['Wheel Position', 'Car Direction'])

    ax[2].plot(t, v)
    ax[2].legend(['Velocity'])
    ax[2].set_ylabel('m/s')
    for axes in ax:
        axes.grid(True)

    plt.show()



# access the results
t = np.array([t * m.tf() for t in m.t])

av = np.array([m.av[t]() * L / (m.tf() ** 2) for t in m.t])
phi = np.array([m.phi[t]() for t in m.t])

x = np.array([m.x[t]() * L for t in m.t])
y = np.array([m.y[t]() * L for t in m.t])
a = np.array([m.a[t]() for t in m.t])
v = np.array([m.v[t]() * L / m.tf() for t in m.t])

ar = v ** 2 * np.sin(phi) / L

plot_results(t, x, y, a, v, av, phi)

scl=0.2


def draw_car(x=0, y=0, a=0, phi=0):
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    car = np.array([[0.2, 0.5], [-0.2, 0.5], [0, 0.5], [0, -0.5],
                    [0.2, -0.5], [-0.2, -0.5], [0, -0.5], [0, 0], [L, 0], [L, 0.5],
                    [L + 0.2 * np.cos(phi), 0.5 + 0.2 * np.sin(phi)],
                    [L - 0.2 * np.cos(phi), 0.5 - 0.2 * np.sin(phi)], [L, 0.5], [L, -0.5],
                    [L + 0.2 * np.cos(phi), -0.5 + 0.2 * np.sin(phi)],
                    [L - 0.2 * np.cos(phi), -0.5 - 0.2 * np.sin(phi)]])
    carz = scl * R.dot(car.T)
    plt.plot(x + carz[0], y + carz[1], 'k', lw=2)
    plt.plot(x, y, 'k.', ms=10)


plt.figure(figsize=(10, 10))
for xs, ys, ts, ps in zip(x, y, a, phi):
    draw_car(xs, ys, ts, scl * ps)
plt.plot(x, y, 'r--', lw=0.8)
plt.axis('square')
plt.grid(True)

plt.show()