# Authors:
# A. del Rio Chanona  - Imperial College, https://www.imperial.ac.uk/people/a.del-rio-chanona
# P. Petsagkourakis   - University College London, https://panos108.github.io/ppetsag/

# Optimisation and Machine Learning for Process Systems Engineering:
# https://www.imperial.ac.uk/optimisation-and-machine-learning-for-process-engineering/about-us/

# Other codes: https://www.imperial.ac.uk/optimisation-and-machine-learning-for-process-engineering/codes/

# To cite please use the publications [1,2,3,4] at the end of the document.


import scipy.integrate as scp
from pyomo.environ import *

from pyomo import environ as po
from pyomo import dae as pod
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd


#################################
# function that defines problem #
#################################

def createmodel(t0, tf, x0, xtf):
    # create the model
    m = po.ConcreteModel()

    # -- define variables -- #
    # define time as the continous variable
    m.t = pod.ContinuousSet(bounds=(t0, tf))
    m.tf = po.Param(initialize=tf)
    # define state variables (dependent on time)
    m.x1 = po.Var(m.t)
    m.u = po.Var(m.t, bounds=(-20, 20))



    m.x2 = po.Var(m.t)
    m.L = po.Var(m.t)
    # definr derivative variebles
    m.dx1dt = pod.DerivativeVar(m.x1, wrt=m.t)
    m.dx2dt = pod.DerivativeVar(m.x2, wrt=m.t)
    m.dLdt = pod.DerivativeVar(m.L, wrt=m.t)

    # -- define differential equations -- #
    # define diff eq 1
    def xdiffeq_rule(m, t):
        return m.dx1dt[t] == 1.004* m.x2[t]

    m.xdiffeq = po.Constraint(m.t, rule=xdiffeq_rule)
    m.x1[t0].fix(x0[0])

    # define diff eq 2
    def odiffeq_rule(m, t):
        return m.dx2dt[t] == -0.3721 - 0.164*m.x1[t] - 2.252*m.x2[t] + 0.588*m.u[t]

    m.odiffeq = po.Constraint(m.t, rule=odiffeq_rule)
    m.x2[t0].fix(x0[1])

    # define path constraint
    def path_con_rule(m, t):
        return m.x2[t] + 0.5 - 8 * (t - 0.5) ** 2 <= 0

    m.path_con = po.Constraint(m.t, rule=path_con_rule)

    # Lagrange term
    def Ldiffeq_rule(m, t):
        return m.dLdt[t] == m.x1[t] ** 2 + m.x2[t] ** 2 + 5 * 10 ** (-3) * m.u[t] ** 2

    m.Ldiffeq = po.Constraint(m.t, rule=Ldiffeq_rule)
    m.L[t0].fix(0)

    # define objective function
    print(tf)

    def objective_rule(m):
        return m.L[m.tf] - m.L[0]

    m.objective = \
        po.Objective(rule=objective_rule, sense=po.minimize)

    return m


##########################################
# function discretization specifications #
##########################################

def discretize(model, fe, cp, cp_in):
    if cp_in > cp:
        print('Warning: Reduced collocation for manipulated cp_in<=cp')
        print('Replaced with cp_in = cp')
        cp_in = cp
    discretizer = po.TransformationFactory('dae.collocation')
    discretizer.apply_to(model, nfe=fe, ncp=cp, scheme='LAGRANGE-RADAU')
    discretizer.reduce_collocation_points(model, var=model.u, ncp=cp_in, \
                                          contset=model.t)

    # model.u[0].fix(17.13485215)
    # model.u[0.1].fix(17.90966723)
    # model.u[0.2].fix(15.69551206)
    # model.u[0.3].fix(12.02073079)
    # model.u[0.4].fix(6.3854598)
    # model.u[0.5].fix(5.78803769)
    # model.u[0.6].fix(5.61547509)
    # model.u[0.7].fix(4.82217782)
    # model.u[0.8].fix(3.66145129)
    # model.u[0.9].fix(2.69432302)
    # model.u[1].fix(3.21533437)

    return discretizer, model


################################
# function that solves problem #
################################

def solvemodel(model, solverofchoice):
    solver = po.SolverFactory(solverofchoice)
    solver.solve(model)

    return model


#####################################
# function for plotting and display #
#####################################

def presentresults(model):
    t = []
    x1 = []
    u = []
    x2 = []
    for time in model.t:
        t.append(time)
        x1.append(po.value(model.x1[time]))
        u.append(po.value(model.u[time]))
        x2.append(po.value(model.x2[time]))

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.plot(t, x1, label='x1 trajectory')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t, u, label='u profile')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(t, x2, label='x2')
    plt.plot(t, -0.5 + 8 * (np.array(t) - 0.5) ** 2, label='path constraint for x2')
    plt.legend()

    plt.show()
    # plt.close()
    print('Value of objective is {0}'.format(po.value(model.L[model.t.last()])))

    return t, x1, x2, u

nfe                = 10
cp_in              = 1

concat_df = pd.DataFrame(columns=["t", "x1", "x2", "u"])

# GENERATE DATA
for _ in range(1):
    x1_init_rng = np.random.uniform(-10, 10)
    x2_init_rng = np.random.uniform(-18.5, 1.5)

    model_i            = createmodel(t0=0, tf=1, x0=[-4.269957139,	-10.90194926], xtf=[])
    discretizer, model = discretize(model_i, nfe, 4, cp_in)

    model              = solvemodel(model_i, 'ipopt')
    model.display()
    print('Number of finite elements: {0}'.format(nfe))
    print('Number of collocation points for manipulated variable (u): {0}'.format(cp_in))
    t, x1, x2, u = presentresults(model_i)

    data_df = pd.DataFrame(
        {"t": t,
         "x1": x1,
         "x2": x2,
         "u": u
        })

    data_fe_intervals = data_df.iloc[::4, :]
    # data_df.drop(index=data_df.iloc[5::4, :].index.tolist(), inplace=True)

    # print(data_fe_intervals)

    concat_df = pd.concat([concat_df, data_fe_intervals])

    print(concat_df)

# concat_df.to_csv("simple_proper.csv", sep=',')
