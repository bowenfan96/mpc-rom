# Authors:
# A. del Rio Chanona  - Imperial College, https://www.imperial.ac.uk/people/a.del-rio-chanona
# P. Petsagkourakis   - University College London, https://panos108.github.io/ppetsag/

# Optimisation and Machine Learning for Process Systems Engineering:
# https://www.imperial.ac.uk/optimisation-and-machine-learning-for-process-engineering/about-us/

# Other codes: https://www.imperial.ac.uk/optimisation-and-machine-learning-for-process-engineering/codes/

# To cite please use the publications [1,2,3,4] at the end of the document.

from pyomo import environ as po
from pyomo import dae as pod

# import basinhopping
from simple_ctg_nn import *
from archive import gridsearch


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
        return m.dx1dt[t] == m.x2[t]

    m.xdiffeq = po.Constraint(m.t, rule=xdiffeq_rule)
    m.x1[t0].fix(x0[0])

    # define diff eq 2
    def odiffeq_rule(m, t):
        return m.dx2dt[t] == -m.x2[t] + m.u[t]

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

    # model.u[0].fix(13.66394499)
    # model.u[0.1].fix(14.20041063)
    # model.u[0.2].fix(14.8043678)
    # model.u[0.3].fix(15.22740341)
    # model.u[0.4].fix(15.48928715)
    # model.u[0.5].fix(15.85557914)
    # model.u[0.6].fix(16.18091321)
    # model.u[0.7].fix(16.45137047)
    # model.u[0.8].fix(16.71366667)
    # model.u[0.9].fix(16.96693872)
    # model.u[1].fix(17.21033071)

    # model.u[0].fix(20.70174186)
    # model.u[0.1].fix(20.20688981)
    # model.u[0.2].fix(20.76074092)
    # model.u[0.3].fix(20.51253288)
    # model.u[0.4].fix(10.10454163)
    # model.u[0.5].fix(2.56125446)
    # model.u[0.6].fix(2.43778702)
    # model.u[0.7].fix(2.76538845)
    # model.u[0.8].fix(3.14020152)
    # model.u[0.9].fix(3.40329642)
    # model.u[1].fix(3.60322947)

    # GET RANDOM CONTROLS
    # u_rng = np.random.uniform(low=-20, high=20, size=10)

    # model.u[0.1].fix(u_rng[0])
    # model.u[0.2].fix(u_rng[1])
    # model.u[0.3].fix(u_rng[2])
    # model.u[0.4].fix(u_rng[3])
    # model.u[0.5].fix(u_rng[4])
    # model.u[0.6].fix(u_rng[5])
    # model.u[0.7].fix(u_rng[6])
    # model.u[0.8].fix(u_rng[7])
    # model.u[0.9].fix(u_rng[8])
    # model.u[1].fix(u_rng[9])

    return discretizer, model


################################
# function that solves problem #
################################

def solvemodel(model, solverofchoice):
    solver = po.SolverFactory(solverofchoice)
    results = solver.solve(model)

    return results, model


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

    fig, axs = plt.subplots(3, constrained_layout=True)
    fig.set_size_inches(5, 10)

    axs[0].plot(t, x1, label='$x_1$')
    axs[0].legend()

    axs[1].plot(t, x2, label='$x_2$')
    axs[1].plot(t, -0.5 + 8 * (np.array(t) - 0.5) ** 2, label='Path constraint for $x_2$')
    axs[1].legend()

    axs[2].plot(t, u, label='Neural net controller action')
    axs[2].legend()

    fig.suptitle('Control policy and system state after 20 rounds of training')
    plt.xlabel("Time")
    plt.show()


    # plt.close()
    print('Value of objective is {0}'.format(po.value(model.L[model.t.last()])))
    for time in model.t:
        print(po.value(model.L[time]))

    return t, x1, x2, u

nfe                = 10
cp_in              = 1

model_i = createmodel(t0=0, tf=1, x0=[0, -1], xtf=[])
discretizer, model = discretize(model_i, nfe, 4, cp_in)

# for i in range(11):
#     x1 = po.value(model.x1[i/10])
#     x2 = po.value(model.x2[i/10])
#     # Get controls from basinhopping
#     gs = gridsearch.GridSearch()
#     u_opt = gs.search([x1, x2])
#
#     # ADD SOME NOISE
#     u_opt += np.random.uniform(-1, 1)
#
#     # for time in model.t:
#     #     if i/10 <= time < (i+1)/10:
#     #         model.u[time].fix(u_opt)
#
#     results, model = solvemodel(model_i, 'ipopt')

results, model = solvemodel(model_i, 'ipopt')


model.display()
print('Number of finite elements: {0}'.format(nfe))
print('Number of collocation points for manipulated variable (u): {0}'.format(cp_in))
t, x1, x2, u = presentresults(model_i)

print("Status:")
print(results)

data_df = pd.DataFrame(
    {"t": t,
     "x1": x1,
     "x2": x2,
     "u": u
     })

data_fe_intervals = data_df.iloc[::4, :]
# data_fe_intervals.to_csv("run1.csv", sep=",")
# Append ctg
#
def _cost(cx1, cx2, cu):
    return cx1 ** 2 + cx2 ** 2 + 5E-3 * cu ** 2

cost_to_go = []
data_fe_intervals_np = np.array(data_fe_intervals)
for t in range(data_fe_intervals_np.shape[0]):
    cost_to_go.append(_cost(data_fe_intervals_np[t][1], data_fe_intervals_np[t][2], data_fe_intervals_np[t][3]))
for t in reversed(range(data_fe_intervals_np.shape[0] - 1)):
    cost_to_go[t] += cost_to_go[t + 1]
cost_to_go = np.array(cost_to_go).reshape(-1, 1)

# print(data_fe_intervals_np)
# print(cost_to_go)
# time.sleep(5)

w_ctg = np.hstack((data_fe_intervals_np, cost_to_go))
w_ctg_df = pd.DataFrame(
            {"t": w_ctg[:,0],
             "x1": w_ctg[:,1],
             "x2": w_ctg[:,2],
             "u": w_ctg[:,3],
             "ctg": w_ctg[:,4]
            })

# w_ctg_df.to_csv("run20.csv", sep=',')


# GENERATE DATA
# for _ in range(50):
#     # x1_init_rng = np.random.uniform(-10, 10)
#     # x2_init_rng = np.random.uniform(-18.5, 1.5)
#
#     model_i            = createmodel(t0=0, tf=1, x0=[0,	-1], xtf=[])
#     discretizer, model = discretize(model_i, nfe, 4, cp_in)
#
#     results, model              = solvemodel(model_i, 'ipopt')
#     model.display()
#     print('Number of finite elements: {0}'.format(nfe))
#     print('Number of collocation points for manipulated variable (u): {0}'.format(cp_in))
#     t, x1, x2, u = presentresults(model_i)
#
#     print("Status:")
#     print(results.solver.status)
#
#     # Only record trajectory if no constraint was broken, i.e. solver status = ok
#
#     if results.solver.status == "ok":
#         data_df = pd.DataFrame(
#             {"t": t,
#              "x1": x1,
#              "x2": x2,
#              "u": u
#             })
#
#         data_fe_intervals = data_df.iloc[::4, :]
#         # data_df.drop(index=data_df.iloc[5::4, :].index.tolist(), inplace=True)
#
#         # print(data_fe_intervals)
#         # Append ctg
#
#         def cost(x1, x2, u):
#             return x1 ** 2 + x2 ** 2 + 5E-3 * u ** 2
#
#         cost_to_go = []
#         data_fe_intervals_np = np.array(data_fe_intervals)
#         for t in range(data_fe_intervals_np.shape[0]):
#             cost_to_go.append(cost(data_fe_intervals_np[t][1], data_fe_intervals_np[t][2], data_fe_intervals_np[t][3]))
#         for t in reversed(range(data_fe_intervals_np.shape[0] - 1)):
#             cost_to_go[t] += cost_to_go[t + 1]
#         cost_to_go = np.array(cost_to_go).reshape(-1, 1)
#
#         # print(data_fe_intervals_np)
#         # print(cost_to_go)
#         # time.sleep(5)
#
#         w_ctg = np.hstack((data_fe_intervals_np, cost_to_go))
#
#         # print(w_ctg)
#         # time.sleep(5)
#
#         w_ctg_df = pd.DataFrame(
#             {"t": w_ctg[:,0],
#              "x1": w_ctg[:,1],
#              "x2": w_ctg[:,2],
#              "u": w_ctg[:,3],
#              "ctg": w_ctg[:,4]
#             })
#
#         concat_df = pd.concat([concat_df, w_ctg_df])
#         print(concat_df)
#
#     #
#     elif results.solver.status == "warning":
#         data_df = pd.DataFrame(
#             {"t": t,
#              "x1": x1,
#              "x2": x2,
#              "u": u
#             })
#
#         data_fe_intervals = data_df.iloc[::4, :]
#
#         def cost(x1, x2, u):
#             return 10 * (x1 ** 2 + x2 ** 2 + 5E-3 * u ** 2)
#
#         cost_to_go = []
#         data_fe_intervals_np = np.array(data_fe_intervals)
#         for t in range(data_fe_intervals_np.shape[0]):
#             cost_to_go.append(cost(data_fe_intervals_np[t][1], data_fe_intervals_np[t][2], data_fe_intervals_np[t][3]))
#         for t in reversed(range(data_fe_intervals_np.shape[0] - 1)):
#             cost_to_go[t] += cost_to_go[t + 1]
#         cost_to_go = np.array(cost_to_go).reshape(-1, 1)
#
#         w_ctg = np.hstack((data_fe_intervals_np, cost_to_go))
#
#         w_ctg_df = pd.DataFrame(
#             {"t": w_ctg[:,0],
#              "x1": w_ctg[:,1],
#              "x2": w_ctg[:,2],
#              "u": w_ctg[:,3],
#              "ctg": w_ctg[:,4]
#             })
#
#         warn_concat_df = pd.concat([warn_concat_df, w_ctg_df])
#         print(warn_concat_df)

# concat_df.to_csv("simple_proper_rng_controls_init_fix.csv", sep=',')
# warn_concat_df.to_csv("simple_proper_rng_controls_init_fix_warn.csv", sep=',')