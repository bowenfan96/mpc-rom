import scipy.optimize
import numpy as np
import pickle

import pymoo
import leap_ec

import deap
import deap.base
import deap.tools
import deap.algorithms





class GdOpt:
    def __init__(self, mor_nn):
        print("Gradient descent optimizer")
        # Pass neural net for evaluation to the optimizer
        self.mor_nn = mor_nn

    def scipy_opt(self, x_k):
        """
        Gradient descent optimizer using scipy
        :param x_k: x_k_tilde parameters, fixed
        :return: optimal u_k_tilde parameters
        """
        init_guess = np.zeros(10) # u.size
        scipy.optimize.minimize(
            fun=self.nn_func, x0=init_guess, args=x_k, method='BFGS'
        )

        return

    def nn_func(self, u_k, *x_k):
        return 1


class DeapOpt():
    def __init__(self, x_k_init, mor_nn):
        self.x_k = x_k_init
        self.toolbox = deap.base.Toolbox()
        print("Evolutionary algorithm optimizer")
        # Pass neural net for evaluation to the optimizer
        self.mor_nn = mor_nn

        self.toolbox.register("mate", deap.tools.cxTwoPoint)
        self.toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", deap.tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate, x_k=self.x_k)

    def evaluate(self, indivdual, x_k):
        return 2


class NnController:
    def __init__(self, x_k_init, optimizer='gd'):
        # Load pickle neural net
        with open('mor_nn.pickle', 'rb') as model:
            self.mor_nn = pickle.load(model)
        if optimizer == 'gd':
            opt = GdOpt()
        elif optimizer == 'deap':
            opt = DeapOpt()

        # Initial reduced state variables, x_k_tilde
        self.x_k = x_k_init

