import scipy.optimize
import numpy as np
import pickle

import pymoo
import leap_ec
import deap


class GdOpt:
    def __init__(self):
        print("Gradient descent optimizer")

    def nn_func(self, u_k, *x_k):
        return 1

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


class EaOpt:
    def __init__(self):
        print("Evolutionary algorithm optimizer")


class NnController:
    def __init__(self, optimizer='gd'):
        # Load pickle neural net
        with open('mor_nn.pickle', 'rb') as model:
            self.mor_nn = pickle.load(model)
        if optimizer == 'gd':
            opt = GdOpt()
        elif optimizer == 'ea':
            opt = EaOpt()