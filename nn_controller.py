import random

import scipy.optimize
import numpy as np

import pickle
# Import mor_nn namespace for pickle to work
from mor_nn import *

import pymoo
import leap_ec

from deap import base, creator, tools


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
        init_guess = np.zeros(2)  # u_tilde.size

        gd_options = {}
        gd_options["maxiter"] = 1000
        gd_options["disp"] = True

        result = scipy.optimize.minimize(
            fun=self.nn_func, x0=init_guess, args=x_k,
            options=gd_options
        )

        return result

    def nn_func(self, u_k, *x_k):
        """
        Predict cost to go using neural net
        :param u_k: u_k_tilde, the reduced control variables (to be found)
        :param x_k: x_k_tilde, the reduced state variables
        :return: Cost to go, predicted using the neural net
        """
        ctg_pred = self.mor_nn.predict_ctg(x_k[0], u_k)
        return ctg_pred


class DeapOpt():
    def __init__(self, x_k_init, mor_nn):
        self.x_k = x_k_init
        self.toolbox = base.Toolbox()

        # Number of genes in an individual (dimension of u_tilde)
        IND_SIZE = 2
        self.toolbox.register("attribute", random.random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attribute, n=IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        print("Evolutionary algorithm optimizer")
        # Pass neural net for evaluation to the optimizer
        self.mor_nn = mor_nn

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate, x_k=self.x_k)

    def evaluate(self, individual, x_k):
        """
        Predict cost to go using neural net
        :param individual: u_k_tilde, the reduced control variables (to be found)
        :param x_k: x_k_tilde, the reduced state variables
        :return:
        """
        ctg_pred = self.mor_nn.predict_ctg(x_k, individual)
        return ctg_pred

    def optimize(self):
        pop = self.toolbox.population(n=50)
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = map(self.toolbox.clone, offspring)

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        print(pop)
        return pop


class NnController:
    def __init__(self, x_k_init, optimizer='deap'):
        # Load pickle neural net
        with open('mor_nn.pickle', 'rb') as model:
            self.mor_nn = pickle.load(model)

        if optimizer == 'gd':
            self.opt = GdOpt(self.mor_nn)
            results = self.opt.scipy_opt(np.random.rand(2))

            print(results)

        elif optimizer == 'deap':
            x_k = np.zeros(2)
            self.opt = DeapOpt(x_k, self.mor_nn)

        # Initial reduced state variables, x_k_tilde
        self.x_k = x_k_init


if __name__ == "__main__":
    controller = NnController(np.zeros(2))
