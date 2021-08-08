import random

import numpy
import scipy.optimize
import numpy as np

import pickle
# Import mor_nn namespace for pickle to work
import mor_nn
from mor_nn import *

import pymoo
import leap_ec

from deap import base, creator, tools


class GdOpt:
    def __init__(self, mor_nn):
        print("Gradient descent optimizer")
        # Pass neural unet for evaluation to the optimizer
        self.mor_nn = mor_nn

    def scipy_opt(self, x_k):
        """
        Gradient descent optimizer using scipy
        :param x_k: x_k_tilde parameters, fixed
        :return: optimal u_k_tilde parameters
        """
        init_guess = np.random.randint(low=0, high=1000, size=1)  # u_tilde.size

        gd_options = {}
        gd_options["maxiter"] = 1000
        gd_options["disp"] = True
        gd_options["eps"] = 10

        result = scipy.optimize.minimize(
            fun=self.nn_func, x0=init_guess, args=x_k,
            options=gd_options
        )

        return result["x"]

    def nn_func(self, u_k, *x_k):
        """
        Predict cost to go using neural unet
        :param u_k: u_k_tilde, the reduced control variables (to be found)
        :param x_k: x_k_tilde, the reduced state variables
        :return: Cost to go, predicted using the neural unet
        """
        x_tilde = x_k[0]
        x_tilde = x_tilde.flatten()
        ctg_pred = self.mor_nn.predict_ctg(x_tilde, u_k)

        print("nn_ctg_pred")
        print(ctg_pred)

        return ctg_pred


class DeapOpt():
    def __init__(self, x_k_init, mor_nn):
        self.x_k = x_k_init
        self.toolbox = base.Toolbox()

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Number of genes in an individual (dimension of u_tilde)
        IND_SIZE = 2
        self.toolbox.register("attribute", random.random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attribute, n=IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        print("Evolutionary algorithm optimizer")
        # Pass neural unet for evaluation to the optimizer
        self.mor_nn = mor_nn

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate, x_k=self.x_k)

    def evaluate(self, individual, x_k):
        """
        Predict cost to go using neural unet
        :param individual: u_k_tilde, the reduced control variables (to be found)
        :param x_k: x_k_tilde, the reduced state variables
        :return:
        """
        ctg_pred = self.mor_nn.predict_ctg(x_k, individual)
        return ctg_pred

    def optimize(self):
        pop = self.toolbox.population(n=100)
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

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
        print("Best individuals")
        best_ind = tools.selBest(pop, 1)[0]
        print(best_ind)

        return best_ind


class NnController:
    def __init__(self, x_k_init=None, optimizer='gd'):
        # Load pickle neural unet
        with open('mor_nn.pickle', 'rb') as model:
            self.mor_nn = pickle.load(model)

        # Initial reduced state variables, x_k_tilde (CAN REMOVE I THINK)
        self.x_k = x_k_init

        self.optimizer = optimizer
        if self.optimizer == 'gd':
            self.opt = GdOpt(self.mor_nn)
        elif self.optimizer == 'deap':
            x_k = np.zeros(14)
            self.opt = DeapOpt(x_k, self.mor_nn)

    def get_controls(self, x_full):
        """
        Get best control action given full state variables x
        :param x_full: Full set of state variables
        :return: Optimal control actions in full controller variables
        """
        # 1. Get x_tilde from x_full by passing through the neural unet
        x_tilde = self.mor_nn.encode_x(x_full)
        x_tilde = x_tilde.flatten()
        print(x_tilde)
        # 2. Get optimal controls given x_tilde
        if self.optimizer == 'gd':
            u_tilde_opt = self.opt.scipy_opt(x_tilde)
        elif self.optimizer == 'deap':
            u_tilde_opt = self.opt.optimize()

        u_tilde_opt = numpy.array(u_tilde_opt)
        print("Hi im u tilde opt")
        print(u_tilde_opt)

        # 3. Decode optimal u_tilde into full set of controls
        u_full_opt = self.mor_nn.decode_u(u_tilde_opt)
        print("U FULL OPT")
        print(u_full_opt)
        return u_full_opt


if __name__ == "__main__":
    controller = NnController()

    print("RESULTS")
    print(controller.get_controls(np.random.randint(low=-1000, high=1000, size=200).reshape(1, 200)))
