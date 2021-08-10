import pickle

import numpy as np

from simple_ctg_nn import *


class GridSearch:
    def __init__(self):
        with open('simple_proper_wctg.pickle', 'rb') as model:
            self.predict_model = pickle.load(model)

    def search(self, x):
        """
        Gradient descent optimizer using scipy
        :param x: x_tilde parameters, fixed
        :return: optimal u_k_tilde parameters
        """

        best_u = 0
        best_ctg = np.inf
        x = np.array(x).flatten().reshape((1, 2))
        for u in np.linspace(start=-20, stop=20, num=81):
            # print(u)
            ctg_pred = self.predict_model.predict_ctg(x, u)
            # print(ctg_pred)
            if ctg_pred < best_ctg:
                best_u = u
                best_ctg = ctg_pred

        return best_u


# searcher = GridSearch()
# print("Best u")
# print(searcher.search([0,1]))