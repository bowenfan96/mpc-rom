import numpy as np
import scipy
from scipy.optimize import minimize
import pickle

from simple_ctg_nn import *



class Basinhopper:
    def __init__(self):
        with open('simple_system_unsorted_data/simple_proper_wctg.pickle', 'rb') as model:
            self.predict_model = pickle.load(model)

    def scipy_opt(self, x):
        """
        Gradient descent optimizer using scipy
        :param x: x_tilde parameters, fixed
        :return: optimal u_k_tilde parameters
        """

        init_guess = np.random.randint(low=-20, high=20, size=1)  # u_tilde.size

        gd_options = {}
        # gd_options["maxiter"] = 1000
        gd_options["disp"] = True
        # gd_options["eps"] = 1

        min_kwargs = {
            "args": x,
            "method": 'nelder-mead',
            "options": gd_options
        }

        result = scipy.optimize.basinhopping(
            func=self.nn_func, x0=[0], minimizer_kwargs=min_kwargs
        )

        # This is the optimal u, don't be confused by the name!
        return result["x"]


    def nn_func(self, u_k, *x_k):
        """
        Predict cost to go using neural unet
        :param u_k: u_k_tilde, the reduced control variables (to be found)
        :param x_k: x_k_tilde, the reduced state variables
        :return: Cost to go, predicted using the neural unet
        """
        x = x_k[0]
        x = np.array(x).flatten().reshape((1,2))
        print(x)
        ctg_pred = self.predict_model.predict_ctg(x, u_k[0])

        print("nn_ctg_pred")
        print(ctg_pred)

        return ctg_pred

# scipy_opt([0, -16])