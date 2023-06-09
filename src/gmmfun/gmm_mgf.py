# Jesse Windle <jesse@bayesfactor.net>, 2023


import numpy as np
import scipy as sp
import jax.numpy as jnp
from jax import grad
from jax.tree_util import Partial
from .gmm_base import GmmBase
from . import utils


class GmmMgf(GmmBase):

    def __init__(self, mgf, bounds, num_moments, weights=None):
        super().__init__(bounds, num_moments)
        # Param
        if not weights is None:
            self.weights = weights
        else:
            self.weights = jnp.ones((num_moments,))
        # Functions
        self.mgf = mgf
        self.moment_functions = utils.moment_functions(mgf, num_moments)

    def set_other_data_for_loss(self, x):
        self.x_poly = jnp.column_stack([x**n for n in range(1, self.num_moments+1)])
        self.x_moments = jnp.mean(self.x_poly, axis=0)

    def loss_components(self, theta):
        mgf_moments = jnp.array([fun(theta) for fun in self.moment_functions])
        return (self.x_moments - mgf_moments) * self.weights

    def loss_components_expanded(self, theta):
        mgf_moments = jnp.array([fun(theta) for fun in self.moment_functions])
        return (self.x_poly - mgf_moments.reshape((1, self.num_moments))) * self.weights.reshape((1, self.num_moments))

