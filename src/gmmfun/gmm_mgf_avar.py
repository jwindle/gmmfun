# Jesse Windle <jesse@bayesfactor.net>, 2023

import numpy as np
import scipy as sp
import jax.numpy as jnp
from jax import grad
from jax.tree_util import Partial
# from .gmm_base_avar import GmmBaseMgfAvar
from .gmm_base import GmmBase
from . import utils


class GmmMgfAvar(GmmBase):

    def __init__(self, mgf, bounds, num_moments, x):
        super().__init__(bounds, num_moments, x)
        # Functions
        self.moment_functions_2 = utils.moment_functions(mgf, 2*num_moments)
        self.moment_functions = [fun for fun in self.moment_functions_2[0:num_moments]]
        # Data
        self.x_poly = jnp.column_stack([x**n for n in range(1, self.num_moments+1)])
        self.x_moments = jnp.mean(self.x_poly, axis=0)

    def loss_components(self, theta):
        mgf_moments = jnp.array([fun(theta) for fun in self.moment_functions])
        return (self.x_moments - mgf_moments)

    def loss_components_expanded(self, theta):
        mgf_moments = jnp.array([fun(theta) for fun in self.moment_functions])
        return (self.x_poly - mgf_moments.reshape((1, self.num_moments)))

    def avar(self, theta):
        S = utils.mgf_avar(self.moment_functions_2, theta)
        return S

