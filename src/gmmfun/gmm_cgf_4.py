# Jesse Windle <jesse@bayesfactor.net>, 2023

import jax.numpy as jnp
from jax import grad
from jax.tree_util import Partial
from .gmm_base import GmmBase
from .utils import first_four_moments


class GmmCgf4(GmmBase):

    def __init__(self, cgf, bounds):
        super().__init__(bounds, 4)
        self.cgf = cgf
        self.c1f = grad(self.cgf)
        self.c2f = grad(self.c1f)
        self.c3f = grad(self.c2f)
        self.c4f = grad(self.c3f)
        self.c1p = Partial(self.c1f, 0.0)
        self.c2p = Partial(self.c2f, 0.0)
        self.c3p = Partial(self.c3f, 0.0)
        self.c4p = Partial(self.c4f, 0.0)

    def set_other_data_for_loss(self, x):
        self.moments = first_four_moments(x)
        
    def loss_components(self, theta):
        x = self.moments
        k1 = self.c1p(theta)
        k2 = self.c2p(theta)
        k3 = self.c3p(theta)
        k4 = self.c4p(theta)
        return jnp.array([
            x[0] - k1,
            (x[1] - 2*x[0]*k1 + k1**2) - k2,
            (x[2] - 3*x[1]* k1 + 3*x[0]* k1**2 - k1**3)  - k3,
            (x[3] - 4*x[2]* k1 + 6*x[1]* k1**2 - 4*x[0]*k1**3 + k1**4) - (k4 + 3 * k2**2)
        ])

    def loss_components_expanded(self, theta):
        k1 = self.c1p(theta)
        k2 = self.c2p(theta)
        k3 = self.c3p(theta)
        k4 = self.c4p(theta)
        x_m_k1 = self.x - k1
        return jnp.column_stack([
            x_m_k1,
            x_m_k1**2 - k2,
            x_m_k1**3 - k3,
            x_m_k1**4 - 3 * k2**2 - k4
        ])        
