
# Jesse Windle <jesse@bayesfactor.net>, 2023


import jax.numpy as jnp
from jax import grad
from jax.tree_util import Partial
from .gmm_base import GmmBase
from .utils import first_four_moments


class GmmMgf4(GmmBase):

    def __init__(self, mgf, bounds, x):
        super().__init__(bounds, 4, x)
        self.mgf = mgf
        self.m1f = grad(self.mgf)
        self.m2f = grad(self.m1f)
        self.m3f = grad(self.m2f)
        self.m4f = grad(self.m3f)
        self.m1p = Partial(self.m1f, 0.0)
        self.m2p = Partial(self.m2f, 0.0)
        self.m3p = Partial(self.m3f, 0.0)
        self.m4p = Partial(self.m4f, 0.0)
        self.moments = first_four_moments(x)

    def loss_components(self, theta):
        m1 = self.m1p(theta)
        m2 = self.m2p(theta)
        m3 = self.m3p(theta)
        m4 = self.m4p(theta)
        return jnp.array([self.moments[0] - m1, self.moments[1] - m2, self.moments[2] - m3, self.moments[3] - m4])

    def loss_components_expanded(self, theta):
        m1 = self.m1p(theta)
        m2 = self.m2p(theta)
        m3 = self.m3p(theta)
        m4 = self.m4p(theta)
        return jnp.column_stack([self.x - m1, self.x**2 - m2, self.x**3 - m3, self.x**4 - m4])


