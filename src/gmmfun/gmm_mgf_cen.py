# Jesse Windle <jesse@bayesfactor.net>, 2023


import numpy as np
import scipy as sp
import jax.numpy as jnp
from jax import grad
from jax.tree_util import Partial
from .gmm_base import GmmBase
from . import utils


class GmmMgfCen(GmmBase):

    def __init__(self, mgf, bounds, num_moments, x):
        super().__init__(bounds, num_moments, x)
        # Param
        self.binom_coef = list()
        for n in range(1, num_moments+1):
            self.binom_coef.append(np.zeros((n,)))
            for k in range(1, n+1):
                self.binom_coef[n-1][k-1] = sp.special.comb(n, k) * (-1)**k
        # Functions
        self.first_moment = Partial(grad(mgf), 0.0)
        self.central_moment_functions = utils.central_moment_functions(mgf, num_moments)
        # Data
        self.x_poly_N1 = np.column_stack([x**n for n in range(self.num_moments, -1, -1)])
        self.x_poly_1N = np.flip(self.x_poly_N1, axis=1)
        self.moments_N1 = np.mean(self.x_poly_N1, axis=0)
        self.moments_1N = np.flip(self.moments_N1)
        self.a = np.zeros((self.num_moments, self.num_moments))
        self.A = np.zeros((self.n_x, self.num_moments, self.num_moments))
        for n in range(1, self.num_moments+1):
            idc = range(self.num_moments + 1 - n, self.num_moments + 1)
            b = self.binom_coef[n-1]
            self.A[:, n-1, 0:n] = np.matmul(self.x_poly_N1[:, idc], np.diag(b))
            self.a[n-1, 0:n] = self.moments_N1[idc] * b

    def loss_components(self, theta):
        central_moments = jnp.array([self.central_moment_functions[i](theta) for i in range(0, self.num_moments)])
        mu1 = self.first_moment(theta)
        mu_poly = jnp.array([mu1**k for k in range(1, self.num_moments+1)])
        return (jnp.matmul(self.a, mu_poly) + self.moments_1N[1:] - central_moments)

    def loss_components_expanded(self, theta):
        central_moments = jnp.array([self.central_moment_functions[i](theta) for i in range(0, self.num_moments)])
        mu1 = self.first_moment(theta)
        mu_poly = jnp.array([mu1**k for k in range(1, self.num_moments+1)])
        # This is like what we do when using cumulant generating functions, so
        # these two options should be identical.
        # x_m_mu1 = self.x - mu1
        # return jnp.column_stack([x_m_mu1**(k+1) - central_moments[k] for k in range(0, self.num_moments)])
        return (jnp.matmul(self.A, mu_poly) + self.x_poly_1N[:,1:] - central_moments.reshape((1, self.num_moments)))

