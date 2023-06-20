# Jesse Windle <jesse@bayesfactor.net>, 2023

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import grad, jit
from jaxopt import ScipyBoundedMinimize
from functools import partial
from scipy.stats import chi2
import scipy as sp
from datetime import datetime


class GmmBase(ABC):

    def __init__(self, bounds, num_moments, x):
        # Param
        self.bounds = bounds
        self.num_params = len(bounds[0])
        self.num_moments = num_moments
        self.df = self.num_moments - self.num_params
        self.I = jnp.identity(self.num_moments)
        self.x = x
        self.n_x = x.shape[0]
        # States
        self.W = jnp.identity(self.num_moments)
        self.S = jnp.identity(self.num_moments)
        self.theta = None
        self.jstat = None
        self.pval = None

    # PART I: simple case --- fixed weight matrix.

    @abstractmethod
    def loss_components(self, theta):
        pass
    
    # <https://github.com/google/jax/issues/1251>.  See footnote.
    @partial(jit, static_argnums=(0,))
    def loss(self, theta, W):
        lc = self.loss_components(theta)
        return self.n_x * jnp.dot(jnp.matmul(W, lc), lc)
    
    def opt(self, theta0, W, tol=1e-8, maxiter=100):
        # self.W = W
        lbfgsb = ScipyBoundedMinimize(fun = self.loss, method = "l-bfgs-b", tol = tol, maxiter = maxiter)
        out = lbfgsb.run(theta0, self.bounds, W)
        self.theta = out.params
        self.jstat = self.j_stat(out)
        self.pval = self.p_value(out)
        return out

    def opt_I(self, theta0, tol=1e-8, maxiter=100):
        return self.opt(theta0, self.I, tol, maxiter)

    def j_stat(self, opt_out):
        return opt_out.state.fun_val

    def p_value(self, opt_out):
        """Compute p-value using the J statistic"""
        return 1 - chi2.cdf(x = opt_out.state.fun_val, df=self.df)

    # Part II; iterative estimator --- estimate parameters and weight
    # matrix iteratively --- *a more efficient estimator*.
    
    @abstractmethod
    def loss_components_expanded(self, theta):
        pass

    def avar(self, theta):
        """Asymptotic variance estimator for IID data """
        Z = self.loss_components_expanded(theta)
        # return jnp.cov(jnp.transpose(Z))
        nr = Z.shape[0]
        return jnp.matmul(jnp.transpose(Z), Z) / nr

    def update_theta(self, tol=1e-8, maxiter=100, theta0=None):
        if theta0 is None:
            theta0 = self.theta
        out = self.opt(theta0, self.W, tol=tol, maxiter=maxiter)
        return out

    def update_W(self):
        if self.theta is None:
            raise Exception("You must set theta, e.g. call set_state")
        self.S = self.avar(self.theta)
        self.W = sp.linalg.solve(self.S, self.I, assume_a = "pos")
        return self.W

    def update_both(self, n=1, verbose=False):
        trace = []
        for i in range(0, n):
            if verbose:
                print(f"[{datetime.now()}]: Iteration {i+1}")
            W = self.update_W()
            out = self.update_theta()
            trace.append(out)
            # print((out.params, self.pval))
        return trace



# Notes

# It seems that we either have to use CONSTANT values from the class
# when passing a method to opt that has been jit compiled, like
# self.n_x or we have to pass things that change to the function, so
# that it behaves like a pure function.  Previously, I tried setting x
# multiple times for different data, but it seemed that the first x I
# set was the one which was used throughout, leading to the prior
# conclusions.
