# Jesse Windle <jesse@bayesfactor.net>, 2023

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import grad, jit
from jaxopt import ScipyBoundedMinimize
from functools import partial
from scipy.stats import chi2
import scipy as sp


class GmmBase(ABC):

    def __init__(self, bounds, num_moments):
        # Param
        self.bounds = bounds
        self.num_params = len(bounds[0])
        self.num_moments = num_moments
        self.df = self.num_moments - self.num_params
        self.I = jnp.identity(self.num_moments)
        # States
        self.W = jnp.identity(self.num_moments)
        self.theta = None
        self.pval = None

    # PART I: simplest case --- fixed weight matrix, I.  Methods from here to p_value.

    def set_data(self, x):
        # Assume data is matrix or vector?
        self.x = x
        self.n_x = x.shape[0]
        self.set_other_data_for_loss(x)
    
    @abstractmethod
    def set_other_data_for_loss(self, x):
        pass

    @abstractmethod
    def loss_components(self, theta):
        pass
    
    # https://github.com/google/jax/issues/1251
    @partial(jit, static_argnums=(0,))
    def loss_I(self, theta):
        return self.n_x * jnp.sum(jnp.square(self.loss_components(theta)))

    def opt_I(self, theta0, tol=1e-8, maxiter=100, x=None):
        self.W = self.I
        if x:
            self.set_data(x)
        lbfgsb = ScipyBoundedMinimize(fun = self.loss_I, method = "l-bfgs-b", tol = tol, maxiter = maxiter)
        out = lbfgsb.run(theta0, self.bounds)
        self.theta = out.params
        self.pval = self.p_value(out)
        return out

    def p_value(self, opt_out):
        """Compute p-value using the J statistic"""
        return 1 - chi2.cdf(x = opt_out.state.fun_val, df=self.df)

    
    ## Part II: For general weight matrix, W.
    
    @partial(jit, static_argnums=(0,))
    def loss(self, theta, W):
        lc = self.loss_components(theta)
        return self.n_x * jnp.dot(jnp.matmul(W, lc), lc)
    
    def opt(self, theta0, W, tol=1e-8, maxiter=100, x=None):
        self.W = W
        if x:
            self.set_data(x)
        lbfgsb = ScipyBoundedMinimize(fun = self.loss, method = "l-bfgs-b", tol = tol, maxiter = maxiter)
        out = lbfgsb.run(theta0, self.bounds, W)
        self.theta = out.params
        self.pval = self.p_value(out)
        return out
    
    
    # Part III; iterative estimator --- estimate parameters and weight
    # matrix iteratively --- *a more efficient estimator*.
    
    @abstractmethod
    def loss_components_expanded(self, theta):
        pass

    # def set_state(self, theta, W=None):
    #     if W:
    #         self.W = W
    #     else:
    #         self.W = jnp.identity(self.num_moments)
    #     self.theta = theta

    def avar(self, theta):
        """Asymptotic variance estimator for IID data """
        Z = self.loss_components_expanded(theta)
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
        S = self.avar(self.theta)
        self.W = sp.linalg.solve(S, self.I, assume_a = "pos")
        return self.W

    def update_both(self, n=1):
        trace = []
        for i in range(0, n):
            W = self.update_W()
            out = self.update_theta()
            trace.append(out)
            # print((out.params, self.pval))
        return trace
