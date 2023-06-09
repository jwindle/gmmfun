# Jesse Windle <jesse@bayesfactor.net>, 2023

import jax.numpy as jnp
from jax import grad
from jax.tree_util import Partial
import numpy as np


def first_four_moments(X):
    moments = np.mean(np.stack([X, X**2, X**3, X**4]), axis=1)
    return moments


def first_four_cumulants(X):
    mu_hat = np.mean(X)
    sig_hat = np.std(X)
    cumulants = np.array([
        mu_hat,
        sig_hat**2,
        np.mean((X - mu_hat)**3),
        np.mean((X - mu_hat)**4 - 3 * sig_hat**4)
    ])
    return cumulants                   


def polynomial_covariates(x, n):
    return np.column_stack([x**k for k in range(1, n+1)])


def sample_moments(x, n):
    moments = np.mean(polynomial_covariates(x, n), axis=0)
    return moments


def moment_functions(mgf, n):
    mgf_deriv = [grad(mgf)]
    for i in range(1, n):
        mgf_deriv.append(grad(mgf_deriv[i-1]))
    moment_functions = [Partial(fun, 0.0) for fun in mgf_deriv]
    return moment_functions


def mgf_norm(t, theta):
    return jnp.exp(t * theta[0] + 0.5 * jnp.square(theta[1] * t))


def cgf_norm(t, theta):
    return t * theta[0] + 0.5 * jnp.square(theta[1] * t)


def mgf_gamma(t, theta):
    return jnp.power(1 - theta[1] * t, -theta[0])

