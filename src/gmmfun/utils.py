# Jesse Windle <jesse@bayesfactor.net>, 2023

import jax.numpy as jnp
from jax import grad, jit
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


def central_moment_functions(mgf, n):
    first_moment = Partial(grad(mgf), 0.0)
    cmgf = lambda t, theta: jnp.exp(-t*first_moment(theta)) * mgf(t, theta)
    cmgf_deriv = [grad(cmgf)]
    for i in range(1, n):
        cmgf_deriv.append(grad(cmgf_deriv[i-1]))
    central_moment_functions = [Partial(fun, 0.0) for fun in cmgf_deriv]
    return central_moment_functions


def standardized_moment_functions(mgf, n):
    first_moment = Partial(grad(mgf), 0.0)
    second_moment = Partial(grad(grad(mgf)), 0.0)
    inv_scale = lambda theta: jnp.power(second_moment(theta) - first_moment(theta)**2, -0.5)
    smgf = lambda t, theta: jnp.exp(-t*first_moment(theta)*inv_scale(theta)) * mgf(t * inv_scale(theta), theta)
    smgf_deriv = [grad(smgf)]
    for i in range(1, n):
        smgf_deriv.append(grad(smgf_deriv[i-1]))
    std_moment_functions = [Partial(fun, 0.0) for fun in smgf_deriv]
    return std_moment_functions



def mgf_avar(moment_functions, theta):
    """Assumes

    E[(x^i - f_i)(x^j - f_j)] = E[x^(i+j)] - f_i * f_j
    
    """
    # TODO: add check that this is even
    n_moments = int(len(moment_functions) / 2)
    S0 = np.zeros((n_moments, n_moments))
    moments = [fun(theta) for fun in moment_functions]
    # Evaluate all at theta and then
    for i in range(n_moments):
        for j in range(i,n_moments):
            EX_ij = moments[(i+j+1)]
            mu_ij = moments[i] * moments[j]
            S0[i,j] = S0[j,i] = EX_ij - mu_ij
    return S0



def mgf_norm(t, theta):
    return jnp.exp(t * theta[0] + 0.5 * jnp.square(theta[1] * t))


def cgf_norm(t, theta):
    return t * theta[0] + 0.5 * jnp.square(theta[1] * t)


def mgf_gamma(t, theta):
    return jnp.power(1 - theta[1] * t, -theta[0])

