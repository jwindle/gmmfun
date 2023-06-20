# Jesse Windle <jesse@bayesfactor.net>, 2023

import pytest
import numpy as np
from scipy.stats import norm, gamma
from gmmfun import GmmMgf4, GmmCgf4, GmmMgf, GmmMgfCen, GmmMgfCenScl, GmmMgfStd, GmmMgfAvar
from gmmfun.utils import mgf_norm, cgf_norm


np.random.seed(12345)


@pytest.fixture(scope="module")
def norm_init():
    theta = np.array([2.0, 1.0])
    theta0 = np.array([1.5, 0.8])
    lower_bounds = np.array([-np.inf, 0.0])
    upper_bounds = np.array([ np.inf, np.inf])
    bounds = (lower_bounds, upper_bounds)
    x = norm.rvs(theta[0], theta[1], size=100)
    return dict(x=x, bounds=bounds, theta=theta, theta0=theta0, mgf=mgf_norm, cgf=cgf_norm)


# @pytest.fixture(scope="module")
# def gamma_init():
#     theta = np.array([2.0, 1.0])
#     lower_bounds = np.array([0.0, 0.0])
#     upper_bounds = np.array([ np.inf, np.inf])
#     bounds = (lower_bounds_norm, upper_bounds_norm)
#     x = gamma.rvs(a=theta[0], scale=theta[1], size=100)
#     return (x, bounds, theta)


@pytest.fixture(scope="module")
def mgf_4(norm_init):
    mod = GmmMgf4(norm_init['mgf'], norm_init['bounds'], norm_init['x'])
    return (mod, norm_init)


@pytest.fixture(scope="module")
def cgf_4(norm_init):
    mod = GmmCgf4(norm_init['cgf'], norm_init['bounds'], norm_init['x'])
    return (mod, norm_init)


@pytest.fixture(scope="module")
def mgf(norm_init):
    mod = GmmMgf(norm_init['mgf'], norm_init['bounds'], 4, norm_init['x'])
    return (mod, norm_init)


@pytest.fixture(scope="module")
def mgf_cen(norm_init):
    mod = GmmMgfCen(norm_init['mgf'], norm_init['bounds'], 4, norm_init['x'])
    return (mod, norm_init)


@pytest.fixture(scope="module")
def mgf_cen_scl(norm_init):
    mod = GmmMgfCenScl(norm_init['mgf'], norm_init['bounds'], 4, norm_init['x'])
    return (mod, norm_init)

@pytest.fixture(scope="module")
def mgf_std(norm_init):
    mod = GmmMgfStd(norm_init['mgf'], norm_init['bounds'], 4, norm_init['x'])
    return (mod, norm_init)


@pytest.fixture(scope="module")
def mgf_avar(norm_init):
    mod = GmmMgfAvar(norm_init['mgf'], norm_init['bounds'], 4, norm_init['x'])
    return (mod, norm_init)


# @pytest.fixture(scope="function")
# def all_models_and_data(norm_init):
#     models = dict(
#         mgf_4 = GmmMgf4(norm_init['mgf'], norm_init['bounds'], norm_init['x']),
#         cgf_4 = GmmCgf4(norm_init['cgf'], norm_init['bounds'], norm_init['x']),
#         mgf = GmmMgf(norm_init['mgf'], norm_init['bounds'], 4, norm_init['x']),
#         mgf_cen = GmmMgfCen(norm_init['mgf'], norm_init['bounds'], 4, norm_init['x']),
#         mgf_cen_scl = GmmMgfCenScl(norm_init['mgf'], norm_init['bounds'], 4, norm_init['x'])
#         mgf_std = GmmMgfStd(norm_init['mgf'], norm_init['bounds'], 4, norm_init['x'])
#     )
#     return (models, norm_init)


@pytest.fixture(scope="module", params=["mgf_4", "cgf_4", "mgf", "mgf_cen", "mgf_cen_scl", "mgf_avar"])
def model_and_data(request):
    return request.getfixturevalue(request.param)


# @pytest.fixture(scope="module", params=["mgf_4", "mgf", "mgf_cen", "mgf_cen_scl", "mgf_avar"])
@pytest.fixture(scope="module", params=["mgf_avar"])
def model_pair_and_data(request):
    mod1, data = request.getfixturevalue(request.param)
    mod0 = GmmCgf4(data['cgf'], data['bounds'], data['x'])
    return (mod0, mod1, data)




