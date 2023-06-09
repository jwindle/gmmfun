# Introduction

Herein is a simple exercise in 1) estimating population parameters using the
generalized method of moments (GMM) and 2) using automatic differentiation (AD).

It is common to approximate a distribution using its first two moments, e.g with
a Gaussian.  However, using the GMM is possible to make this approximation using
the first, second, and higher moments.  The basic idea is that instead of
fitting the first two moments exactly, one can fit an arbitrary number of moment
conditions using least squares (or weighted least squares).

A challenge to this is determining what the higher moments of the approximating
distribution are given a certain parameterization.  However, using AD this is
trivial.

AD has come to prominence via the fitting of neural nets.  Briefly, AD is not
numerical differentiation (i.e. approximating a derivative).  Rather, by knowing
the symbolic derivatives of common functions and how to represent more
complicated functions using those pieces, it is possible to create the computer
equivalent of a complicated function's symbolic derivative.

We use that here along with the moment generating function to compute the
moments of a distribution for a given parameter value.

The code is all very simple and succinct and can be found in the `src`
directory.


# To test code

After first downloading this repo, you should create a virtual environment and
install the required packages:

```
python -m venv venv-gmmfun
source venv-gmmfun/bin/activate
pip install -r requirements
```

Given the structure of our repo, before running tests for the first time, you
need to do
```
export PYTHONPATH="$PWD/src"
```

Now you can run the tests by just typing:
```
pytest
```

# Jupyter notebook

A more detailed description and example can be found in the accompanying Jupyter
notebook.  To launch Jupyter, do

```
jupyter lab --no-browser
```

