# Intro

In the `gmmfun` package we explore the generalized method of moments (GMM) using automatic differentiation (AD).  GMM is a likelihood-free way of estimating population parameters.  AD is kind of like symbolic differentiation; it is software that can create the gradient of an expression and evaluate it. We employ [Jax](https://jax.readthedocs.io/en/latest/index.html) for automatic differentiation (AD).  After reading this introduction, I would suggest walking through our example [Jupyter Notebooks](https://github.com/jwindle/gmmfun/tree/main/notebooks) to see how it all works.


# GMM for population parameter estimation

[Eric Zivot](https://faculty.washington.edu/ezivot) has an accessible [summary](https://faculty.washington.edu/ezivot/econ583/gmm.pdf) of GMM.  We recapitulate the essential parts here.

The idea is the following: you have $i = 1, \ldots, K$ functions, or moment conditions, that satisfy 

$$
\mathbb{E}[g_i(X; \theta_0)] = 0, i = 1, \ldots, K
$$

(We will use capital letters like $X$ to denote a random variable and lower case letters like $x$ to denote a realization from the random variables distribution.)

The sample equivalent of this is

$$
\bar g_i(\theta) = \frac{1}{n} \sum_{t=1}^{n} g_i(x_t, \theta) = 0.
$$

Because we may have more moment conditions than parameters, we may not be able to solve this exactly using sampled data.  An obvious thing to do is to  minimize the squared error of these sample moment conditions.  

Because it will be useful later, we actually want to think about minimizing the squared error using a symmetric, positive definite weighting matrix $W$.  That is, we want to minimize

$$
J_n(\theta, W) = n \bar g' W \bar g.
$$

The choice of $W$ can have a dramatic effect on the efficiency of the estimator, but, for any choice of $W$, solving for the $\theta$ that minimizes $J_n$ will produce a consistent estimate as $n \rightarrow \infty$.  The most efficient estimator is when $n W$ is the inverse of the asymptotic variance of $\bar g$ as our data grows without bound, i.e. $n \rightarrow \infty$, under the true value $\theta = \theta_0$.  Assuming we have IID data,

$$
Var[ \frac{1}{n} \sum_{t=1}^{n} g(X_t, \theta) ] = \frac{1}{n} Var[ g(X_1, \theta) ].
$$

Let $\hat S$ be an estimator of the variance term,

$$
\hat S(\theta) = V_n(\theta) = \frac{1}{n} \sum_{i=1}^{n} g(x_i, \theta) g(x_i, \theta)'
$$

and let $\hat W = \hat S^{-1}$.  Then we can iteratively cylce through values of $\hat \theta$ and $\hat S$ by:

$$
\hat \theta = \underset{\theta}{\text{argmin}} \; J_n(\theta, \hat S)
$$

and

$$
\hat S = V_n(\hat \theta)
$$

to arrive at the estimator $\hat \theta$ that arises using the optimal weight matrix, approximately speaking.

Lastly, and critically, when we have $K$ moment conditions and only $L$ parameters, then asymptotically, $J_n$ converges to a $\chi^2_{K-L}$ distribution under the null hypothesis.  Thus, we can use $J_n$ as a statistic to create a p-value:

$$
p = 1 - CDF_{\chi^2}(J_n(\hat \theta, \hat S), \; df=K - L).
$$


# Moment conditions for estimating population parameters

We will be using the moment generating function (MGF) to define our moment conditions.  The MGF is defined as

$$
M(t) = \mathbb{E}[e^{t X}]
$$

and has the property $\mathbb{E}[X^k] = M^{(k)}(0)$ under certain regularity conditions.  Thus, the obvious moment conditions are:

$$
g_i(x, \theta) = x^i - M_\theta^{(i)}(0), i = 1, \ldots, K.
$$

where $M^{(i)}$ is the $i$th derivative of the moment generating function with respect to $t$ and $\theta$ is the parameter vector.  AD makes computing $M^{(i)}$ trivial.

The code is very simple and can be found [here](https://github.com/jwindle/gmmfun).


## Known asymptotic variance

We can actually go a step further here by computing the aysmptotic variance directly.  Suppose we want to compute the covariance between the $i$ and $j$ moment condition.  Letting $\mu_i = M^{(i)}(0)$, $i=1, \ldots, 2K$ we have

$$
\mathbb{E}[(X^i - \mu_i)((X^j - \mu_j)] = 
\mathbb{E}[X^{i+j}] - \mu_i \mu_j = \mu_{i+j} - \mu_i \mu_j.
$$

In other words, for the price of computing not $K$ derivatives, but the first $2K$ derivatives, we can compute the asymptotic variance for a given parameter $\theta$ directly.

In our method of moments discussion above, that allows us to replac $\hat S(\theta)$ above with $S(\theta)$ where

$$
S_{ij}(\theta) = M^{(i+j)}(\theta) - M^{(i)}(\theta) M^{(j)}(\theta),
$$

the exact asymptotic variance for a given value of $\theta$.


# Uses

The motivation for this work arose when considering the distribution of a linear combination of iid random variables, and estimating the underlying parameters governing that distribution.  In particular, suppose one observes $Y = v' X$, where $X$ is a vector of iid random variables and $v$ is a constant vector.  We want to estimate the parameters governing the distribution of the components of $X$.

When working in the Bayesian setting, either one would need to marginalize out $X$ to arrive at a closed form likelihood of the observed, or one would need to generate posterior samples from those latent variables themselves in order to generate estimates of the parameters.  The latter can lead to Markov Chain Monte Carlos samplers with high autocorrelation in the parameter samples.

However, if we know the moment generation function of $X$, then we can use that to easily construct the moment generating function of $Y$.  We can then apply the GMM method outlined above to estimate the parameters of the distribution governing the components of $X$.


# gmmfun package

We have written the `gmmfun` package to implement these methods and a series of notebooks that covers:

  - How to use the package
  - How the asymptotics change if one estimates the aymptotic variannce or computes it directly
  - And how one can use these methods to estimate the parameters of iid $X_i, i = 1, \ldots K$ when one observes the linear combination $Y = v' X$

The repo for the package can be found [here](https://github.com/jwindle/gmmfun).
