import numpy as np
from scipy.stats import multivariate_normal
from base import exp_map, unvect_p


def sample_hyperbolic_wg(n, p, mu, Sigma):
    t = multivariate_normal.rvs(mean=mu, cov=Sigma, size=n)
    # return unvect_p(t, p)
    return exp_map(unvect_p(t, p), p)
