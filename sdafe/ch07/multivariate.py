from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult
import scipy.stats as stats


def loglik_mvt(Y: pd.DataFrame, par: np.ndarray) -> float:
    """The log-likelihood of the multivariate normal distribution

    Parameters
    ----------
    Y: pd.DataFrame
        the input data: rows are observations, columns are variables
    par: np.ndarray
        the parameters of the multivariate t-distribution

    Returns
    -------
    float
        the log-likelihood of the multivariate t-distribution evaluated at the given point


    Notes
    -----
    The parameters are passed as a one-dimensional array: first the means,
    then the elements of the upper-triangular matrix A, where A.T @ A ~ Cov(Y),
    followed by the degrees of freedom.
    """
    n = Y.shape[1]  # the number of variables
    n_mu = n
    n_cov = n * (n + 1) // 2  # upper-triangular elements of an n-by-n matrix

    # the means of the multivariate t are in the first `n_mu` elements
    mu = par[:n_mu]
    # the upper triangular elements of A are in the following `n_cov` elements
    A = np.zeros((n, n))
    A[np.triu_indices(n)] = par[n_mu:n_mu + n_cov]
    # the scale matrix
    scale = A.T @ A
    # the degrees of freedom value is the last element of `par`
    df = par[n_mu + n_cov]
    return -np.sum(np.log(stats.multivariate_t.pdf(Y, loc=mu, shape=scale, df=df)))


def fit_mvt(
        Y: pd.DataFrame,
        mu_bounds: Tuple[float, float],
        a_bounds: Tuple[float, float],
        df_bounds: Tuple[float, float]
) -> OptimizeResult:
    """Fit a multivariate t-distribution using maximum likelihood

    Parameters
    ----------
    Y: pd.DataFrame
        the input data: rows are observations, columns are variables
    mu_bounds: Tuple[float, float]
        the lower and upper bounds for means
    a_bounds: Tuple[float, float]
        the lower and upper bounds for values in the matrix A, where A.T @ A ~ Cov(Y)
    df_bounds: Tuple[float, float]
        the lower and upper bounds for the degree of freedom parameter

    Returns
    -------
    OptimizeResult
        the result of fitting a multivariate t-distribution
    """
    n = Y.shape[1]  # the number of variables
    n_mu = n
    n_cov = n * (n + 1) // 2  # upper-triangular elements of an n-by-n matrix

    # the starting values for the search
    A = np.linalg.cholesky(Y.cov()).T
    start = np.concatenate([Y.mean().values, A[np.triu_indices(n)], [4]])

    # the bounds for the search
    lower = np.array([mu_bounds[0]] * n_mu + [a_bounds[0]] * n_cov + [df_bounds[0]])
    upper = np.array([mu_bounds[1]] * n_mu + [a_bounds[1]] * n_cov + [df_bounds[1]])

    def objective_func(beta):
        return loglik_mvt(Y, beta)

    bounds = list(zip(lower, upper))

    return minimize(objective_func, start, method='L-BFGS-B', bounds=bounds)
