"""Translation to Python of utilities implemented by Ruppert and Matteson

See https://people.orie.cornell.edu/davidr/SDAFE2/Rscripts/SDAFE2.R
"""
from typing import Tuple

import numpy as np
from scipy.optimize import minimize


def _sigma_step(l: float, t: int, a_prev: np.ndarray, sigma_prev: np.ndarray) -> np.ndarray:
    """The recursive formula for sigma

    Parameters
    ----------
    l: float
        the value of lambda to use
    t: int
        zero-based time index
    a_prev: np.ndarray
        the current column-vector of innovations
    sigma_prev: np.ndarray
        the covariance matrix from the preceding step

    Returns
    -------
    np.ndarray
        the new covariance matrix
    """
    assert sigma_prev.shape[0] == sigma_prev.shape[1]
    assert a_prev.shape == (sigma_prev.shape[0], 1)
    return ((1 - l) * a_prev @ a_prev.T + l * (1 - l ** (t - 1)) * sigma_prev) / (1 - l ** t)


def _llik_norm(x: np.ndarray, sigma: np.ndarray) -> float:
    """Log-likelihood of observing `x` under the normal distribution N(0, `sigma`)

    Parameters
    ----------
    x: np.ndarray
        the column-vector of observations
    sigma: np.ndarray
        the covariance matrix of the multivariate normal distribution

    Returns
    -------
    float
        the log-likelihood of observing `x` under the normal distribution with
        the covariance matrix `sigma`
    """
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)
    return -0.5 * np.log(det_sigma) - 0.5 * x.T @ inv_sigma @ x


def _nllik_ewma(l: float, innov: np.ndarray) -> float:
    """Objective function for maximising log-likelihood

    Parameters
    ----------
    l: float
        the parameter of the EWMA model
    innov: np.ndarray
        an array of innovations: rows are observations, columns are variables

    Returns
    -------
    float
        the negative of log-likelihood, suitable to use in a minimiser
    """
    # we start with the marginal estimate of covariance
    sigma_hat = np.cov(innov, rowvar=False, ddof=1)

    llik = 0.0
    # the log-likelihood contributions for the first two observations
    llik += _llik_norm(innov[0, :, np.newaxis], sigma_hat)
    at = innov[1, :, np.newaxis]
    llik += _llik_norm(at, sigma_hat)

    n = innov.shape[0]  # number of observations
    for t in range(2, n):
        atm1 = at  # previous observation
        at = innov[t, :, np.newaxis]  # current observation
        sigma_hat = _sigma_step(l, t, atm1, sigma_hat)  # evolve sigma
        llik += _llik_norm(at, sigma_hat)

    return -llik  # the objective function is for minimisation, so negate the value to maximise log-likelihood


def est_ewma(l0: float, innov: np.ndarray) -> Tuple[float, float]:
    """Fit the EWMA model to the supplied innovations

    Parameters
    ----------
    l0: float
        the starting value of lambda
    innov: np.ndarray
        an array of innovations: rows are observations, columns are variables

    Returns
    -------
    Tuple[float, float]
        the estimated values of lambda and its squared error
    """
    res = minimize(_nllik_ewma, l0, innov, bounds=[(0.001, 0.999)])
    se = np.sqrt(res.hess_inv.todense()[0, 0])  # standard error estimate based on Fisher information
    return res.x[0], se


def sigma_ewma(l: float, innov: np.ndarray) -> np.ndarray:
    """Use a recursive EWMA process to estimate Sigma[t]

    Parameters
    ----------
    l: float
        the value of lambda to use
    innov: np.ndarray
        an array of innovation time series: rows are observations, columns are variables

    Returns
    -------
    np.ndarray
        an array of estimated covariance matrices at each time, size (p, p, n),
        where n is the number of observations and p is the number of variables
    """
    n, d = innov.shape  # number of observations and variables
    sigma_hat = np.cov(innov, rowvar=False, ddof=1)  # marginal covariance matrix
    sigma_t = np.zeros((d, d, n))  # result matrix to be populated
    sigma_t[:, :, 0:2] = sigma_hat[:, :, np.newaxis]  # initialise the first two values for subsequent recursion
    for t in range(2, n):
        atm1 = innov[t - 1, :, np.newaxis]  # innovations at time t-1
        sigma_t[:, :, t] = _sigma_step(l, t, atm1, sigma_t[:, :, t - 1])
    return sigma_t
